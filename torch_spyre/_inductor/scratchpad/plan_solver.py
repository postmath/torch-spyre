# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import heapq
import math


@dataclass
class LifetimeBoundBuffer:
    """
    Defines the data fields required for a plan solver.
    """

    name: str
    size: int
    start_time: int
    end_time: int
    address: Optional[int] = None
    in_place_parents: list[str] = field(default_factory=list)


class MemoryPlanSolver(ABC):
    """
    An abstract class for defining algorithms which solve
    memory layout patterns based on provided sizes, lifetimes.
    """

    def __init__(self, size: int, alignment: int = 128):
        """Initialize the solver with a fixed scratchpad capacity and alignment.

        Args:
            size (int): Total scratchpad size in bytes. Buffers whose aligned
                placement would exceed this limit are evicted (address=None).
            alignment (int): Byte alignment boundary. Every buffer is placed at
                the next address that is a multiple of this value. Defaults to 128
                (one Spyre stick).
        """
        self.limit = size
        self.alignment = alignment
        self.usage: list[LifetimeBoundBuffer] = []

    @abstractmethod
    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """
        Utilizes an implementation defined algorithm to determine
        if and where buffers should be placed in scratchpad memory based
        on their attributes.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of candidate buffers for memory planning

        Returns:
            list[LifetimeBoundBuffer]: The set of buffers with their placements defined.
        """
        pass


class GreedyLayoutSolver(MemoryPlanSolver):
    def _get_lowest_addr_in_use(self):
        return min(
            (rec.address for rec in self.usage if rec.address is not None),
            default=0,
        )

    def _get_highest_addr_in_use(self):
        return max(
            (rec.address + rec.size for rec in self.usage if rec.address is not None),
            default=0,
        )

    def _find_free_block(self, size_needed: int) -> Optional[int]:
        assert all(x.address is not None for x in self.usage)
        curr_lo = self._get_lowest_addr_in_use()
        curr_hi = self._get_highest_addr_in_use()
        if self.limit < size_needed:
            return None

        if not self.usage or curr_lo >= size_needed:
            return 0

        address = math.ceil(curr_hi / self.alignment) * self.alignment
        if address + size_needed <= self.limit:
            return address

        # Search for a gap between existing allocations
        self.usage.sort(key=lambda x: (x.address is None, x.address))
        for i in range(len(self.usage) - 1):
            assert (current_address := self.usage[i].address) is not None
            assert (next_address := self.usage[i + 1].address) is not None
            frag_st = (
                math.ceil((current_address + self.usage[i].size) / self.alignment)
                * self.alignment
            )
            if next_address - frag_st >= size_needed:
                return frag_st

        return None

    def _try_allocate(self, buffer: LifetimeBoundBuffer):
        # Check if the current buffer can be in-placed
        for in_place_opt in buffer.in_place_parents:
            matched_obj = next((u for u in self.usage if u.name == in_place_opt), None)
            if matched_obj is not None and buffer.size <= matched_obj.size:
                buffer.address = matched_obj.address
                self.usage.append(buffer)
                self.usage.remove(matched_obj)
                return None

        # Decide where to allocate the block from
        addr = self._find_free_block(buffer.size)

        # Push the allocation result to the buffer and the usage table
        if addr is not None:
            buffer.address = addr
            self.usage.append(buffer)
        else:
            buffer.address = None

    def _try_deallocate(self, bufs: list[LifetimeBoundBuffer] | LifetimeBoundBuffer):
        if isinstance(bufs, LifetimeBoundBuffer):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                self.usage.remove(buf)

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """Allocates addresses to the provided buffer list

        Accepts a set of buffers with pre-defined sizes and lifetimes. These buffers are
        allocated addresses with 0 -> `limit` where the maximum starting address of
        buffers are at most `self.limit` - `LifetimeBoundBuffer.size` - 1. The algorithm
        increments through logical time where time increments 1 unit for each
        step in a computation graph. At each step the lifetimes of all buffers are
        evaluated for allocation and deallocation based on its lifetime relative
        to the time being evaluated. As an optimization, times where no buffers
        enter or exit scope are not evaluated.

        When a buffer enters scope, the current usage is evaluated in the following
        manner:
            1. Check if there is a permissible in-place buffer already allocated
            2. Is there enough space from address 0 -> first usage.
            3. Is there enough space for the current buffer from the max address
                to the maximum memory address. Allocate as current_max + 1 + alignment.
            4. Is there space between allocations. Check for gaps between current
                allocations and find where gaps exceed current size. Allocate if
                current gap is larger than current size + alignment.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of buffers to be planned.

        Returns:
            list[LifetimeBoundBuffer]: The supplied buffers with addresses assigned.
        """
        if not buffers:
            return []
        assert all(buf.address is None for buf in buffers), (
            "Buffers cannot be previously or partially planned"
        )

        self.usage = []

        # Walk through all transition points in chronological order.
        # Include end_time + 1 so deallocation fires even when no other
        # buffer starts or ends at that tick.
        times = set()
        for b in buffers:
            times.add(b.start_time)
            times.add(b.end_time)
        sorted_times = sorted(times)

        for idx in sorted_times:
            # Deallocate all expired buffers before allocating new ones so that
            # freed slots are immediately available at the same time step.
            for buffer in buffers:
                if idx == buffer.end_time:
                    self._try_deallocate(buffer)

            for buffer in buffers:
                if idx == buffer.start_time:
                    self._try_allocate(buffer)

        return buffers


class CappedAllocatorPlanBase(ABC):
    """Shared state and interface for capacity-bounded allocation plans.

    A plan places a set of :class:`LifetimeBoundBuffer` objects into a
    fixed-capacity scratchpad following a *permutation*: an explicit allocation
    order given as a list of buffer indices. Buffer ``permutation[k]`` is
    allocated on top of every already-placed buffer whose lifetime overlaps it
    (respecting in-place parents), rounded up to ``alignment``.

    Addresses are maintained internally and are **not** written back to the
    buffer objects until :meth:`finalize`. Two buffers that are alive at the
    same logical tick may not occupy overlapping address ranges, with the sole
    exception of an in-place parent/child pair, which may share an identical
    address (``P.end_time == C.start_time``).

    The objective being optimized is :meth:`total_size`: the summed size of
    every buffer that fits *entirely* below ``capacity``. Buffers whose
    placement would cross the capacity line keep their (notional) address for
    ordering purposes but are neither counted nor written back on
    :meth:`finalize`.

    Subclasses implement :meth:`_build` (initial placement) and :meth:`swap`
    (incremental re-placement after exchanging two adjacent permutation
    entries).

    Args:
        buffers: The buffers to place. Indices into this list are the values
            used in ``permutation`` and as keys throughout the plan.
        permutation: Allocation order as a permutation of
            ``range(len(buffers))``.
        capacity: Scratchpad capacity in bytes.
        alignment: Byte alignment boundary for placed addresses. Defaults to 128
            (one Spyre stick).
    """

    def __init__(
        self,
        buffers: list[LifetimeBoundBuffer],
        permutation: list[int],
        capacity: int,
        alignment: int = 128,
    ):
        n = len(buffers)
        assert sorted(permutation) == list(range(n)), (
            "permutation must be a permutation of range(len(buffers))"
        )
        self.buffers = buffers
        self.permutation = list(permutation)
        self.capacity = capacity
        self.alignment = alignment
        self._name_to_idx = {buf.name: i for i, buf in enumerate(buffers)}

        # Internal address per buffer index; None means unplaced. Populated by
        # _build and kept in sync by swap. Not written to buffer objects until
        # finalize.
        self.addresses: list[Optional[int]] = [None] * n

        # Sum of buf.size over all fully-allocated buffers (address + size <=
        # capacity). Maintained incrementally; exposed via total_size().
        self.total_allocated_size: int = 0

        self._build()

    @abstractmethod
    def _build(self) -> None:
        """Compute addresses for every buffer in permutation order.

        Populates ``self.addresses`` and ``self.total_allocated_size`` (and any
        subclass-specific structures). Called once from ``__init__``.
        """
        pass

    @abstractmethod
    def swap(self, i: int) -> int:
        """Swap permutation entries ``i`` and ``i + 1`` and re-place buffers.

        Args:
            i: Position in the permutation; entries ``i`` and ``i + 1`` are
                exchanged.

        Returns:
            The change in :meth:`total_size` caused by the swap (new minus old).
        """
        pass

    # --- shared helpers -----------------------------------------------------

    def _align_up(self, addr: int) -> int:
        """Round ``addr`` up to the next multiple of ``self.alignment``."""
        return math.ceil(addr / self.alignment) * self.alignment

    def _top(self, idx: int) -> int:
        """Return ``address + size`` for a placed buffer (its exclusive top)."""
        addr = self.addresses[idx]
        assert addr is not None, f"buffer {idx} is not placed"
        return addr + self.buffers[idx].size

    def _is_fully_allocated(self, idx: int) -> bool:
        """True if buffer ``idx`` has an address and fits below ``capacity``."""
        addr = self.addresses[idx]
        return addr is not None and addr + self.buffers[idx].size <= self.capacity

    def _overlaps(self, i: int, j: int) -> bool:
        """True if buffers ``i`` and ``j`` are alive at a common tick.

        Lifetimes are closed intervals ``[start_time, end_time]``, so an
        in-place parent and child (``parent.end_time == child.start_time``)
        overlap at exactly that boundary tick.
        """
        a = self.buffers[i]
        b = self.buffers[j]
        return a.start_time <= b.end_time and b.start_time <= a.end_time

    def _in_place_pair(self, i: int, j: int) -> Optional[tuple[int, int]]:
        """Return ``(parent_idx, child_idx)`` if ``i`` and ``j`` form an in-place
        pair, else ``None``.

        The relationship is declared on the child via ``in_place_parents``; it is
        symmetric for placement purposes, so either argument may be the parent.
        """
        bi = self.buffers[i]
        bj = self.buffers[j]
        if bj.name in bi.in_place_parents:
            return (j, i)  # j is the parent of i
        if bi.name in bj.in_place_parents:
            return (i, j)  # i is the parent of j
        return None

    def _can_inplace(self, parent: int, child: int) -> bool:
        """True if ``child`` is allowed to share ``parent``'s address.

        A child may only reuse a parent's storage if it fits within it; a
        larger child would still need the parent's inputs while writing past
        the parent's footprint.
        """
        return self.buffers[child].size <= self.buffers[parent].size

    def _placement_decision(
        self, idx: int, candidates: list[int]
    ) -> tuple[int, Optional[int]]:
        """Decide ``idx``'s address given the buffers it must sit on top of.

        ``candidates`` are already-placed buffer indices that overlap ``idx`` in
        time. For the reference plan these are *all* time-overlapping buffers;
        for the incremental plan they are ``idx``'s direct below-neighbours --
        both yield the same decision, because the highest top among them is the
        same and that is all the rule depends on.

        ``idx`` is placed on top of everything it overlaps. The one exception is
        an in-place partner ``P`` (``P.end_time == idx.start_time`` or vice
        versa): ``idx`` may instead drop into ``P``'s slot, reusing ``P``'s
        address, but *only* when every other overlapping buffer already tops out
        at or below ``P``'s address -- otherwise ``idx`` would land partway into
        occupied space. When that holds, dropping onto ``P`` still leaves ``idx``
        above all the others (it saves ``P``'s footprint rather than stacking on
        top of it).

        Returns:
            ``(address, partner)`` where ``partner`` is the candidate whose
            address was reused in-place, or ``None`` if ``idx`` was stacked.
        """
        if not candidates:
            return 0, None
        max_top = max(self._top(p) for p in candidates)
        # Try to drop into an in-place partner's slot. At most one partner can
        # qualify: if two did, each would have to top out below the other's
        # address, which is impossible.
        for partner in candidates:
            pair = self._in_place_pair(idx, partner)
            if pair is None or not self._can_inplace(*pair):
                continue
            partner_addr = self.addresses[partner]
            assert partner_addr is not None
            others_top = max(
                (self._top(q) for q in candidates if q != partner), default=0
            )
            if others_top <= partner_addr:
                return partner_addr, partner
        return self._align_up(max_top), None

    def _address_from_candidates(self, idx: int, candidates: list[int]) -> int:
        """Return only the address from :meth:`_placement_decision`."""
        return self._placement_decision(idx, candidates)[0]

    def total_size(self) -> int:
        """Total size of all buffers fully allocated below capacity (O(1))."""
        return self.total_allocated_size

    def finalize(self) -> None:
        """Write back addresses of fully-allocated buffers to the buffers.

        Buffers that do not fit entirely below ``capacity`` have their
        ``address`` set to ``None`` and are not committed.
        """
        for idx, buf in enumerate(self.buffers):
            if self._is_fully_allocated(idx):
                buf.address = self.addresses[idx]
            else:
                buf.address = None


class ReferenceCappedAllocatorPlan(CappedAllocatorPlanBase):
    """Simple, obviously-correct O(n^2) reference plan.

    Placement scans all previously-placed, time-overlapping buffers for each
    buffer; ``swap`` mutates the permutation and rebuilds from scratch. Kept as
    a permanent oracle for differential testing against the incremental
    :class:`CappedAllocatorPlan`.
    """

    def _build(self) -> None:
        n = len(self.buffers)
        self.addresses = [None] * n
        self.total_allocated_size = 0
        for pos in range(n):
            idx = self.permutation[pos]
            prior = self.permutation[:pos]
            candidates = [p for p in prior if self._overlaps(idx, p)]
            self.addresses[idx] = self._address_from_candidates(idx, candidates)
            if self._is_fully_allocated(idx):
                self.total_allocated_size += self.buffers[idx].size

    def swap(self, i: int) -> int:
        """Swap permutation entries ``i``/``i+1`` and rebuild from scratch."""
        old_total = self.total_allocated_size
        perm = self.permutation
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        self._build()
        return self.total_allocated_size - old_total


class CappedAllocatorPlan(CappedAllocatorPlanBase):
    """Incremental capacity-bounded allocation plan.

    Maintains a neighbor graph (buffers directly below/above each buffer in
    address space, including air-gap dependencies) so that swapping two adjacent
    permutation entries only re-places the affected buffers via change-driven
    propagation, rather than rebuilding the whole layout.

    Attributes:
        below_neighbors: ``below_neighbors[c]`` is the set of buffer indices
            directly below ``c`` -- those that share a tick with ``c`` and are
            the nearest buffer beneath it at that tick (air gaps included), plus
            any in-place partner whose address ``c`` reuses. ``c``'s address is
            a function of exactly this set.
        above_neighbors: the inverse relation; used to find which buffers may
            need re-placing when ``c``'s top moves.
        inplace_reuse: ``inplace_reuse[x] = y`` when buffer ``x`` reused
            partner ``y``'s address in-place (``x`` was placed at ``y``'s
            address).
    """

    def _build(self) -> None:
        n = len(self.buffers)
        self.addresses = [None] * n
        self.total_allocated_size = 0
        # reuser idx -> reused (partner) idx for placements that went in-place.
        self.inplace_reuse: dict[int, int] = {}
        for pos in range(n):
            idx = self.permutation[pos]
            prior = self.permutation[:pos]
            candidates = [p for p in prior if self._overlaps(idx, p)]
            addr, partner = self._placement_decision(idx, candidates)
            self.addresses[idx] = addr
            if partner is not None:
                self.inplace_reuse[idx] = partner
            if self._is_fully_allocated(idx):
                self.total_allocated_size += self.buffers[idx].size
        # Persistent position index, maintained in O(1) by swap().
        self.position: list[int] = [0] * n
        for p, idx in enumerate(self.permutation):
            self.position[idx] = p
        # In-place partners (parents and children) per buffer; the only buffers
        # that may legally share an address with it. Used by swap() to find the
        # other members of a buffer's column.
        self.inplace_partners: dict[int, set[int]] = {i: set() for i in range(n)}
        for child in range(n):
            for pname in self.buffers[child].in_place_parents:
                parent = self._name_to_idx.get(pname)
                if parent is not None:
                    self.inplace_partners[child].add(parent)
                    self.inplace_partners[parent].add(child)
        # Time-overlap sets. Lifetimes never change, so this is computed once
        # and lets swap() find a buffer's candidates in O(degree) instead of
        # scanning all n buffers.
        self.overlaps: dict[int, set[int]] = {i: set() for i in range(n)}
        for a in range(n):
            for b in range(a + 1, n):
                if self._overlaps(a, b):
                    self.overlaps[a].add(b)
                    self.overlaps[b].add(a)
        self._build_neighbor_graph()

    def _build_neighbor_graph(self) -> None:
        """Populate ``below_neighbors`` / ``above_neighbors`` from the layout.

        ``c``'s direct below-neighbours at a tick are the buffers forming the
        nearest *column* entirely below it: we find the greatest top at or under
        ``c``'s address, then take every buffer sharing that column's address
        (in-place siblings share an address and so share a column). Taking only
        the single nearest buffer would be wrong -- a shorter in-place sibling,
        shielded by a taller one whose top reaches a byte higher, would be
        dropped, yet it still occupies the column beneath ``c`` and so must count
        among the "other" buffers when deciding whether ``c`` may drop into an
        in-place partner's slot. Nothing lies between the column and ``c`` (a
        higher entirely-below buffer contradicts "greatest"; a buffer crossing
        ``c``'s address would overlap ``c`` in time, which the layout forbids),
        so this captures the true adjacency, air gaps included. We union over
        every integer tick: an adjacency can first appear at an interior tick
        where two buffers that previously separated the pair have both died,
        with nothing entering or leaving at that tick.

        In-place partners that share ``c``'s own address are handled separately:
        being level with ``c`` they are not "below" it, so the sweep never links
        them. The explicit ``reused -> reuser`` edge instead records that
        dependency, so the reuser re-derives its address from the partner while
        the reused buffer keeps its own geometric below-neighbours.
        """
        n = len(self.buffers)
        self.below_neighbors: dict[int, set[int]] = {i: set() for i in range(n)}
        self.above_neighbors: dict[int, set[int]] = {i: set() for i in range(n)}

        # Explicit in-place edges: the reuser depends on the reused partner.
        for reuser, reused in self.inplace_reuse.items():
            self.below_neighbors[reuser].add(reused)
            self.above_neighbors[reused].add(reuser)

        if n == 0:
            return

        min_start = min(b.start_time for b in self.buffers)
        max_end = max(b.end_time for b in self.buffers)

        for t in range(min_start, max_end + 1):
            alive = [
                i
                for i in range(n)
                if self.buffers[i].start_time <= t <= self.buffers[i].end_time
            ]
            for c in alive:
                ca = self.addresses[c]
                assert ca is not None
                # Address of the highest column entirely below c at this tick.
                nearest_top = -1
                nearest_addr = None
                for b in alive:
                    if b == c:
                        continue
                    top_b = self._top(b)
                    if top_b <= ca and top_b > nearest_top:
                        nearest_top = top_b
                        nearest_addr = self.addresses[b]
                if nearest_addr is None:
                    continue
                # Link c to every buffer in that column (in-place siblings).
                for b in alive:
                    if b != c and self.addresses[b] == nearest_addr:
                        self.below_neighbors[c].add(b)
                        self.above_neighbors[b].add(c)

    def swap(self, i: int) -> int:
        """Swap permutation entries ``i`` and ``i+1`` and re-place incrementally.

        A no-op when the swapped buffers do not overlap in time. Otherwise the
        work is driven by the neighbour graph instead of by scanning positions:

        - Affected buffers are processed in a min-heap keyed by position.
          Dependencies always point to earlier positions, so a buffer is settled
          before anything resting on it, and ``position`` is maintained in O(1).
        - Each processed buffer is re-derived in full (address *and* below/above
          edges): the cascade can move a buffer into or out of another's column,
          changing neighbour sets well beyond the swapped pair, so an
          address-only update is not enough.
        - When a buffer's address changes we enqueue its ``above_neighbors``
          (the buffers resting on it, which a height change can disturb) and, if
          it reused an in-place partner, also the partner's ``above_neighbors``:
          a buffer joining a partner's column -- the only way two overlapping
          buffers share an address -- adds itself to the below-set of everything
          resting on that column, without the partner itself moving. Only
          later-positioned buffers are notified, since a below-set draws solely
          from earlier positions. This visits only the buffers a swap affects.

        Returns:
            The change in :meth:`total_size` (new minus old).
        """
        n = len(self.buffers)
        assert 0 <= i < n - 1
        perm = self.permutation
        x, y = perm[i], perm[i + 1]
        perm[i], perm[i + 1] = y, x
        self.position[x], self.position[y] = i + 1, i
        if not self._overlaps(x, y):
            # Independent buffers: their order does not affect any address.
            return 0

        old_total = self.total_allocated_size
        seed = {x, y} | self.above_neighbors[x] | self.above_neighbors[y]
        heap = [(self.position[idx], idx) for idx in seed]
        heapq.heapify(heap)
        queued = set(seed)
        while heap:
            _, z = heapq.heappop(heap)
            queued.discard(z)
            old_addr = self.addresses[z]
            if self._is_fully_allocated(z):
                self.total_allocated_size -= self.buffers[z].size
            self._replace_buffer(z, self.position)
            if self._is_fully_allocated(z):
                self.total_allocated_size += self.buffers[z].size
            if self.addresses[z] != old_addr:
                # Resters on z, plus resters on any column z now shares with an
                # in-place partner (z just joined that partner's column).
                notify = set(self.above_neighbors[z])
                for partner in self.inplace_partners[z]:
                    if self.addresses[partner] == self.addresses[z]:
                        notify |= self.above_neighbors[partner]
                pos_z = self.position[z]
                for w in notify:
                    if w not in queued and self.position[w] > pos_z:
                        queued.add(w)
                        heapq.heappush(heap, (self.position[w], w))
        return self.total_allocated_size - old_total

    def _replace_buffer(self, z: int, pos: list[int]) -> None:
        """Recompute ``z``'s address and below/above edges from earlier buffers.

        Uses only buffers placed before ``z`` (``pos[w] < pos[z]``); ``z`` stacks
        on top of those it overlaps, except for an in-place reuse. Reciprocal
        ``above_neighbors`` edges are kept consistent. Candidates are drawn from
        ``z``'s precomputed time-overlap set, so this costs O(degree) rather than
        scanning all buffers.
        """
        for b in self.below_neighbors[z]:
            self.above_neighbors[b].discard(z)

        pos_z = pos[z]
        cand = [w for w in self.overlaps[z] if pos[w] < pos_z]
        addr, partner = self._placement_decision(z, cand)
        self.addresses[z] = addr
        if partner is None:
            self.inplace_reuse.pop(z, None)
        else:
            self.inplace_reuse[z] = partner

        below = self._column_below(z, cand)
        if partner is not None:
            below.add(partner)
        self.below_neighbors[z] = below
        for b in below:
            self.above_neighbors[b].add(z)

    def _column_below(self, z: int, cand: list[int]) -> set[int]:
        """Direct below-neighbours of ``z`` among ``cand`` (already placed).

        At each tick of ``z``'s life, take the column with the greatest top at or
        below ``z``'s address and include every buffer sharing that column's
        address (in-place siblings); union over ticks. Mirrors the per-tick rule
        in :meth:`_build_neighbor_graph`, restricted to the candidate set.
        """
        addr = self.addresses[z]
        assert addr is not None
        below: set[int] = set()
        bz = self.buffers[z]
        for t in range(bz.start_time, bz.end_time + 1):
            nearest_top = -1
            nearest_addr = None
            for w in cand:
                bw = self.buffers[w]
                if not (bw.start_time <= t <= bw.end_time):
                    continue
                top_w = self._top(w)
                if top_w <= addr and top_w > nearest_top:
                    nearest_top = top_w
                    nearest_addr = self.addresses[w]
            if nearest_addr is None:
                continue
            for w in cand:
                bw = self.buffers[w]
                if (
                    bw.start_time <= t <= bw.end_time
                    and self.addresses[w] == nearest_addr
                ):
                    below.add(w)
        return below
