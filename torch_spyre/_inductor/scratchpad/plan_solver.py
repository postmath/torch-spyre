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
import bisect
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

    def overlaps_in_time(self, other: "LifetimeBoundBuffer") -> bool:
        """Returns true iff self and other overlap in time."""
        return self.start_time < other.end_time and other.start_time < self.end_time


def _coalesce_segments(
    starts: list[int], labels: list[Optional[int]]
) -> tuple[list[int], list[Optional[int]]]:
    """Merge adjacent segments carrying equal labels. ``starts`` has length
    ``len(labels) + 1``; segment ``i`` covers ``[starts[i], starts[i+1])``."""
    out_starts = [starts[0]]
    out_labels: list[Optional[int]] = []
    for i, label in enumerate(labels):
        if out_labels and out_labels[-1] == label:
            out_starts[-1] = starts[i + 1]  # extend the previous segment
        else:
            out_labels.append(label)
            out_starts.append(starts[i + 1])
    return out_starts, out_labels


class Profile:
    """A step function from a half-open span ``[span_start, span_end)`` to labels
    (each an ``Optional[int]``; ``None`` means "no neighbour here").

    Stored as parallel lists: ``starts`` of length ``n + 1`` and ``labels`` of
    length ``n``; segment ``i`` covers ``[starts[i], starts[i + 1])`` carrying
    ``labels[i]``, with ``starts[-1] == span_end``.

    Canonical form (every mutating operation restores it): ``starts`` strictly
    increasing, and no two adjacent segments carry equal labels.
    """

    __slots__ = ("starts", "labels")

    def __init__(self, starts: list[int], labels: list[Optional[int]]):
        self.starts = starts
        self.labels = labels

    @classmethod
    def uniform(cls, span_start: int, span_end: int, label: Optional[int]) -> "Profile":
        """A single-segment profile over ``[span_start, span_end)``."""
        assert span_start < span_end
        return cls([span_start, span_end], [label])

    @classmethod
    def from_segments(cls, starts: list[int], labels: list[Optional[int]]) -> "Profile":
        """Build a canonical profile from segments that tile the span (coalescing
        adjacent equal labels)."""
        assert len(starts) == len(labels) + 1 and len(labels) >= 1
        return cls(*_coalesce_segments(starts, labels))

    @property
    def span_start(self) -> int:
        return self.starts[0]

    @property
    def span_end(self) -> int:
        return self.starts[-1]

    def label_at(self, t: int) -> Optional[int]:
        """The label of the segment containing column ``t`` (``t`` in span)."""
        assert self.starts[0] <= t < self.starts[-1]
        return self.labels[bisect.bisect_right(self.starts, t) - 1]

    def segments(self, a: int, b: int) -> tuple[list[int], list[Optional[int]]]:
        """The segments clipped to ``[a, b)`` as fresh lists (no aliasing): the
        first segment's start is clamped to ``a`` and the last end to ``b``.
        An empty range yields ``([a], [])``."""
        assert self.starts[0] <= a <= b <= self.starts[-1]
        if a == b:
            return [a], []
        out_starts = [a]
        out_labels: list[Optional[int]] = []
        i = bisect.bisect_right(self.starts, a) - 1
        while self.starts[i] < b:
            out_labels.append(self.labels[i])
            out_starts.append(min(self.starts[i + 1], b))
            i += 1
        return out_starts, out_labels

    def splice(
        self, a: int, b: int, seg_starts: list[int], seg_labels: list[Optional[int]]
    ) -> None:
        """Replace the function on ``[a, b)`` with the given segments (which must
        exactly tile ``[a, b)``), coalescing at both seams. No-op if ``a == b``."""
        assert self.starts[0] <= a <= b <= self.starts[-1]
        if a == b:
            return
        assert seg_starts[0] == a and seg_starts[-1] == b
        left_s, left_l = self.segments(self.starts[0], a)
        right_s, right_l = self.segments(b, self.starts[-1])
        new_s = left_s[:-1] + list(seg_starts[:-1]) + right_s
        new_l = left_l + list(seg_labels) + right_l
        self.starts, self.labels = _coalesce_segments(new_s, new_l)

    def relabel(self, a: int, b: int, mapping: dict) -> None:
        """For every segment within ``[a, b)`` whose label is a key of
        ``mapping``, replace it with ``mapping[label]`` (splitting straddling
        segments at the boundaries); coalesce afterwards. No-op if ``a == b``."""
        if a == b:
            return
        seg_s, seg_l = self.segments(a, b)
        new_l = [mapping[label] if label in mapping else label for label in seg_l]
        self.splice(a, b, seg_s, new_l)

    def label_set(self) -> set:
        """The set of labels appearing anywhere in the profile.

        (Named ``label_set`` rather than ``labels`` because ``labels`` is the
        segment-label list attribute.)"""
        return set(self.labels)

    def validate(self) -> None:
        """Raise ``AssertionError`` if the canonical-form invariants are broken."""
        assert len(self.starts) == len(self.labels) + 1, "length mismatch"
        assert len(self.labels) >= 1, "profile must have at least one segment"
        for i in range(len(self.starts) - 1):
            assert self.starts[i] < self.starts[i + 1], "starts not strictly increasing"
        for i in range(len(self.labels) - 1):
            assert self.labels[i] != self.labels[i + 1], "adjacent labels equal"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Profile)
            and self.starts == other.starts
            and self.labels == other.labels
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        segs = ", ".join(
            f"[{self.starts[i]},{self.starts[i + 1]})={self.labels[i]}"
            for i in range(len(self.labels))
        )
        return f"Profile({segs})"


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
    def __init__(self, size: int, alignment: int = 128):
        super().__init__(size, alignment)
        self.usage: list[LifetimeBoundBuffer] = []

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


class PermutationBasedLayoutSolverBase(ABC):
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
    address (``P.end_time == C.start_time + 1``).

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
        self.addresses: list[int] = [0] * n

        # Sum of buf.size over all fully-allocated buffers (address + size <=
        # capacity). Maintained incrementally; exposed via quality(). Also, the
        # count of these buffers, exposed via count_allocated().
        self.total_allocated_size: int = 0
        self.total_allocated_count: int = 0

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
            The change in :meth:`quality` caused by the swap (new minus old).
        """
        pass

    # --- shared helpers -----------------------------------------------------

    def rotate(self, i: int, j: int) -> int:
        """Modify the permutation by taking ``self.permutation[i]`` out of the permutation and
        reinserting it at position ``j``. Returns the change in :meth:`quality` caused by the
        rotation (new minus old)."""
        # A product of swaps, even over the full distance, beats a permutation-edit + _build():
        # most of the swaps are O(1) no-ops, so the chain is far cheaper than an O(n^2) rebuild in
        # the realistic (sparse-overlap) regime. (A rebuild only wins for dense overlap, where it
        # is a symptom of swap propagation degenerating -- a thing to fix, not to route around. See
        # benchmarks/copy_vs_swap_results.md.)
        delta = 0
        if i < j:
            for k in range(i, j):
                delta += self.swap(k)
        elif j < i:
            for k in range(i - 1, j - 1, -1):
                delta += self.swap(k)
        return delta

    def _align_up(self, addr: int) -> int:
        """Round ``addr`` up to the next multiple of ``self.alignment``."""
        return math.ceil(addr / self.alignment) * self.alignment

    def _top(self, idx: int) -> int:
        """Return ``address + size`` for a placed buffer (its exclusive top)."""
        addr = self.addresses[idx]
        return addr + self.buffers[idx].size

    def _is_fully_allocated(self, idx: int) -> bool:
        """True if buffer ``idx`` has an address and fits below ``capacity``."""
        addr = self.addresses[idx]
        return addr + self.buffers[idx].size <= self.capacity

    def _overlaps(self, i: int, j: int) -> bool:
        """True if buffers ``i`` and ``j`` are alive at a common tick.

        Lifetimes are half-open intervals ``[start_time, end_time)``, so an
        in-place parent and child (``parent.end_time == child.start_time + 1``)
        overlap at exactly that boundary tick (``child.start_time``).
        """
        return self.buffers[i].overlaps_in_time(self.buffers[j])

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
        an in-place partner ``P`` (``P.end_time == idx.start_time + 1`` or vice
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
            others_top = max(
                (self._top(q) for q in candidates if q != partner), default=0
            )
            if others_top <= partner_addr:
                return partner_addr, partner
        return self._align_up(max_top), None

    def _address_from_candidates(self, idx: int, candidates: list[int]) -> int:
        """Return only the address from :meth:`_placement_decision`."""
        return self._placement_decision(idx, candidates)[0]

    def quality(self) -> int:
        """Total size of all buffers fully allocated below capacity (O(1))."""
        return self.total_allocated_size

    def count_allocated(self) -> int:
        """Count of all buffers fully allocated below capacity (O(1))."""
        return self.total_allocated_count

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


class ReferencePermutationBasedLayoutSolver(PermutationBasedLayoutSolverBase):
    """Simple, obviously-correct O(n^2) reference plan.

    Placement scans all previously-placed, time-overlapping buffers for each
    buffer; ``swap`` mutates the permutation and rebuilds from scratch. Kept as
    a permanent oracle for differential testing against the incremental
    :class:`PermutationBasedLayoutSolver`.
    """

    def _build(self) -> None:
        n = len(self.buffers)
        self.addresses = [0] * n
        self.total_allocated_size = 0
        self.total_allocated_count = 0
        for pos in range(n):
            idx = self.permutation[pos]
            prior = self.permutation[:pos]
            candidates = [p for p in prior if self._overlaps(idx, p)]
            self.addresses[idx] = self._address_from_candidates(idx, candidates)
            if self._is_fully_allocated(idx):
                self.total_allocated_size += self.buffers[idx].size
                self.total_allocated_count += 1

    def swap(self, i: int) -> int:
        """Swap permutation entries ``i``/``i+1`` and rebuild from scratch."""
        old_total = self.total_allocated_size
        perm = self.permutation
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        self._build()
        return self.total_allocated_size - old_total


class PermutationBasedLayoutSolver(PermutationBasedLayoutSolverBase):
    """Incremental capacity-bounded allocation plan.

    Maintains, for each buffer, a *contact profile* -- a step function over its
    lifetime giving the buffer directly below / above it in the per-column
    stacking order (or None at the ends). Swapping two adjacent permutation
    entries transposes them only over their shared column range, so the profiles
    are updated by O(segments) splices rather than rebuilt; addresses are then
    re-placed for the buffers the change actually reaches, propagated along the
    time-overlap dependency graph.

    The contact relation is purely order-based (a function of the permutation
    and lifetimes): at a column the alive buffers are ordered by permutation
    position, and ``below_profile[c]`` at that column is ``c``'s immediate
    predecessor in that order. In-place placement is ignored by the relation
    (parent-before-child means parent-below-child); it still affects addresses,
    which are computed separately.

    Attributes:
        below_profile: ``below_profile[c]`` maps each column of ``c``'s lifetime
            to the buffer directly below ``c`` there, or None.
        above_profile: the inverse relation; used to find which buffers may need
            re-placing when ``c``'s top moves.
        inplace_reuse: ``inplace_reuse[x] = y`` when buffer ``x`` reused
            partner ``y``'s address in-place (``x`` was placed at ``y``'s
            address).
    """

    def _build(self) -> None:
        n = len(self.buffers)
        self.addresses = [0] * n
        self.total_allocated_size = 0
        self.total_allocated_count = 0
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
                self.total_allocated_count += 1
        # Persistent position index, maintained in O(1) by swap().
        self.position: list[int] = [0] * n
        for p, idx in enumerate(self.permutation):
            self.position[idx] = p
        # Time-overlap sets. Lifetimes never change, so this is computed once
        # and lets the address recompute find a buffer's candidates in O(degree)
        # instead of scanning all n buffers.
        self.overlaps: dict[int, set[int]] = {i: set() for i in range(n)}
        for a in range(n):
            for b in range(a + 1, n):
                if self._overlaps(a, b):
                    self.overlaps[a].add(b)
                    self.overlaps[b].add(a)
        self._build_profiles()

    def _build_profiles(self) -> None:
        """Build the below/above contact profiles from ground truth.

        At each column the buffers alive there are totally ordered by
        permutation position (the bottom-to-top stacking order); a buffer's
        below/above neighbour is its immediate predecessor / successor in that
        per-column order, or None at the ends. Sweeping the breakpoint intervals
        and reading adjacent pairs gives each buffer's contact step function over
        its lifetime. In-place placement is ignored -- the relation is purely a
        function of the permutation and lifetimes.
        """
        n = len(self.buffers)
        self.below_profile: dict[int, Profile] = {}
        self.above_profile: dict[int, Profile] = {}
        if n == 0:
            return
        bufs = self.buffers
        below_segs: dict[int, tuple[list[int], list[Optional[int]]]] = {
            i: ([], []) for i in range(n)
        }
        above_segs: dict[int, tuple[list[int], list[Optional[int]]]] = {
            i: ([], []) for i in range(n)
        }
        breakpoints = sorted({b.start_time for b in bufs} | {b.end_time for b in bufs})
        for t0 in breakpoints[:-1]:
            alive = sorted(
                (i for i in range(n) if bufs[i].start_time <= t0 < bufs[i].end_time),
                key=lambda i: self.position[i],
            )
            for idx, c in enumerate(alive):
                below = alive[idx - 1] if idx > 0 else None
                above = alive[idx + 1] if idx + 1 < len(alive) else None
                below_segs[c][0].append(t0)
                below_segs[c][1].append(below)
                above_segs[c][0].append(t0)
                above_segs[c][1].append(above)
        for i in range(n):
            bs, bl = below_segs[i]
            bs.append(bufs[i].end_time)
            self.below_profile[i] = Profile.from_segments(bs, bl)
            as_, al = above_segs[i]
            as_.append(bufs[i].end_time)
            self.above_profile[i] = Profile.from_segments(as_, al)

    def swap(self, i: int) -> int:
        """Swap permutation entries ``i`` and ``i+1`` and re-place incrementally.

        A no-op when the swapped buffers do not overlap in time. Otherwise:

        1. Over their shared column range the two buffers' per-column order
           transposes and nothing else changes, so the contact profiles are
           updated by a handful of splices (:meth:`_update_profiles_for_swap`).
        2. Addresses are then re-placed for the buffers the change reaches,
           processed in a min-heap by position (dependencies always point to
           earlier positions, so a buffer is settled before anything resting on
           it; ``position`` is maintained in O(1)). Two kinds of edge feed the
           dirty set:

           - *Order-above.* When ``z``'s address changes, the buffers directly
             above it -- ``above_profile[z]`` -- are dirtied. This is the cheap
             contact-profile frontier and it is exactly right whenever the
             buffer a dependent rests on is also its order-below neighbour.

           - *In-place transition.* In-placement makes the contact order and the
             rest-on order diverge: a transparent in-place child sits low while
             its taller parent pokes through and binds the buffer above the
             child. While that in-placement is stable the order-above frontier
             still suffices (the child's address tracks the parent it reuses, so
             a change in the parent reaches the buffer above the child through
             the child). The gap is at the *transition*: when a buffer ``z``'s
             in-place status flips (activates or deactivates), the poke-through
             appears or vanishes, so the buffer resting on it must be revisited
             even though nothing it can see changed value. So on a status change
             we dirty the order-above neighbour of *both* members of the pair at
             their shared (overlap) tick -- the parent's above-neighbour is the
             child, and the child's above-neighbour is the buffer that gains or
             loses the poke-through.

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

        # 1. Transpose the contact profiles over the shared column range.
        a = max(self.buffers[x].start_time, self.buffers[y].start_time)
        b = min(self.buffers[x].end_time, self.buffers[y].end_time)
        self._update_profiles_for_swap(x, y, a, b)

        # 2. Re-place affected addresses, propagating along order-above edges and
        # in-place transitions (see the method docstring). Seed with the swapped
        # pair and whatever rested on them before the swap.
        old_total = self.total_allocated_size
        seed: set[int] = {x, y}
        for lbl in (
            self.above_profile[x].label_set() | self.above_profile[y].label_set()
        ):
            if lbl is not None:
                seed.add(lbl)
        heap = [(self.position[idx], idx) for idx in seed]
        heapq.heapify(heap)
        queued = set(seed)

        def _dirty(w: Optional[int], pos_z: int) -> None:
            if w is not None and w not in queued and self.position[w] > pos_z:
                queued.add(w)
                heapq.heappush(heap, (self.position[w], w))

        while heap:
            _, z = heapq.heappop(heap)
            queued.discard(z)
            pos_z = self.position[z]
            old_addr = self.addresses[z]
            old_partner = self.inplace_reuse.get(z)
            if self._is_fully_allocated(z):
                self.total_allocated_size -= self.buffers[z].size
                self.total_allocated_count -= 1
            self._recompute_address(z)
            if self._is_fully_allocated(z):
                self.total_allocated_size += self.buffers[z].size
                self.total_allocated_count += 1
            new_partner = self.inplace_reuse.get(z)
            if self.addresses[z] != old_addr:
                for w in self.above_profile[z].label_set():
                    _dirty(w, pos_z)
            if new_partner != old_partner:
                # In-place status changed: revisit the buffers resting on the
                # pair at the tick where parent and child overlap.
                for partner in (old_partner, new_partner):
                    if partner is None:
                        continue
                    pair = self._in_place_pair(z, partner)
                    assert pair is not None  # partner is a recorded in-place reuse
                    parent, child = pair
                    t = self.buffers[child].start_time
                    _dirty(self.above_profile[child].label_at(t), pos_z)
                    _dirty(self.above_profile[parent].label_at(t), pos_z)
        return self.total_allocated_size - old_total

    def _update_profiles_for_swap(self, x: int, y: int, a: int, b: int) -> None:
        """Transpose ``x`` (was lower) and ``y`` (was upper) in the contact
        profiles over the shared column range ``[a, b)``.

        Captures both views before mutating, then runs the same splice logic
        once per side (downward and upward are exact mirrors).
        """
        old_x_below = self.below_profile[x].segments(a, b)
        old_y_above = self.above_profile[y].segments(a, b)
        self._splice_half(
            self.below_profile, self.above_profile, x, y, a, b, old_x_below
        )
        self._splice_half(
            self.above_profile, self.below_profile, y, x, a, b, old_y_above
        )

    @staticmethod
    def _splice_half(
        primary: dict[int, Profile],
        reverse: dict[int, Profile],
        lo: int,
        hi: int,
        a: int,
        b: int,
        old_lo: tuple[list[int], list[Optional[int]]],
    ) -> None:
        """One side of the transposition. ``lo`` was directly below ``hi`` (in
        the ``primary`` direction) over ``[a, b)``; after the swap ``hi`` is.

        - ``primary[lo]`` over ``[a, b)`` becomes ``hi``.
        - ``primary[hi]`` over ``[a, b)`` inherits ``lo``'s old ``primary`` view.
        - Each buffer ``lo`` pointed at keeps the relationship but now via
          ``hi``, so its ``reverse`` profile relabels ``lo -> hi`` over that
          segment.
        """
        primary[lo].splice(a, b, [a, b], [hi])
        seg_starts, seg_labels = old_lo
        primary[hi].splice(a, b, list(seg_starts), list(seg_labels))
        for k, label in enumerate(seg_labels):
            if label is not None:
                reverse[label].relabel(seg_starts[k], seg_starts[k + 1], {lo: hi})

    def _recompute_address(self, z: int) -> None:
        """Re-place ``z``'s address from its earlier-positioned overlapping
        candidates (the contact profiles are not consulted; they were already
        updated by the splice)."""
        pos_z = self.position[z]
        cand = [w for w in self.overlaps[z] if self.position[w] < pos_z]
        addr, partner = self._placement_decision(z, cand)
        self.addresses[z] = addr
        if partner is None:
            self.inplace_reuse.pop(z, None)
        else:
            self.inplace_reuse[z] = partner

    def contact_at(self, c: int, t: int) -> Optional[int] | tuple[int, int]:
        """What buffer ``c`` rests on at column ``t``, surfacing in-place reuse.

        The stored ``below_profile`` records only the *order*-below neighbour,
        which is not always what ``c`` physically rests on: across an active
        in-placement the order-below buffer is a transparent child reusing its
        parent's address, and the taller parent pokes through to actually carry
        ``c``. This is the faithful view, derived on demand from the order
        profile and :attr:`inplace_reuse` (nothing extra is stored):

        - ``None`` -- ``c`` is on the floor at ``t`` (no buffer below).
        - ``int m`` -- ``c`` rests on buffer ``m``.
        - ``(parent, child)`` -- the slot directly below ``c`` is held by an
          active in-place pair: ``c`` rests on ``parent`` while ``child`` (which
          reuses ``parent``'s address and is ``c``'s order-below neighbour) sits
          transparently inside it. Reported only at the single tick where parent
          and child overlap; elsewhere the parent is gone and ``c`` rests on the
          (former) child as a plain ``int``.
        """
        m = self.below_profile[c].label_at(t)
        if m is None:
            return None
        partner = self.inplace_reuse.get(m)
        if partner is not None:
            pair = self._in_place_pair(m, partner)
            assert pair is not None  # m reuses partner: they form a pair
            parent, child = pair
            pbuf = self.buffers[parent]
            if m == child and pbuf.start_time <= t < pbuf.end_time:
                return (parent, child)
        return m

    def copy(self) -> "PermutationBasedLayoutSolver":
        """Return an independent layout snapshot that can be mutated (via
        :meth:`swap` / :meth:`rotate`) without affecting this one.

        Structures fixed for the lifetime of the plan -- ``buffers``,
        ``_name_to_idx``, ``overlaps`` -- are shared by reference; only the
        dynamic layout state (permutation, addresses, positions, contact
        profiles and running totals) is deep-copied. So this costs O(n + profile
        size), not a rebuild. The result is always a plain
        :class:`PermutationBasedLayoutSolver`, regardless of subclass.
        """
        clone = PermutationBasedLayoutSolver.__new__(PermutationBasedLayoutSolver)
        # Shared, immutable-during-planning structures.
        clone.buffers = self.buffers
        clone._name_to_idx = self._name_to_idx
        clone.capacity = self.capacity
        clone.alignment = self.alignment
        clone.overlaps = self.overlaps
        # Deep-copied dynamic state.
        clone.permutation = list(self.permutation)
        clone.addresses = list(self.addresses)
        clone.position = list(self.position)
        clone.total_allocated_size = self.total_allocated_size
        clone.total_allocated_count = self.total_allocated_count
        clone.inplace_reuse = dict(self.inplace_reuse)
        clone.below_profile = {
            k: Profile(list(p.starts), list(p.labels))
            for k, p in self.below_profile.items()
        }
        clone.above_profile = {
            k: Profile(list(p.starts), list(p.labels))
            for k, p in self.above_profile.items()
        }
        return clone
