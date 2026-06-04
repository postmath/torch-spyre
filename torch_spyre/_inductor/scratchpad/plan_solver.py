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

    def _inplace_is_safe(
        self, base: int, size: int, candidates: list[int], exclude: int
    ) -> bool:
        """True if placing a buffer at ``[base, base + size)`` collides with no
        candidate other than ``exclude``.

        Used to validate in-place reuse: an in-place partner only shields the
        child from buffers that also coexisted with the parent. Any other
        candidate alive during the child's lifetime that intrudes on the reused
        range forbids the reuse (we never place a child partway into occupied
        space).
        """
        hi = base + size
        for p in candidates:
            if p == exclude:
                continue
            assert (addr := self.addresses[p]) is not None
            if addr < hi and base < self._top(p):
                return False
        return True

    def _address_from_candidates(self, idx: int, candidates: list[int]) -> int:
        """Compute ``idx``'s address given the buffers it must sit on top of.

        ``candidates`` are already-placed buffer indices that overlap ``idx`` in
        time. For the reference plan these are *all* time-overlapping buffers;
        for the incremental plan they are ``idx``'s direct below-neighbours --
        both yield the same address because shielded buffers neither raise the
        high-water mark nor affect the safety check.

        ``idx`` stacks immediately above the highest candidate, unless that
        highest candidate is an in-place partner whose address can be safely
        reused (see :meth:`_inplace_is_safe`).
        """
        if not candidates:
            return 0
        # Topmost candidate: highest exclusive top; ties prefer an in-place
        # partner (so reuse gets a chance), then lowest index for determinism.
        top_buf = max(
            candidates,
            key=lambda p: (self._top(p), self._in_place_pair(idx, p) is not None, -p),
        )
        max_top = self._top(top_buf)
        pair = self._in_place_pair(idx, top_buf)
        if pair is not None and self._can_inplace(*pair):
            base = self.addresses[top_buf]
            assert base is not None
            if self._inplace_is_safe(base, self.buffers[idx].size, candidates, top_buf):
                return base
        return self._align_up(max_top)

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
        raise NotImplementedError("Step 4")


class CappedAllocatorPlan(CappedAllocatorPlanBase):
    """Incremental capacity-bounded allocation plan.

    Maintains a neighbor graph (buffers directly below/above each buffer in
    address space, including air-gap dependencies) so that swapping two adjacent
    permutation entries only re-places the affected buffers via change-driven
    propagation, rather than rebuilding the whole layout.
    """

    def _build(self) -> None:
        # Step 3: neighbor-graph construction + placement.
        pass

    def swap(self, i: int) -> int:
        raise NotImplementedError("Step 4")
