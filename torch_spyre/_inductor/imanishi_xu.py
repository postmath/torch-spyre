# Implement the algorithm from this paper:
#
# Imanishi, Akifumi, and Zijian Xu. "A heuristic for periodic memory allocation with little
# fragmentation to train neural networks." In Proceedings of the 2024 ACM SIGPLAN International
# Symposium on Memory Management, pp. 82-94. 2024.
#
# The paper describes a few algorithms that work together to come up with a good allocation scheme.
# The problem setting differs slightly from ours in that they have a fixed set of buffers that are
# all to be allocated, and they want to do it in as little space as possible. By contrast, in our
# case, we have a fixed amount of space and we want to allocate those buffers that will give the
# best performance -- which we can probably approximate by saying, we want to minimize the volume of
# HBM transfers.
#
# The first algorithm we implement is called Algorithm 2. It allocates addresses for buffers *given*
# a permutation (ordering) of the buffers. This is a fairly simple algorithm. The smarts are in
# coming up with the right permutation.
#
# Algorithm 4 is the simulated annealing algorithm that comes up with the permutation. It takes as
# inputs an annealing schedule, a list of buffers, and an initial permutation.

from dataclasses import dataclass
from heapq import heappush, heappop
import math
from typing import Iterable, Iterator, Optional, override, Callable
from abc import ABC, abstractmethod
import random as rnd
import numpy as np
import copy


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


@dataclass
class Buffer:
    name: str
    size: int
    # The buffer is allocated from tick first_use up to, but not including, last_use.
    first_use: int
    last_use: int

    def __post_init__(self):
        # The original paper doesn't require this, so that it can support cyclic/periodic
        # allocations. We could relatively easily extend the code to allow for this.
        assert self.last_use >= self.first_use

    @classmethod
    def random(
        cls, name, size_range: int, time_range: int, random: Optional[rnd.Random] = None
    ) -> "Buffer":
        if random is None:
            random = rnd.Random()

        duration = random.randrange((time_range - 1) // 2)
        # Bias towards smaller time ranges:
        duration = duration * duration // (time_range - 1)
        t_start = random.randrange(time_range - duration)
        t_end = t_start + duration + 1

        size = random.randrange(size_range)
        # Bias towards larger sizes:
        size = math.isqrt(size * size_range)

        return cls(
            name=name,
            size=size,
            first_use=t_start,
            last_use=t_end,
        )

    def overlaps_in_time(self, other: "Buffer") -> bool:
        return overlaps(
            (self.first_use, self.last_use + 1), (other.first_use, other.last_use + 1)
        )


@dataclass
class BufferList:
    _list: list[Buffer]
    max_time: int

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, i: int) -> Buffer:
        return self._list[i]

    def __iter__(self) -> Iterator[Buffer]:
        return iter(self._list)

    @classmethod
    def from_buffers(cls, buffers: list[Buffer]) -> "BufferList":
        max_time = max(b.last_use for b in buffers) if buffers else 0
        return cls(buffers, max_time)


class MaxRangeTree(ABC):
    @abstractmethod
    def __init__(self, n: int):
        """Create an instance representing an array, say A, of length n."""
        ...

    @abstractmethod
    def increase_values(
        self,
        left: int,
        right: int,
        val: int,
    ):
        """Increase A[left:right] to val. Precondition: max(A[left:right]) <= val."""
        ...

    @abstractmethod
    def max(
        self,
        left: int,
        right: int,
    ) -> int:
        """Return the maximum of A[left:right]."""
        ...


class MaxRangeTree_Array(MaxRangeTree):
    @override
    def __init__(self, n: int):
        self.array = np.zeros(n, dtype="int64")

    @override
    def increase_values(
        self,
        left: int,
        right: int,
        val: int,
    ):
        self.array[left:right] = val

    @override
    def max(
        self,
        left: int,
        right: int,
    ) -> int:
        return self.array[left:right].max()


class MaxRangeTree_List(MaxRangeTree):
    @override
    def __init__(self, n: int):
        self.list = [0] * n

    @override
    def increase_values(
        self,
        left: int,
        right: int,
        val: int,
    ):
        self.list[left:right] = [val] * (right - left)

    @override
    def max(
        self,
        left: int,
        right: int,
    ) -> int:
        return max(self.list[left:right])


class MaxRangeTree_Tree(MaxRangeTree):
    """Data structure representing an array A of integers that allows one to quickly compute the
    maximum of a subarray, and to set a subarray to a constant value *higher than its current
    value*. In the public interface, indexing works as you'd expect in python:

    ```
    t = MaxRangeTree(n)
    # (...)
    t.max(l, r) # returns the max of A[l:r] for 0 <= l < r <= n
    t.increase_values(l, r, val) # corresponds to setting A[l:r] = val, assuming t.max(l, r) <= val
    ```

    Internally, self.tree[1] is the root of the tree; self.tree[2*i] and self.tree[2*i+1] are the
    children of node i. self.tree[0] is not used. This means that node i corresponds to  If
    self.lazy[i] is not None, then the whole segment under node i is set to self.lazy[i]. If *all of
    self.lazy[i]'s parents* (but not necessarily self.lazy[i] itself) are None, then self.tree[i] is
    the max of the corresponding segment.

    This data structure is actually slower than the other two, but if we need something like this,
    we should implement it in C++. (It is actually a significant use of time, even with the other
    two versions.)
    """

    @override
    def __init__(self, n: int):
        """Initialize the data structure representing an array of n zeroes. We need n > 0."""
        assert n > 0
        self.n = n
        self.log_n = (n - 1).bit_length()
        # Round n up to a power of two, then double it, then add 1. This ensures that a full level
        # of the tree is filled, and that every node's children are allocated. This simplifies the
        # code below.
        list_length = 1 + 1 << (1 + (n - 1).bit_length())
        self.tree: list[int] = [0] * list_length
        self.lazy: list[Optional[int]] = [None] * list_length

    def _push_down(self, node: int):
        """If the given node is lazy, push its laziness down to its children."""
        if self.lazy[node] is not None:
            for child in (2 * node, 2 * node + 1):
                self.tree[child] = self.lazy[node]  # type: ignore[assignment]
                self.lazy[child] = self.lazy[node]
            self.lazy[node] = None

    @override
    def max(
        self,
        left: int,
        right: int,
    ) -> int:
        result = -1
        stack: list[tuple[int, int, int, int]] = []
        node, lo, hi, log_k = 1, 0, self.n, self.log_n

        while True:
            # Base case 1: A[lo:hi] does not overlap with A[left:right].
            if right <= lo or hi <= left:
                if not stack:
                    return result
                node, lo, hi, log_k = stack.pop()
                continue

            # Base case 2: A[left:right] is contained in A[lo:hi].
            if left <= lo and hi <= right:
                if result < self.tree[node]:
                    result = self.tree[node]

                if not stack:
                    return result
                node, lo, hi, log_k = stack.pop()
                continue

            # "Recursive" case: partial overlap with the current node.
            self._push_down(node)
            log_k -= 1
            mid = lo + (1 << log_k)
            if mid <= hi:
                # Two branches to explore. Push the left child to the stack to visit later.
                stack.append((2 * node, lo, mid, log_k))

                # Immediately traverse down the left child.
                node, lo = 2 * node + 1, mid

            else:
                # Only one branch to explore -- the left one.
                node, hi = 2 * node, mid

    @override
    def increase_values(
        self,
        left: int,
        right: int,
        val: int,
    ):
        """If only left, right, and val are passed, set A[left:right] = val. This *requires* that
        max(A[left:right]) <= val. Otherwise, incorrect results may be obtained.

        Otherwise, all arguments should be passed. Then, this method sets A on the intersection
        between left:right and lo:hi. `node` is the node representing A[lo:hi], and no parent of
        node is lazy. There is m such that lo == m * 2**log_k and hi <= (m+1) * 2**log_k."""
        stack: list[tuple[int, int, int, int]] = []
        update_stack = []
        node, lo, hi, log_k = 1, 0, self.n, self.log_n
        while True:
            # Base case 1: A[lo:hi] does not overlap with A[left:right].
            if right <= lo or hi <= left:
                if not stack:
                    break
                node, lo, hi, log_k = stack.pop()
                continue

            # Base case 2: A[left:right] is contained in A[lo:hi].
            if left <= lo and hi <= right:
                self.tree[node] = val
                self.lazy[node] = val

                if not stack:
                    break
                node, lo, hi, log_k = stack.pop()
                continue

            # "Recursive" case: partial overlap with the current node.
            self._push_down(node)

            # Record this node for updating on the way up.
            update_stack.append(node)

            log_k -= 1
            mid = lo + (1 << log_k)
            if mid <= hi:
                # Two branches to explore. Push the left child to the stack to visit later.
                stack.append((2 * node, lo, mid, log_k))

                # Immediately traverse down the left child.
                node, lo = 2 * node + 1, mid

            else:
                # Only one branch to explore -- the left one.
                node, hi = 2 * node, mid

        for node in reversed(update_stack):
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])


@dataclass
class Allocations:
    buffers: BufferList
    addresses: list[int]

    def __post_init__(self):
        assert len(self.buffers) == len(self.addresses)

    @classmethod
    def from_order(
        cls, buffers: BufferList, order: list[int]
    ) -> tuple[int, "Allocations"]:
        """Algorithm 2 from the paper."""
        n = len(buffers)
        if n < len(order):
            # NOTE: we sometimes call from_order to allocate only *some* of the buffers, in which
            # case we have n > len(order). That is okay.
            raise ValueError(
                f"Expected len(buffers) >= len(order), but got {n} < {len(order)}"
            )
        max_time = buffers.max_time

        # height[i] is the max height of all currently allocated blocks at time i.
        height = MaxRangeTree_List(max_time + 1)
        addresses = [0] * n
        for j in order:
            buffer = buffers[j]
            # Allocate buffer on top of currently allocated blocks.
            addresses[j] = height.max(buffer.first_use, buffer.last_use + 1)
            height.increase_values(
                buffer.first_use, buffer.last_use + 1, addresses[j] + buffer.size
            )

        return height.max(0, max_time + 1), cls(buffers, list(addresses))

    def to_order(self) -> list[int]:
        return sorted(range(len(self.buffers)), key=lambda i: self.addresses[i])

    def plot(self, max_height=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots()

        for i, buffer in enumerate(self.buffers):
            rect = patches.Rectangle(
                xy=(buffer.first_use, self.addresses[i]),
                width=buffer.last_use - buffer.first_use + 1,
                height=buffer.size,
                linewidth=0.3,
                edgecolor="r",
                facecolor="b",
                fill=True,
            )

            ax.add_patch(rect)

        ax.set_xlim(0, self.buffers.max_time + 1)
        if max_height is None:
            max_height = max(
                [self.addresses[i] + b.size for i, b in enumerate(self.buffers)]
            )
        ax.set_ylim(0, max_height)

        return fig


class CoolingSchedule(ABC):
    def __iter__(self) -> "CoolingSchedule":
        """We need iter() to return a *new* copy of the schedule."""
        return copy.deepcopy(self)

    @abstractmethod
    def __next__(self) -> float: ...


class ExponentialCoolingSchedule(CoolingSchedule):
    def __init__(self, *, t0: float, t_end: float, steps_per_epoch: int, epochs: int):
        self.t = t0
        self.alpha = (t_end / t0) ** (1 / epochs)
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.i = 0

    @override
    def __next__(self) -> float:
        self.i += 1
        if self.i % self.steps_per_epoch == 0:
            if self.i == self.steps_per_epoch * self.epochs:
                raise StopIteration
            self.t *= self.alpha
        return self.t


class CoolingScheduleFromPaper(CoolingSchedule):
    def __init__(self, *, buffers: BufferList, n: int = 1000000):
        buffers_sorted = sorted(buffers._list, key=lambda b: b.first_use)
        current_load = 0
        peak_load = 0
        # When we encounter a buffer, we include (last_use, size) in end_points, which is a min-heap
        # ordered by last_use.
        end_points: list[tuple[int, int]] = []
        for buffer in buffers_sorted:
            while end_points and end_points[0][0] < buffer.first_use:
                current_load -= heappop(end_points)[1]

            current_load += buffer.size
            peak_load = max(peak_load, current_load)

            heappush(end_points, (buffer.last_use, buffer.size))

        tau_s = peak_load / 300.0
        tau_e = min(100.0, tau_s / 1000.0)
        self.log_tau_s = math.log(tau_s)
        self.log_tau_e = math.log(tau_e)
        self.i = 0
        self.n = n

    @override
    def __next__(self) -> float:
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return math.exp(
            (self.log_tau_e - self.log_tau_s) * self.i / self.n + self.log_tau_s
        )


class DeterministicHeuristic:
    def __init__(self, buffers: list[Buffer] | BufferList):
        self.buffers = (
            buffers
            if isinstance(buffers, BufferList)
            else BufferList.from_buffers(buffers)
        )
        self.addresses = self.compute()
        allocations = Allocations(self.buffers, self.addresses)
        self.order = allocations.to_order()

    def __call__(self) -> list[int]:
        return self.order

    def compute(self) -> list[int]:
        raise NotImplementedError("This is an abstract class")


class FirstFit(DeterministicHeuristic):
    @staticmethod
    def all_minus(
        intervals: list[tuple[int, int]],
        interval: tuple[int, int],
        minimum_size: int,
    ) -> list[tuple[int, int]]:
        result = []
        for a, b in intervals:
            if a < interval[0]:
                if b < interval[0]:
                    if b - a >= minimum_size:
                        result.append((a, b))
                else:
                    if interval[0] - a >= minimum_size:
                        result.append((a, interval[0]))

            if b > interval[1]:
                if a > interval[1]:
                    if b - a >= minimum_size:
                        result.append((a, b))
                else:
                    if b - interval[1] >= minimum_size:
                        result.append((interval[1], b))

        return result

    def allocate_using_large_gaps(
        self, f: Callable[[list[tuple[int, int]], int], int]
    ) -> list[int]:
        buffers_sorted = sorted(
            self.buffers._list, key=lambda b: (-b.size, b.first_use)
        )
        total_buffer_size = sum(b.size for b in buffers_sorted)
        # We should use an interval tree here to keep track of at what time each allocated buffer is
        # alive.
        allocations = [0] * len(buffers_sorted)

        for i, buffer in enumerate(buffers_sorted):
            # A list of pairs (a, b) such that b - a >= buffer.size and gaps[i][1] < gaps[i+1][0],
            # and we have not seen any buffers overlapping (a, b).
            large_gaps = [(0, total_buffer_size)]

            for j, other_buffer in enumerate(buffers_sorted[:i]):
                if not overlaps(
                    (buffer.first_use, buffer.last_use + 1),
                    (other_buffer.first_use, other_buffer.last_use + 1),
                ):
                    continue

                # The buffers overlap in time. We want to find all entries of large_gaps that
                # overlap with (allocations[j], allocations[j] + other_buffer.size). Find a, b such
                # that large_gaps[a:b] are the gaps overlapping with that interval.
                #
                # Invariants: large_gaps[:a] are all strictly less than allocations[j] and
                # large_gaps[a1:] are all *not* strictly less than allocations[j].
                a = 0
                a1 = len(large_gaps)
                while a < a1:
                    mid = (a + a1) // 2
                    if large_gaps[mid][1] <= allocations[j]:
                        a = mid + 1
                    else:
                        a1 = mid

                # Invariant: large_gaps[b:] are all greater than or equal to allocations[j] +
                # other_buffer.size and large_gaps[:b1] are all *not* greater than or equal to
                # allocations[j] + other_buffer.size.
                b = len(large_gaps)
                b1 = a
                while b1 < b:
                    mid = (b1 + b) // 2
                    if large_gaps[mid][0] >= allocations[j] + other_buffer.size:
                        b = mid
                    else:
                        b1 = mid + 1

                if a < b:
                    large_gaps = (
                        large_gaps[:a]
                        + FirstFit.all_minus(
                            large_gaps[a:b],
                            (allocations[j], allocations[j] + other_buffer.size),
                            buffer.size,
                        )
                        + large_gaps[b:]
                    )

            allocations[i] = f(large_gaps, buffer.size)
        return list(allocations)

    @override
    def compute(self) -> list[int]:
        def allocate_first_fit(large_gaps: list[tuple[int, int]], size: int) -> int:
            return large_gaps[0][0]

        return self.allocate_using_large_gaps(allocate_first_fit)


class BestFit(FirstFit):
    @override
    def compute(self) -> list[int]:
        def allocate_best_fit(large_gaps: list[tuple[int, int]], size: int) -> int:
            best_gap = large_gaps[0]
            for gap in large_gaps[1:]:
                if gap[1] - gap[0] < best_gap[1] - best_gap[0]:
                    best_gap = gap
            return best_gap[0]

        return self.allocate_using_large_gaps(allocate_best_fit)


class ImanishiXuAllocator:
    def __init__(
        self,
        buffers: list[Buffer] | BufferList,
        *,
        order: list[int] | str = "first_fit",
        schedule: Iterable[float] | str = "from_paper",
        iterations: int = 1000000,
        random: Optional[rnd.Random] = None,
        ordering_fuzz_factor: float = 1.0,
        starts: int = 1,
    ):
        self.buffers = (
            buffers
            if isinstance(buffers, BufferList)
            else BufferList.from_buffers(buffers)
        )

        if isinstance(order, str):
            if order == "first_fit":
                order = FirstFit(self.buffers).order
            elif order == "best_fit":
                order = BestFit(self.buffers).order
            else:
                raise ValueError(f"Unknown order: {order}")
        self.order = order

        if isinstance(schedule, str):
            if schedule == "from_paper":
                schedule = CoolingScheduleFromPaper(buffers=self.buffers, n=iterations)
            elif schedule == "exponential":
                root = int(math.sqrt(iterations))
                schedule = ExponentialCoolingSchedule(
                    t0=max(b.size for b in self.buffers._list) * 10,
                    t_end=min(b.size for b in self.buffers._list) / 10,
                    steps_per_epoch=root,
                    epochs=root,
                )
            else:
                raise ValueError(f"Unknown schedule: {schedule}")
        self.schedule = iter(schedule)

        self.best_order = order
        self.best_height = Allocations.from_order(self.buffers, order)[0]
        if random:
            self.random = random
        else:
            self.random = rnd.Random()

        self.ordering_fuzz_factor = ordering_fuzz_factor
        self.height_logs: list[list[int]] = []
        self.starts = starts

    def solve(self):
        saved_order = copy.copy(self.order)
        for _ in range(self.starts):
            self.order = copy.copy(saved_order)
            self.anneal()

    def anneal(self):
        height_log = []
        for temperature in iter(self.schedule):
            match self.annealing_step_rotate(temperature):
                case (i, j):
                    self.annealing_step_swap(i, j)

            height = Allocations.from_order(self.buffers, self.order)[0]
            height_log.append(height)
            if height < self.best_height:
                self.best_height = height
                self.best_order = self.order

        self.height_logs.append(height_log)

    def annealing_step_swap(self, i: int, j: int):
        """This is the loop mentioned as Algorithms 5 and 6 in the paper."""
        _, allocations = Allocations.from_order(self.buffers, self.order)

        assert i != j, (
            "for a rotation i -> i, we should return None from the rotation method"
        )
        assert 0 <= i < len(self.order)
        assert 0 <= j < len(self.order)

        if i > j:
            i, j = j, i
        # Now i < j, and self.order[:i] and self.order[j+1:] are "clean"; that is, there is no k
        # such that self.order[k] and self.order[k+1] are buffers that *do not overlap* in time, and
        # have self.order[k] have a higher end point in memory than self.order[k+1]. Because
        # self.order[i] up to and including self.order[j] changed, we need to examine i-1 <= k <= j
        # -- except if that would take us outside the bounds of self.order, of course.
        i -= 1

        # Ensure that both i and j+1 are valid indices.
        if i < 0:
            i = 0
        if j == len(self.order) - 1:
            j = len(self.order) - 2

        while i <= j:
            if (
                not self.buffers[self.order[i]].overlaps_in_time(
                    self.buffers[self.order[i + 1]]
                )
            ) and (
                allocations.addresses[self.order[i]] + self.buffers[self.order[i]].size
                > allocations.addresses[self.order[i + 1]]
                + self.buffers[self.order[i + 1]].size
            ):
                # Swap buffers i and i+1. This makes no difference for the quality of the result
                # *now*, but it makes it easier to rotate to an improved state.
                self.order[i], self.order[i + 1] = self.order[i + 1], self.order[i]

                # Adjust the bounds of what we need to examine.
                if i == j and j < len(self.order) - 2:
                    j += 1

                if i > 0:
                    i -= 1
                else:
                    i = 1
            else:
                i += 1

    def annealing_step_rotate(self, temperature: float) -> Optional[tuple[int, int]]:
        """This is the inner loop of Algorithm 4 from the paper. The return value is (i, j) iff we
        accepted a rotation inserting entry i into position j."""
        n = len(self.buffers)
        i = self.random.randrange(n)
        buffer = self.buffers[self.order[i]]

        order_minus_i = self.order[:i] + self.order[i + 1 :]
        height_order_minus_i = Allocations.from_order(self.buffers, order_minus_i)[0]

        # "Up sweep".
        bottom_heights = [0] * (self.buffers.max_time + 1)
        bottom_addresses = [0] * n
        for j, other_i in enumerate(order_minus_i):
            other_buffer = self.buffers[other_i]
            # Consider the order that would be given by order_minus_i.insert(j, order[i]). Compute
            # the address where we would allocate buffer i given the order:
            #   order_minus_i[:j] + [order[i]].
            bottom_addresses[j] = max(
                bottom_heights[buffer.first_use : buffer.last_use + 1]
            )
            # For the next iteration, plan to allocate other_buffer next.
            other_allocation = max(
                bottom_heights[other_buffer.first_use : other_buffer.last_use + 1]
            )
            bottom_heights[other_buffer.first_use : other_buffer.last_use + 1] = [
                other_allocation + other_buffer.size
            ] * (other_buffer.last_use + 1 - other_buffer.first_use)
        bottom_addresses[n - 1] = max(
            bottom_heights[buffer.first_use : buffer.last_use + 1]
        )

        # "Down sweep".
        top_heights = [0] * (self.buffers.max_time + 1)
        top_addresses = [0] * n
        for j in range(n - 1, 0, -1):
            other_i = order_minus_i[j - 1]
            other_buffer = self.buffers[other_i]
            # Consider the order that would be given by order_minus_i.insert(j, order[i]). We now
            # compute the address where we would allocate buffer i given the order:
            #   reversed(order_minus_i[j:]) + [order[i]].
            # Imagine this allocation "upside down", hanging from the ceiling. We can obtain the
            # full height of the allocation order_minus_i.insert(j, order[i]) as
            #   max(height_order_minus_i, bottom_heights[j] + top_heights[j] + buffer.size).
            top_addresses[j] = max(top_heights[buffer.first_use : buffer.last_use + 1])

            # For the next iteration, plan to allocate other_buffer next.
            other_allocation = max(
                top_heights[other_buffer.first_use : other_buffer.last_use + 1]
            )
            top_heights[other_buffer.first_use : other_buffer.last_use + 1] = [
                other_allocation + other_buffer.size
            ] * (other_buffer.last_use + 1 - other_buffer.first_use)
        top_addresses[0] = np.max(top_heights[buffer.first_use : buffer.last_use + 1])

        # Actual total heights are the pointwise max of heights_through_buffer with
        # height_order_minus_i, but as the paper explains on page 89, it's better to work with just
        # heights_through_buffer.
        height_through_buffer = [
            ba + ta + buffer.size for ba, ta in zip(bottom_addresses, top_addresses)
        ]

        # The paper does not explain how they select a rotation if multiple rotations are accepted.
        # We implement this by sorting the insertion points descending by improvement plus a random
        # number scaled by the temperature, and then choosing the first accepted improvement. (An
        # alternative might be to choose an insertion point among the accepted ones by softmax of
        # the improvement.)
        insertion_points = sorted(
            range(n),
            key=lambda j: (
                -height_through_buffer[j]
                + self.random.random() * temperature * self.ordering_fuzz_factor
            ),
        )

        for j in insertion_points:
            if j == i:
                continue
            improvement = height_through_buffer[i] - height_through_buffer[j]
            if improvement > 0 or self.random.random() < math.exp(
                improvement / temperature
            ):
                self.order = order_minus_i[:j] + [self.order[i]] + order_minus_i[j:]

                new_height = max(height_through_buffer[j], height_order_minus_i)
                if new_height < self.best_height:
                    self.best_height = new_height
                    self.best_order = self.order

                return (i, j)

        # No rotation was accepted.
        return None

    def height_temperature_plot(self):
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        for log in self.height_logs:
            ax1.plot(log, lw=1, alpha=0.25)

        ax2 = ax1.twinx()
        ax2.set_yscale("log")
        ax2.plot([t for t in iter(self.schedule)])  # type: ignore[has-type]

        return fig

    def plot(self, max_height=None):
        _, allocation = Allocations.from_order(self.buffers, self.order)
        return allocation.plot(max_height=max_height)


if __name__ == "__main__":
    if False:
        buffers = [
            Buffer("B0", 8, 0, 1),
            Buffer("B1", 4, 1, 4),
            Buffer("B2", 2, 2, 5),
            Buffer("B3", 8, 3, 5),
        ]
        bl = BufferList.from_buffers(buffers)
        order = [0, 1, 2, 3]
        _, a1 = Allocations.from_order(bl, order)
        a1.plot(max_height=22).savefig("plot.png", dpi=300)

        # order = [0, 3, 1, 2]
        # _, a1 = Allocations.from_order(bl, order)
        # a1.plot(max_height=22).savefig("plot.png", dpi=300)

        # schedule = ExponentialCoolingSchedule(
        #     t0=10.0, t_end=1.0, steps_per_epoch=10, epochs=10
        # )
        # allocator = ImanishiXuAllocator(buffers=bl, order=order, schedule=schedule)
        # allocator.annealing_step_rotate(next(schedule))

    elif False:
        random = rnd.Random()
        random.seed(0)
        N = 100  # Number of buffers and also time range
        buffers = [Buffer.random(f"B_{i}", 1000000, N, random) for i in range(N)]
        random_height = Allocations.from_order(
            BufferList.from_buffers(buffers), list(range(N))
        )[0]
        print(f"Random arrangement: {random_height}")
        allocator = ImanishiXuAllocator(
            buffers=buffers,
            schedule=ExponentialCoolingSchedule(
                t0=1000000.0, t_end=1000.0, steps_per_epoch=10, epochs=1000
            ),
            random=random,
            order="first_fit",
            ordering_fuzz_factor=1000.0,
        )
        print(f"Initial arrangement: {allocator.best_height}")
        allocator.solve()
        print(f"Final arrangement: {allocator.best_height}")

        try:
            fig = allocator.plot()
            fig.savefig("plot.png", dpi=300)

        except ImportError:
            print("Not creating plot (matplotlib not installed)")

    else:
        random = rnd.Random()
        random.seed(0)

        buffers = [
            Buffer("A", 60, 0, 2),  # A: 0
            Buffer("B", 30, 1, 4),  # B: 1
            Buffer("C", 30, 2, 13),  # C: 2
            Buffer("D", 30, 3, 4),  # D: 3
            Buffer("E", 30, 4, 5),  # E: 4
            Buffer("F", 60, 5, 6),  # F: 5
            Buffer("G", 30, 6, 15),  # G: 6
            Buffer("H", 30, 7, 8),  # H: 7
            Buffer("I", 30, 8, 9),  # I: 8
            Buffer("J", 15, 9, 16),  # J: 9
            Buffer("K", 15, 10, 12),  # K: 10
            Buffer("L", 15, 11, 12),  # L: 11
            Buffer("M", 15, 12, 13),  # M: 12
            Buffer("N", 30, 13, 15),  # N: 13
            Buffer("O", 45, 14, 15),  # O: 14
            Buffer("P", 30, 15, 16),  # P: 15 (in-place)
            Buffer("Q", 75, 16, 17),  # Q: 16
        ]

        if False:
            # Original - no in-place: 150
            pass
        elif True:
            # P is in-place from G: 120
            buffers[6] = Buffer("PG", 30, 6, 16)
            del buffers[15]
        else:
            # P is in-place from N: 135
            buffers[13] = Buffer("PN", 30, 13, 16)
            del buffers[15]

        def schedule() -> Iterable[float]:
            return ExponentialCoolingSchedule(
                t0=10.0, t_end=1.0, steps_per_epoch=10, epochs=250
            )

        import time

        N = 100
        start = time.perf_counter()
        allocator = ImanishiXuAllocator(
            buffers=buffers,
            random=random,
            order="first_fit",
            schedule=schedule(),  # type: ignore[has-type]
            ordering_fuzz_factor=1000.0,
            starts=N,
        )
        allocator.solve()
        end = time.perf_counter()
        print(f"Time per iteration: {(end - start) / N}")

        from collections import Counter

        results = Counter(h[-1] for h in allocator.height_logs)
        print(f"Results: {results}")

        try:
            fig = allocator.plot()
            fig.savefig("plot.png", dpi=300)

        except ImportError:
            print("Not creating plot (matplotlib not installed)")
