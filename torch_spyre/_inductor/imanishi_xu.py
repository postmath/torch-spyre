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
from typing import Iterable, Optional, override, Callable
import random as rnd
import numpy as np


@dataclass
class Buffer:
    size: int
    # The buffer is allocated from tick first_use up to, but not including, last_use.
    first_use: int
    last_use: int

    def __post_init__(self):
        # The original paper doesn't require this, so that it can support cyclic/periodic
        # allocations. We could relatively easily extend the code to allow for this.
        assert self.last_use > self.first_use

    @classmethod
    def random(
        cls, size_range: int, time_range: int, random: Optional[rnd.Random] = None
    ) -> "Buffer":
        if random is None:
            random = rnd.Random()

        duration = random.randrange(time_range - 1)
        duration = (
            duration * duration // (time_range - 1)
        )  # Bias towards smaller time ranges
        t_start = random.randrange(time_range - duration)
        t_end = t_start + duration + 1

        return cls(size=random.randrange(size_range), first_use=t_start, last_use=t_end)


@dataclass
class BufferList:
    _list: list[Buffer]
    max_time: int

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, i: int) -> Buffer:
        return self._list[i]

    @classmethod
    def from_buffers(cls, buffers: list[Buffer]) -> "BufferList":
        max_time = max(b.last_use for b in buffers) if buffers else 0
        return cls(buffers, max_time)


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

        # height[i] is the max height of all currently allocated blocks at time i.
        #
        # NOTE: The operations on 'height' are "max of a sub-array" and "set a sub-array to a
        # constant value". There are specialized data structures for that, I think, but I doubt that
        # they will be worth it. But an ndarray should be faster than a list.
        height = np.zeros(buffers.max_time, dtype="int64")
        addresses = np.zeros(n, dtype="int64")
        for j in order:
            buffer = buffers[j]
            # Allocate buffer on top of currently allocated blocks.
            addresses[j] = np.max(height[buffer.first_use : buffer.last_use])
            height[buffer.first_use : buffer.last_use] = addresses[j] + buffer.size

        return np.max(height), cls(buffers, list(addresses))


class ExponentialCoolingSchedule:
    def __init__(self, *, t0: float, alpha: float, steps_per_epoch: int, epochs: int):
        self.t = t0
        self.alpha = alpha
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.i = 0

    def __iter__(self) -> "ExponentialCoolingSchedule":
        return self

    def __next__(self) -> float:
        self.i += 1
        if self.i % self.steps_per_epoch == 0:
            if self.i == self.steps_per_epoch * self.epochs:
                raise StopIteration
            self.t *= self.alpha
        return self.t


class CoolingScheduleFromPaper:
    def __init__(self, *, buffers: BufferList, n: int = 1000000):
        buffers_sorted = sorted(buffers._list, key=lambda b: b.first_use)
        current_load = 0
        peak_load = 0
        # When we encounter a buffer, we include (last_use, size) in end_points, which is a min-heap
        # ordered by last_use.
        end_points: list[tuple[int, int]] = []
        for buffer in buffers_sorted:
            while end_points and end_points[0][0] <= buffer.first_use:
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

    def __iter__(self) -> "CoolingScheduleFromPaper":
        return self

    def __next__(self) -> float:
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return math.exp(
            (self.log_tau_s - self.log_tau_e) * self.i / self.n + self.log_tau_s
        )


class DeterministicHeuristic:
    def __init__(self, buffers: list[Buffer] | BufferList):
        self.buffers = (
            buffers
            if isinstance(buffers, BufferList)
            else BufferList.from_buffers(buffers)
        )
        self.order = self.compute()

    def __call__(self) -> list[int]:
        return self.order

    def compute(self) -> list[int]:
        raise NotImplementedError("This is an abstract class")


class FirstFit(DeterministicHeuristic):
    @staticmethod
    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or a[0] >= b[1])

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
        allocations = np.zeros(len(buffers_sorted), dtype="int64")

        for i, buffer in enumerate(buffers_sorted):
            # A list of pairs (a, b) such that b - a >= buffer.size and gaps[i][1] < gaps[i+1][0],
            # and we have not seen any buffers overlapping (a, b).
            large_gaps = [(0, total_buffer_size)]

            for j, other_buffer in enumerate(buffers_sorted[:i]):
                if not FirstFit.overlaps(
                    (buffer.first_use, buffer.last_use),
                    (other_buffer.first_use, other_buffer.last_use),
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
                        a = mid
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
                        b1 = mid

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
            return large_gaps[1][0]

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
    ):
        """Implement later: good initial permutation (the paper suggests obtaining it from
        first-fit)."""
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
                    alpha=math.exp(-math.log(50) / root),
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

    def solve(self):
        for i, temperature in enumerate(self.schedule):
            if i % 2 == 0:
                self.annealing_step_rotate(temperature)
            else:
                self.annealing_step_swap(temperature)

            height = Allocations.from_order(self.buffers, self.order)[0]
            if height < self.best_height:
                self.best_height = height
                self.best_order = self.order

    def annealing_step_swap(self, temperature: float):
        """This is the loop mentioned as Algorithm 5 in the paper."""
        # TODO - implement this!

    def annealing_step_rotate(self, temperature: float):
        """This is the inner loop of Algorithm 4 from the paper."""
        n = len(self.buffers)
        i = self.random.randrange(n)
        buffer = self.buffers[self.order[i]]

        order_minus_i = self.order[:i] + self.order[i + 1 :]
        height_order_minus_i = Allocations.from_order(self.buffers, order_minus_i)[0]

        # "Up sweep".
        bottom_heights = np.zeros(self.buffers.max_time, dtype="int64")
        bottom_addresses = np.zeros(n, dtype="int64")
        for j, other_i in enumerate(order_minus_i):
            other_buffer = self.buffers[other_i]
            print(f"Up sweep: j={j}, other_i={other_i}, other_buffer={other_buffer}")
            # Consider the order that would be given by order_minus_i.insert(j, order[i]). Compute
            # the address where we would allocate buffer i given the order:
            #   order_minus_i[:j] + [order[i]].
            bottom_addresses[j] = np.max(
                bottom_heights[buffer.first_use : buffer.last_use]
            )
            # For the next iteration, plan to allocate other_buffer next.
            other_allocation = np.max(
                bottom_heights[other_buffer.first_use : other_buffer.last_use]
            )
            bottom_heights[other_buffer.first_use : other_buffer.last_use] = (
                other_allocation + other_buffer.size
            )
        bottom_addresses[n - 1] = np.max(
            bottom_heights[buffer.first_use : buffer.last_use]
        )
        print(f"Bottom addresses: {bottom_addresses}")

        # "Down sweep".
        top_heights = np.zeros(self.buffers.max_time, dtype="int64")
        top_addresses = np.zeros(n, dtype="int64")
        for j in range(n - 1, 0, -1):
            other_i = order_minus_i[j - 1]
            other_buffer = self.buffers[other_i]
            print(f"Down sweep: j={j}, other_i={other_i}, other_buffer={other_buffer}")
            # Consider the order that would be given by order_minus_i.insert(j, order[i]). We now
            # compute the address where we would allocate buffer i given the order:
            #   reversed(order_minus_i[j:]) + [order[i]].
            # Imagine this allocation "upside down", hanging from the ceiling. We can obtain the
            # full height of the allocation order_minus_i.insert(j, order[i]) as
            #   max(height_order_minus_i, bottom_heights[j] + top_heights[j] + buffer.size).
            top_addresses[j] = np.max(top_heights[buffer.first_use : buffer.last_use])

            # For the next iteration, plan to allocate other_buffer next.
            other_allocation = np.max(
                top_heights[other_buffer.first_use : other_buffer.last_use]
            )
            top_heights[other_buffer.first_use : other_buffer.last_use] = (
                other_allocation + other_buffer.size
            )
        top_addresses[0] = np.max(top_heights[buffer.first_use : buffer.last_use])
        print(f"Top addresses: {top_addresses}")

        # Actual total heights are the pointwise max of heights_through_buffer with
        # height_order_minus_i, but as the paper explains on page 89, it's better to work with just
        # heights_through_buffer.
        height_through_buffer = bottom_addresses + top_addresses + buffer.size

        # The paper does not explain how they select a rotation if multiple rotations are accepted.
        # We implement this by sorting the insertion points descending by improvement and choosing
        # the first accepted improvement. (An alternative might be to choose an insertion point
        # among the accepted ones by softmax of the improvement.)
        insertion_points = sorted(range(n), key=lambda j: -height_through_buffer[j])

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

                return

        # No rotation was accepted.


if __name__ == "__main__":
    if False:
        buffers = [
            Buffer(1, 0, 2),
            Buffer(3, 0, 2),
            Buffer(4, 2, 3),
            Buffer(5, 0, 1),
            Buffer(3, 1, 4),
        ]
        bl = BufferList.from_buffers(buffers)
        order = [0, 2, 3, 1, 4]
        schedule = ExponentialCoolingSchedule(
            t0=10.0, alpha=0.8, steps_per_epoch=10, epochs=10
        )
        allocator = ImanishiXuAllocator(buffers=bl, order=order, schedule=schedule)
        allocator.annealing_step_rotate()

    else:
        rnd.seed(0)
        N = 100
        buffers = [Buffer.random(1000000, N) for _ in range(N)]
        allocator = ImanishiXuAllocator(buffers=buffers)
        allocator.solve()
