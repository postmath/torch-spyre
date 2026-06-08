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
# Algorithm 4 is the simulated annealing algorithm that comes up with the permutation. It takes as
# inputs an annealing schedule, a list of buffers, and an initial permutation. One iteration
# randomly selects a buffer, and then cleverly compares all possible positions where the buffer
# could be reinserted. In effect, it cheaply considers (n-1) neighbours every iteration.
#
# In order to adjust this algorithm to our setting, we use CappedAllocatorPlan from plan_solver. It
# allows us to use a permutation of buffers as a source of a layout plan, and modify the permutation
# and see the modification in the layout plan by repeated swapping. We also adjust our random
# sampling: a buffer that is currently allocated legally gets to consider being inserted into all
# other positions, whereas a buffer that is not currently allocated legally only gets to consider
# being reinserted in positions of (nearly) legally allocated buffers, so that we don't spend too
# much time on swaps that have no effect.

import math
import copy
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Iterator, Optional, override
import random as rnd
from heapq import heappush, heappop

from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    PermutationBasedLayoutSolver,
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)


class CoolingSchedule(ABC):
    """A *responsive* temperature controller for simulated annealing.

    Unlike a blind temperature iterator, after every step the annealer reports
    whether the step accepted a move, so a schedule may adapt -- e.g. detect a
    stall and reheat. :meth:`reset` begins a fresh anneal and returns the first
    temperature; :meth:`update` consumes the latest step's acceptance and
    returns the next temperature, or ``None`` to stop. ``reset`` must fully
    reinitialize transient state (so a schedule can be reused across anneals).
    """

    @abstractmethod
    def reset(self) -> Optional[float]:
        """Reinitialize and return the first temperature (None for no steps)."""

    @abstractmethod
    def update(self, accepted: bool) -> Optional[float]:
        """Return the next temperature given the last step's acceptance, or None
        to stop."""


class ExponentialCoolingSchedule(CoolingSchedule):
    """Geometric cooling over ``steps_per_epoch * epochs`` steps, dropping by a
    constant factor once per epoch. Ignores acceptance."""

    def __init__(self, *, t0: float, t_end: float, steps_per_epoch: int, epochs: int):
        self.t0 = t0
        self.alpha = (t_end / t0) ** (1 / epochs)
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self._t = t0
        self._i = 0

    @override
    def reset(self) -> Optional[float]:
        self._t = self.t0
        self._i = 0
        return self._t

    @override
    def update(self, accepted: bool) -> Optional[float]:
        self._i += 1
        if self._i >= self.steps_per_epoch * self.epochs:
            return None
        if self._i % self.steps_per_epoch == 0:
            self._t *= self.alpha
        return self._t


class CoolingScheduleFromPaper(CoolingSchedule):
    """Log-linear schedule between tau_s and tau_e derived from the peak memory
    load, as in the paper. Ignores acceptance."""

    def __init__(self, *, buffers: list[LifetimeBoundBuffer], n: int = 1000000):
        buffers_sorted = sorted(buffers, key=lambda b: b.start_time)
        current_load = 0
        peak_load = 0
        # When we encounter a buffer, we include (last_use, size) in end_points, which is a min-heap
        # ordered by last_use.
        end_points: list[tuple[int, int]] = []
        for buffer in buffers_sorted:
            while end_points and end_points[0][0] <= buffer.start_time:
                current_load -= heappop(end_points)[1]

            current_load += buffer.size
            peak_load = max(peak_load, current_load)

            heappush(end_points, (buffer.end_time, buffer.size))

        tau_s = peak_load / 300.0
        tau_e = min(100.0, tau_s / 1000.0)
        self.log_tau_s = math.log(tau_s)
        self.log_tau_e = math.log(tau_e)
        self.n = n
        self._i = 0

    @override
    def reset(self) -> Optional[float]:
        self._i = 0
        return math.exp(self.log_tau_s)

    @override
    def update(self, accepted: bool) -> Optional[float]:
        self._i += 1
        if self._i >= self.n:
            return None
        return math.exp(
            (self.log_tau_e - self.log_tau_s) * self._i / self.n + self.log_tau_s
        )


class IterableCoolingSchedule(CoolingSchedule):
    """Adapts a plain iterable of temperatures to the responsive interface,
    ignoring acceptance. ``reset`` restarts iteration from the source, so the
    source should be re-iterable (a list, not a one-shot generator)."""

    def __init__(self, temperatures: Iterable[float]):
        self._source = temperatures
        self._it: Optional[Iterator[float]] = None

    @override
    def reset(self) -> Optional[float]:
        self._it = iter(self._source)
        return next(self._it, None)

    @override
    def update(self, accepted: bool) -> Optional[float]:
        assert self._it is not None
        return next(self._it, None)


class ReheatingSchedule(CoolingSchedule):
    """Locate the productive ("critical") temperature, then warm-restart around
    it.

    Phase 1 (cool): start at ``t0`` and multiply by ``alpha`` every step,
    tracking the acceptance rate over a sliding window of the last ``window``
    steps. When that rate first drops below ``stall_rate`` (the chain has
    frozen at the current temperature), record that temperature as ``T1``.

    Phase 2 (reheat): perform ``restarts`` cycles, each cooling by ``alpha``
    from ``T1 * delta`` down to ``T1 / delta`` -- a fixed band around the
    critical temperature -- then stop. This concentrates the budget where moves
    are useful but not frozen, rather than re-cooling from a high temperature.

    The acceptance signal makes phase 1 adaptive; the band cycling is fixed.
    A cycle is ``2 * ln(delta) / ln(1/alpha)`` steps, so a run is roughly
    ``len(phase 1) + restarts * cycle_length`` steps.
    """

    def __init__(
        self,
        *,
        t0: float,
        alpha: float,
        window: int,
        stall_rate: float,
        delta: float,
        restarts: int,
        min_temp: Optional[float] = None,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if delta <= 1.0:
            raise ValueError("delta must be > 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 0.0 <= stall_rate <= 1.0:
            raise ValueError("stall_rate must be in [0, 1]")
        if restarts < 0:
            raise ValueError("restarts must be >= 0")
        self.t0 = t0
        self.alpha = alpha
        self.window = window
        self.stall_rate = stall_rate
        self.delta = delta
        self.restarts = restarts
        # Safety floor: cooling reaches zero acceptance eventually, but guard
        # against never stalling (e.g. stall_rate == 0).
        self.min_temp = min_temp if min_temp is not None else t0 * 1e-12

    @override
    def reset(self) -> Optional[float]:
        self._phase = "cool"
        self._t = self.t0
        self._recent: deque[bool] = deque()
        self._accepts = 0
        self._t1: Optional[float] = None
        self._cycles_done = 0
        return self._t

    @override
    def update(self, accepted: bool) -> Optional[float]:
        if self._phase == "cool":
            self._recent.append(accepted)
            self._accepts += int(accepted)
            if len(self._recent) > self.window:
                self._accepts -= int(self._recent.popleft())
            stalled = (
                len(self._recent) == self.window
                and self._accepts / self.window < self.stall_rate
            )
            if stalled or self._t <= self.min_temp:
                self._t1 = self._t  # critical temperature
                if self.restarts <= 0:
                    return None
                self._phase = "reheat"
                self._t = self._t1 * self.delta
                return self._t
            self._t *= self.alpha
            return self._t

        # reheat: cool within the band, cycling `restarts` times.
        assert self._t1 is not None
        self._t *= self.alpha
        if self._t <= self._t1 / self.delta:
            self._cycles_done += 1
            if self._cycles_done >= self.restarts:
                return None
            self._t = self._t1 * self.delta  # next cycle
        return self._t


class SolverToPermutation:
    def __init__(self, solver: MemoryPlanSolver):
        self.solver = solver

    def permutation(self, buffers: list[LifetimeBoundBuffer]) -> list[int]:
        """Lay out the given buffers, then sort them by their addresses. Any non-allocated buffers
        come after all allocated buffers. Return this ordering as a list of indices; the first index
        is i such that buffers[i] is one of the buffers allocated at address 0, etc. This yields a
        permutation that gives the given layout, or an equivalent one, or occasionally even a better
        one."""
        allocated_buffers = self.solver.plan_layout(copy.deepcopy(buffers))
        # Typically, allocated_buffers is just the argument to plan_layout, which has been modified
        # in-place. But we can't assume that. Moreover, we need to protect the passed in buffers
        # from being modified by the given solver.

        max_address = max(
            (b.address for b in allocated_buffers if b.address is not None), default=0
        )
        name_to_address = {
            b.name: (b.address if b.address is not None else max_address + 1)
            for b in allocated_buffers
        }
        return sorted(
            list(range(len(buffers))), key=lambda i: name_to_address[buffers[i].name]
        )


class ImanishiXuLayoutSolver(MemoryPlanSolver):
    """We can only do the full initialization when we know the list of buffers, so this class is
    just a shim to create the actual solver."""

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        *,
        initial: list[int] | str | MemoryPlanSolver = "first_fit",
        schedule: "CoolingSchedule | Iterable[float] | str" = "from_paper",
        random: Optional[rnd.Random] = None,
        starts: int = 1,
    ):
        super().__init__(size, alignment)
        self.initial = initial
        self.schedule = schedule
        self.random = random
        self.starts = starts

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        solver = ImanishiXuSolverWithBuffers(
            buffers,
            self.limit,
            self.alignment,
            initial=self.initial,
            schedule=self.schedule,
            random=self.random,
            starts=self.starts,
        )
        solver.solve()
        solver.finalize()
        return buffers


class ImanishiXuSolverWithBuffers(PermutationBasedLayoutSolver):
    def __init__(
        self,
        buffers: list[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
        *,
        initial: list[int] | str | MemoryPlanSolver = "first_fit",
        schedule: "CoolingSchedule | Iterable[float] | str" = "from_paper",
        random: Optional[rnd.Random] = None,
        starts: int = 1,
    ):
        if isinstance(initial, list):
            self.initial = initial
            if not sorted(self.initial) == list(range(len(buffers))):
                raise ValueError(
                    f"given initial list is not a permutation of range({len(buffers)})"
                )
        else:
            if initial == "first_fit":
                initial = FirstFitLayoutSolver(size, alignment)
            elif initial == "best_fit":
                initial = BestFitLayoutSolver(size, alignment)
            elif initial == "greedy":
                initial = GreedyLayoutSolver(size, alignment)
            elif isinstance(initial, str):
                raise ValueError(
                    f"this string does not describe a known solver: {initial}"
                )

            assert isinstance(initial, MemoryPlanSolver)
            convertor = SolverToPermutation(initial)
            self.initial = convertor.permutation(buffers)

        super().__init__(buffers, self.initial, size, alignment)

        self.buffers = buffers
        self.starts = starts
        self.quality_logs: list[list[int]] = []
        self.best_quality = self.quality()
        self.best_permutation = copy.copy(self.initial)

        if isinstance(schedule, str):
            if schedule == "from_paper":
                self.schedule: CoolingSchedule = CoolingScheduleFromPaper(
                    buffers=buffers
                )
            else:
                raise ValueError(
                    f"this string does not describe a known schedule: {schedule}"
                )
        elif isinstance(schedule, CoolingSchedule):
            self.schedule = schedule
        else:
            self.schedule = IterableCoolingSchedule(schedule)

        if random:
            self.random = random
        else:
            self.random = rnd.Random()

    def solve(self) -> None:
        for _ in range(self.starts):
            self.anneal()
        # Restore the best permutation seen so finalize() commits it rather than
        # whatever state annealing happened to end in.
        if self.permutation != self.best_permutation:
            self.permutation = copy.copy(self.best_permutation)
            self._build()

    def anneal(self) -> None:
        quality_log: list[int] = []

        temperature = self.schedule.reset()
        while temperature is not None:
            move = self.annealing_step_rotate(temperature)
            if move is not None:
                self.annealing_step_swap(*move)

            quality = self.quality()
            quality_log.append(quality)
            if quality > self.best_quality:
                self.best_quality = quality
                self.best_permutation = copy.copy(self.permutation)

            temperature = self.schedule.update(move is not None)

        self.quality_logs.append(quality_log)

    def annealing_step_swap(self, i: int, j: int) -> None:
        """This is the loop mentioned as Algorithms 5 and 6 in the paper."""
        assert i != j, (
            "for a rotation i -> i, we should return None from the rotation method"
        )
        assert 0 <= i < len(self.permutation)
        assert 0 <= j < len(self.permutation)

        if i > j:
            i, j = j, i
        # Now i < j, and self.permutation[:i] and self.permutation[j+1:] are "clean"; that is, there
        # is no k such that self.permutation[k] and self.permutation[k+1] are buffers that *do not
        # overlap* in time, and have self.permutation[k] have a higher end point in memory than
        # self.permutation[k+1]. Because self.permutation[i] up to and including self.permutation[j]
        # changed, we need to examine i-1 <= k <= j -- except if that would take us outside the
        # bounds of self.permutation, of course.
        i -= 1

        # Ensure that both i and j+1 are valid indices.
        if i < 0:
            i = 0
        if j == len(self.permutation) - 1:
            j = len(self.permutation) - 2

        while i <= j:
            pi = self.permutation[i]
            pi1 = self.permutation[i + 1]

            if (not self._overlaps(pi, pi1)) and self.addresses[pi] + self.buffers[
                pi
            ].size > self.addresses[pi1] + self.buffers[pi1].size:
                # Swap buffers pi and pi1. This makes no difference for the quality of the result
                # *now*, but it makes it easier to rotate to an improved state.
                self.swap(i)

                # Adjust the bounds of what we need to examine.
                if i == j and j < len(self.permutation) - 2:
                    j += 1
                if i > 0:
                    i -= 1
                else:
                    i = 1
            else:
                i += 1

    def annealing_step_rotate(self, temperature: float) -> Optional[tuple[int, int]]:
        """This is the inner loop of Algorithm 4 from the paper. The return value is (i, j) iff we
        accepted a rotation inserting entry i of the permutation into position j != i; None if we
        accepted no rotation. We never accept a trivial rotation."""
        n = len(self.buffers)
        allocated = [self._is_fully_allocated(self.permutation[i]) for i in range(n)]
        n_allocated = sum(1 if b else 0 for b in allocated)
        # Choose each allocated buffer with weight n and each non-allocated buffer with weight
        # n_allocated + 1.
        i = self.random.choices(
            range(n), weights=[n if b else n_allocated + 1 for b in allocated]
        )[0]

        # qualities[j] is the quality if we rotate i to position j in the permutation, or None if we
        # don't consider rotating i to position j.
        qualities: list[Optional[int]] = [None] * n
        quality_before = self.quality()

        # Consider all reinsertion positions. First rotate i to position 0, then bubble it forward
        # one step at a time, recording the quality at each position it visits. Buffer x ends at
        # position ``upper_bound``.
        if i != 0:
            self.rotate(i, 0)
            qualities[0] = self.quality()
        if allocated[i]:
            upper_bound = n - 1
        else:
            # x is not legally allocated, so it can only be made to fit by moving it earlier; the
            # last legally-allocated buffer sits at position k, so only positions 0..k+1 can change
            # the quality. (See the monotonicity argument: x's address is non-decreasing in its
            # position.)
            upper_bound = (
                max((pos for pos, b in enumerate(allocated) if b), default=0) + 1
            )
            if upper_bound > n - 1:
                upper_bound = n - 1

        for p in range(1, upper_bound + 1):
            self.swap(p - 1)  # bubble x from position p-1 to position p
            if p != i:
                qualities[p] = self.quality()

        insertion_points = [pos for pos, q in enumerate(qualities) if q is not None]
        insertion_points = sorted(
            insertion_points,
            key=lambda pos: -qualities[pos],  # type: ignore
        )

        for j in insertion_points:
            assert i != j
            qj = qualities[j]
            assert qj is not None
            if qj > quality_before or self.random.random() < math.exp(
                (qj - quality_before) / temperature
            ):
                self.rotate(upper_bound, j)
                return (i, j)

        # Nothing accepted: leave the chain where it was by restoring x to position i.
        self.rotate(upper_bound, i)
        return None
