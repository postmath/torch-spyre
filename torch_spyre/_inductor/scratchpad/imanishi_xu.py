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
# In order to adjust this algorithm to our setting, we hold a PermutationBasedLayoutSolver from
# plan_solver as a member. It lets us use a permutation of buffers as a source of a layout plan, and
# modify the permutation and see the modification in the layout plan by repeated swapping. Each
# reinsertion sweep runs on a throwaway plan.copy(), so the live plan only performs the rotation
# that is actually accepted rather than sweeping and restoring. We also adjust our random sampling:
# a buffer that is currently allocated legally gets to consider being inserted into all other
# positions, whereas a buffer that is not currently allocated legally only gets to consider being
# reinserted in positions of (nearly) legally allocated buffers, so that we don't spend too much
# time on swaps that have no effect.

import math
import copy
from abc import ABC, abstractmethod
from collections import deque
from typing import Literal, Optional, TypeAlias, override
import random as rnd
from heapq import heappush, heappop

from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
)


def peak_memory_load(buffers: list[LifetimeBoundBuffer]) -> int:
    """Maximum total size of simultaneously-live buffers (a lower bound on the
    space any layout needs). Swept over lifetime start points."""
    by_start = sorted(buffers, key=lambda b: b.start_time)
    current_load = 0
    peak_load = 0
    end_points: list[tuple[int, int]] = []  # (end_time, size) min-heap
    for buffer in by_start:
        while end_points and end_points[0][0] <= buffer.start_time:
            current_load -= heappop(end_points)[1]
        current_load += buffer.size
        peak_load = max(peak_load, current_load)
        heappush(end_points, (buffer.end_time, buffer.size))
    return peak_load


def default_initial_temperature(buffers: list[LifetimeBoundBuffer]) -> float:
    """A principled starting temperature from the peak memory load -- the paper's
    tau_s. Used when a schedule is not given an explicit ``t0``."""
    return peak_memory_load(buffers) / 300.0


class CoolingSchedule(ABC):
    """A *responsive* temperature controller for simulated annealing.

    Unlike a blind temperature iterator, after every step the annealer reports
    whether the step accepted a move, so a schedule may adapt -- e.g. detect a
    stall and reheat. :meth:`reset` begins a fresh anneal and returns the first
    temperature; :meth:`update` consumes the latest step's acceptance and
    returns the next temperature, or ``None`` to stop. ``reset`` must fully
    reinitialize transient state (so a schedule can be reused across anneals).
    """

    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        """Preparation hook: the solver calls this with the buffer set before
        annealing, so a schedule may derive parameters (e.g. an initial
        temperature from the peak load). Default: no-op."""

    @abstractmethod
    def reset(self) -> Optional[float]:
        """Reinitialize and return the first temperature (None for no steps)."""

    @abstractmethod
    def update(self, accepted: bool) -> Optional[float]:
        """Return the next temperature given the last step's acceptance, or None
        to stop."""


class Calibratable(ABC):
    def warmup_steps_requested(self) -> int:
        """How many warm-up moves the solver should sample (on a throwaway plan
        copy) to calibrate this schedule. Default 0 -- no calibration needed."""
        return 0

    def calibrate(self, abs_quality_deltas: list[float]) -> None:
        """Calibration hook paired with :meth:`warmup_steps_requested`. The
        solver runs that many random moves and passes their absolute quality
        deltas here, so a schedule may size its temperatures to the instance's
        move scale before :meth:`reset` is first called. Default: no-op."""


class ExponentialCoolingSchedule(CoolingSchedule):
    """Geometric cooling over ``steps_per_epoch * epochs`` steps, dropping by a
    constant factor once per epoch. Ignores acceptance."""

    def __init__(
        self, *, t_initial: float, t_final: float, steps_per_epoch: int, epochs: int
    ):
        """A schedule that starts at temperature `t_initial` and ends at `t_final`, cooling down by
        a constant factor every `steps_per_epoch` steps. There are `epochs` such epochs.

        If `epochs == 1`, then the temperature stays at `t_initial`."""
        self.t_initial = t_initial
        if epochs <= 1:
            self.alpha = 1.0
        else:
            self.alpha = (t_final / t_initial) ** (1 / (epochs - 1))
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self._t = t_initial
        self._i = 0

    @override
    def reset(self) -> Optional[float]:
        self._t = self.t_initial
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


class SelfCalibratingCoolingSchedule(CoolingSchedule, Calibratable):
    """Geometric cooling whose start/end temperatures are auto-calibrated to the
    instance from a short warm-up. This is the default schedule.

    NOTE: this is **not** claimed to be optimal or well-tuned -- it is a
    reasonable, self-calibrating default for the situation where we do *not* yet
    have representative example models to tune against. It pairs the universal SA
    cooling law (a fixed-length geometric cool-down) with temperatures sized from
    the data, so the same schedule transfers across problems of very different
    byte scales instead of relying on hand-picked constants (which would silently
    degenerate into a random walk or a greedy search on a differently-scaled
    instance). Expect to revisit it -- retune the budget, or move to an
    acceptance-targeting adaptive schedule -- once we can benchmark on real
    workloads.

    Calibration: the solver samples ``warmup_steps`` random reinsertions on a
    throwaway ``plan.copy()`` and passes their ``|Δquality|`` to
    :meth:`calibrate`. With ``d`` = their mean (over moves that change quality),
    the start temperature accepts a mean-magnitude *worsening* move with
    probability ``accept_hi`` and the end temperature with probability
    ``accept_lo`` -- i.e. ``t0 = d / -ln(accept_hi)`` and
    ``t_end = d / -ln(accept_lo)`` -- so the run begins exploratory and ends
    near-greedy. Cooling is then geometric from ``t0`` to ``t_end`` over
    ``total_steps``.

    Budget knobs:
        total_steps: the annealing budget. ``None`` -> adaptive,
            ``clamp(steps_per_buffer * n, min_steps, max_steps)``.
        warmup_steps: number of calibration moves (extra, cheap work).
            ``None`` -> ``warmup_fraction`` of ``total_steps``.
        max_steps: hard cap on the adaptive budget (default 5000, keeping the
            n=100 random-buffer example bounded).
    """

    def __init__(
        self,
        *,
        total_steps: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        warmup_fraction: float = 0.1,
        steps_per_buffer: int = 30,
        min_steps: int = 500,
        max_steps: int = 5000,
        accept_hi: float = 0.8,
        accept_lo: float = 0.01,
    ):
        if not 0.0 < accept_lo < accept_hi < 1.0:
            raise ValueError("need 0 < accept_lo < accept_hi < 1")
        if not 0.0 < warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must be in (0, 1]")
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps
        self.warmup_fraction = warmup_fraction
        self.steps_per_buffer = steps_per_buffer
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.accept_hi = accept_hi
        self.accept_lo = accept_lo
        # Sized in set_buffers; the geometric cooler is built in calibrate.
        self.total_steps = total_steps or 0
        self.warmup_steps = warmup_steps or 0
        self._geom: Optional[ExponentialCoolingSchedule] = None

    @override
    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        if self._total_steps is None:
            self.total_steps = min(
                self.max_steps,
                max(self.min_steps, self.steps_per_buffer * len(buffers)),
            )
        else:
            self.total_steps = max(1, self._total_steps)
        if self._warmup_steps is None:
            self.warmup_steps = max(1, round(self.warmup_fraction * self.total_steps))
        else:
            self.warmup_steps = self._warmup_steps

    @override
    def warmup_steps_requested(self) -> int:
        return self.warmup_steps

    @override
    def calibrate(self, abs_quality_deltas: list[float]) -> None:
        nonzero = [d for d in abs_quality_deltas if d > 0]
        if nonzero:
            d = sum(nonzero) / len(nonzero)
            t0 = d / -math.log(self.accept_hi)
            t_end = d / -math.log(self.accept_lo)
        else:
            # No sampled move changed quality (degenerate instance): every
            # positive temperature behaves the same near-greedily, so pick 1.0
            # to avoid a zero temperature (the acceptance test divides by it).
            t0 = t_end = 1.0
        self._geom = ExponentialCoolingSchedule(
            t_initial=t0, t_final=t_end, steps_per_epoch=1, epochs=self.total_steps
        )

    @override
    def reset(self) -> Optional[float]:
        if self._geom is None:
            raise ValueError(
                "AutoExponentialCoolingSchedule must be calibrated before use; "
                "run it through ImanishiXuSolverWithBuffers, which samples the "
                "warm-up moves and calls calibrate()."
            )
        return self._geom.reset()

    @override
    def update(self, accepted: bool) -> Optional[float]:
        assert self._geom is not None
        return self._geom.update(accepted)


class CoolingScheduleFromPaper(CoolingSchedule):
    """Log-linear schedule between tau_s and tau_e derived from the peak memory
    load, as in the paper. Ignores acceptance."""

    def __init__(self, *, n: int = 1000000):
        self.n = n
        self._i = 0
        self.log_tau_s: Optional[float] = None
        self.log_tau_e: Optional[float] = None

    @override
    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        tau_s = default_initial_temperature(buffers)
        tau_e = min(100.0, tau_s / 1000.0)
        self.log_tau_s = math.log(tau_s)
        self.log_tau_e = math.log(tau_e)

    @override
    def reset(self) -> Optional[float]:
        self._i = 0
        if self.log_tau_s is None:
            raise RuntimeError(
                "need to set buffers before extracting values from this schedule"
            )
        return math.exp(self.log_tau_s)

    @override
    def update(self, accepted: bool) -> Optional[float]:
        self._i += 1
        if self._i >= self.n:
            return None
        if self.log_tau_s is None or self.log_tau_e is None:
            raise RuntimeError(
                "need to set buffers before extracting values from this schedule"
            )
        return math.exp(
            (self.log_tau_e - self.log_tau_s) * self._i / self.n + self.log_tau_s
        )


class ReheatingSchedule(CoolingSchedule):
    """Locate the productive ("critical") temperature, then warm-restart around
    it.

    Phase 1 (cool): start at ``t0`` and multiply by ``alpha`` every step,
    tracking the acceptance rate over a sliding window of the last ``window``
    steps. When that rate first drops below ``stall_rate`` (the chain has
    frozen at the current temperature), record that temperature as ``T1``.

    ``t0`` defaults to the peak-load estimate (:func:`default_initial_temperature`)
    when run through a solver: leave it ``None`` and the solver fills it in via
    :meth:`set_buffers`. Pass ``t0`` explicitly to override.

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
        t0: Optional[float] = None,
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
        # t0/min_temp may be None until set_buffers() derives them from the peak
        # load. An explicit t0 is never overridden.
        self._explicit_t0 = t0
        self._explicit_min_temp = min_temp
        self.t0 = t0
        self.alpha = alpha
        self.window = window
        self.stall_rate = stall_rate
        self.delta = delta
        self.restarts = restarts

    @override
    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        if self._explicit_t0 is None:
            self.t0 = default_initial_temperature(buffers)

    @override
    def reset(self) -> Optional[float]:
        if self.t0 is None:
            raise ValueError(
                "ReheatingSchedule needs t0; pass it explicitly or run via a "
                "solver, which derives it from the buffers via set_buffers()"
            )
        # Safety floor: cooling reaches zero acceptance eventually, but guard
        # against never stalling (e.g. stall_rate == 0).
        self._min_temp = (
            self._explicit_min_temp
            if self._explicit_min_temp is not None
            else self.t0 * 1e-12
        )
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
            if stalled or self._t <= self._min_temp:
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


SolverInitialOption: TypeAlias = (
    list[int] | Literal["first_fit", "best_fit", "greedy"] | MemoryPlanSolver
)
SolverScheduleOption: TypeAlias = CoolingSchedule | Literal["auto", "from_paper"]


class ImanishiXuLayoutSolver(MemoryPlanSolver):
    """We can only do the full initialization when we know the list of buffers, so this class is
    just a shim to create the actual solver."""

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        *,
        initial: SolverInitialOption = "first_fit",
        schedule: SolverScheduleOption = "auto",
        random: Optional[rnd.Random] = None,
        starts: int = 1,
    ):
        super().__init__(size, alignment)
        self.initial = initial
        self.schedule = schedule
        self.random = random
        self.starts = starts

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer], log_lx_usage: bool = False
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


class ImanishiXuSolverWithBuffers:
    """Drives simulated annealing over a :class:`PermutationBasedLayoutSolver`.

    The layout is held as a *member* (``self.plan``), not a base class. This lets
    each reinsertion sweep run on a throwaway ``plan.copy()`` while the live plan
    only ever performs the single rotation that is actually accepted (and the
    cleanup swaps) -- so the live layout is never churned through a full sweep
       initial: 'list[int] | Literal["first_fit", "best_fit", "greedy"] | MemoryPlanSolver' = "first_fit",
    """

    def __init__(
        self,
        buffers: list[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
        *,
        initial: SolverInitialOption = "first_fit",
        schedule: SolverScheduleOption = "auto",
        random: Optional[rnd.Random] = None,
        starts: int = 1,
    ):
        if isinstance(initial, list):
            if sorted(initial) != list(range(len(buffers))):
                raise ValueError(
                    f"given initial list is not a permutation of range({len(buffers)})"
                )
            self.initial = initial
        else:
            if initial == "first_fit":
                initial = FirstFitLayoutSolver(size, alignment)
            elif initial == "best_fit":
                initial = BestFitLayoutSolver(size, alignment)
            elif initial == "greedy":
                initial = GreedyLayoutSolver(size, alignment)

            assert isinstance(initial, MemoryPlanSolver)
            convertor = SolverToPermutation(initial)
            self.initial = convertor.permutation(buffers)

        self.buffers = buffers
        self.size = size
        self.alignment = alignment
        self.plan = PermutationBasedLayoutSolver(buffers, self.initial, size, alignment)
        self.starts = starts
        self.quality_logs: list[list[float]] = []
        self.temperature_logs: list[list[float]] = []
        self.best_quality = self.plan.quality()
        self.best_permutation = copy.copy(self.initial)

        if isinstance(schedule, str):
            self.schedule: CoolingSchedule
            if schedule == "auto":
                self.schedule = SelfCalibratingCoolingSchedule()
            elif schedule == "from_paper":
                self.schedule = CoolingScheduleFromPaper()
            else:
                raise ValueError(
                    f"this string does not describe a known schedule: {schedule}"
                )
        else:
            self.schedule = schedule
        # Let the schedule derive any buffer-dependent parameters (e.g. t0).
        self.schedule.set_buffers(buffers)

        if random is not None:
            self.random = random
        else:
            # Default to a fixed seed so layout planning is deterministic: the
            # same graph must compile to the same scratchpad layout across runs
            # (build reproducibility and compilation caching depend on it). Pass
            # an explicit Random to vary the search (e.g. in benchmarks/tests).
            self.random = rnd.Random(0)

    def finalize(self) -> None:
        self.plan.finalize()

    def _is_optimal(self) -> bool:
        """True once every buffer is fully allocated below capacity.

        Quality is then at its upper bound -- each buffer already contributes
        its full :func:`buffer_quality`, so no rotation or swap can improve it
        and the search can stop.
        """
        return self.plan.count_allocated() == len(self.buffers)

    def solve(self) -> None:
        # If the initial layout already fits every buffer it is globally
        # optimal, so skip calibration and annealing outright. (Once annealing
        # is under way the inner loop's own check terminates it; this guard is
        # only ever reached with the untouched initial plan.)
        if self._is_optimal():
            return
        self._calibrate_schedule()
        for _ in range(self.starts):
            self.anneal()
        # Commit the best permutation seen, so finalize() writes it rather than
        # whatever state annealing happened to end in.
        if self.plan.permutation != self.best_permutation:
            self.plan = PermutationBasedLayoutSolver(
                self.buffers, list(self.best_permutation), self.size, self.alignment
            )

    def _calibrate_schedule(self) -> None:
        """If the schedule asks for a warm-up, sample that many random moves on a
        throwaway plan copy and hand it their absolute quality deltas, so it can
        size its temperatures to this instance's move scale. The live plan is
        untouched (the warm-up walks the copy)."""
        if not isinstance(self.schedule, Calibratable):
            return
        k = self.schedule.warmup_steps_requested()
        if k <= 0:
            return
        n = len(self.buffers)
        deltas: list[float] = []
        if n >= 2:  # with <2 buffers there is no non-trivial move to sample
            probe = self.plan.copy()
            for _ in range(k):
                i = self.random.randrange(n)
                j = self.random.randrange(n)
                while j == i:
                    j = self.random.randrange(n)
                # rotate() returns the quality change; its magnitude is the move
                # scale we want (a random reinsertion stands in for the real
                # annealing move for calibration purposes).
                deltas.append(abs(probe.rotate(i, j)))
        self.schedule.calibrate(deltas)

    def anneal(self) -> None:
        quality_log: list[float] = []
        temperature_log: list[float] = []

        temperature = self.schedule.reset()
        while temperature is not None:
            move = self.annealing_step_rotate(temperature)
            if move is not None:
                self.annealing_step_swap(*move)

            quality = self.plan.quality()
            quality_log.append(quality)
            temperature_log.append(temperature)
            if quality > self.best_quality:
                self.best_quality = quality
                self.best_permutation = copy.copy(self.plan.permutation)

            if self._is_optimal():
                # All buffers fit: quality is maximal, so stop cooling early --
                # the best layout above is this (globally optimal) one.
                break

            temperature = self.schedule.update(move is not None)

        self.quality_logs.append(quality_log)
        self.temperature_logs.append(temperature_log)

    def annealing_step_swap(self, i: int, j: int) -> None:
        """This is the loop mentioned as Algorithms 5 and 6 in the paper."""
        plan = self.plan
        perm = plan.permutation
        assert i != j, (
            "for a rotation i -> i, we should return None from the rotation method"
        )
        assert 0 <= i < len(perm)
        assert 0 <= j < len(perm)

        if i > j:
            i, j = j, i
        # Now i < j, and perm[:i] and perm[j+1:] are "clean"; that is, there is no k such that
        # perm[k] and perm[k+1] are buffers that *do not overlap* in time, and have perm[k] have a
        # higher end point in memory than perm[k+1]. Because perm[i] up to and including perm[j]
        # changed, we need to examine i-1 <= k <= j -- except if that would take us outside the
        # bounds of perm, of course.
        i -= 1

        # Ensure that both i and j+1 are valid indices.
        if i < 0:
            i = 0
        if j == len(perm) - 1:
            j = len(perm) - 2

        while i <= j:
            pi = perm[i]
            pi1 = perm[i + 1]

            if (not plan.overlaps(pi, pi1)) and plan.addresses[pi] + self.buffers[
                pi
            ].size > plan.addresses[pi1] + self.buffers[pi1].size:
                # Swap buffers pi and pi1. This makes no difference for the quality of the result
                # *now*, but it makes it easier to rotate to an improved state.
                plan.swap(i)

                # Adjust the bounds of what we need to examine.
                if i == j and j < len(perm) - 2:
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
        accepted no rotation. We never accept a trivial rotation.

        The reinsertion sweep runs on a throwaway copy of the plan; only the accepted rotation (if
        any) is applied to the live plan, so the live layout never has to sweep-and-restore."""
        plan = self.plan
        n = len(self.buffers)
        allocated = [plan.is_fully_allocated(plan.permutation[i]) for i in range(n)]
        n_allocated = sum(1 if b else 0 for b in allocated)
        # Choose each allocated buffer with weight n and each non-allocated buffer with weight
        # n_allocated + 1.
        i = self.random.choices(
            range(n), weights=[n if b else n_allocated + 1 for b in allocated]
        )[0]

        # qualities[j] is the quality if we rotate i to position j in the permutation, or None if we
        # don't consider rotating i to position j.
        qualities: list[Optional[float]] = [None] * n
        quality_before = plan.quality()

        # Probe all reinsertion positions on a copy: rotate i to position 0, then bubble it forward
        # one step at a time, recording the quality at each position it visits.
        probe = plan.copy()
        if i != 0:
            probe.rotate(i, 0)
            qualities[0] = probe.quality()
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
            probe.swap(p - 1)  # bubble x from position p-1 to position p
            if p != i:
                qualities[p] = probe.quality()

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
                # Apply only the accepted rotation to the live plan (others keep their order).
                plan.rotate(i, j)
                return (i, j)

        # Nothing accepted: the live plan was never touched.
        return None
