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
from typing import Literal, Optional, Sequence, TypeAlias, override
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
    both whether the step accepted a move and the *move scale* -- the mean
    ``|Δquality|`` over the reinsertion positions it probed (ignoring no-op
    positions) -- so a schedule may adapt: detect a stall, reheat, or size its
    temperatures to the instance's move magnitudes online. :meth:`reset` begins
    a fresh anneal and returns the first temperature; :meth:`update` consumes the
    latest step's acceptance and move scale and returns the next temperature, or
    ``None`` to stop. ``reset`` must fully reinitialize transient state (so a
    schedule can be reused across anneals).
    """

    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        """Preparation hook: the solver calls this with the buffer set before
        annealing, so a schedule may derive parameters (e.g. an initial
        temperature from the peak load). Default: no-op."""

    @abstractmethod
    def reset(self) -> Optional[float]:
        """Reinitialize and return the first temperature (None for no steps)."""

    @abstractmethod
    def update(self, accepted: bool, move_scale: float) -> Optional[float]:
        """Return the next temperature given the last step's acceptance and move
        scale (mean ``|Δquality|`` over probed reinsertions, ``0.0`` if none
        changed quality), or None to stop."""


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
    def update(self, accepted: bool, move_scale: float) -> Optional[float]:
        self._i += 1
        if self._i >= self.steps_per_epoch * self.epochs:
            return None
        if self._i % self.steps_per_epoch == 0:
            self._t *= self.alpha
        return self._t


class SelfCalibratingReheatingSchedule(CoolingSchedule):
    """Self-calibrating simulated-annealing schedule with reheating cycles.

    This is the default schedule. It needs no tuning beyond the step budget: it
    sizes its temperatures to the instance online from the move scale the
    annealer streams back, locates the productive temperature, and spends the
    budget on reheating cycles around it -- concentrating moves where they are
    useful but not frozen, rather than cooling monotonically once.

    NOTE: like its predecessor this is a *reasonable* self-calibrating default,
    not a tuned or provably-good one -- we do not yet have representative example
    models to benchmark against. Two bets are unvalidated for our landscape:
    that reheating beats a single long cool, and that learning the move scale
    online beats a pre-committed warm-up. Both are bounded by the solver's
    best-seen tracking, so they can waste budget but never worsen the result.
    Expect to revisit ``cycles`` and the acceptance band once we can benchmark.

    Temperature scale (self-calibration). With ``A = -ln(accept_hi)`` and
    ``B = -ln(accept_lo)``, a band centered on temperature ``center`` accepts a
    mean-magnitude *worsening* move with probability ``accept_hi`` at its top
    ``center * delta`` and ``accept_lo`` at its bottom ``center / delta``, where::

        delta  = sqrt(B / A)              # band half-width, scale-independent
        center = d_hat / sqrt(A * B)      # productive temperature

    ``d_hat`` is an exponential moving average of the streamed move scale (mean
    ``|Δquality|`` over probed reinsertions), so ``center`` tracks the move scale
    as the layout improves and the landscape flattens.

    Bootstrap. Before any move scale is known, ``center`` is seeded from the
    peak-load estimate (:func:`default_initial_temperature`), placed at the band
    top. That estimate is in bytes rather than quality units, so it is only a
    rough magnitude -- but it governs a single step before the first real
    samples snap ``center`` onto the data scale, and best-seen tracking absorbs
    that step regardless.

    Reheating. The budget is split into ``cycles`` equal cycles (the last
    absorbing any remainder). Each cools geometrically from ``center * delta``
    down to ``center / delta``. ``center`` is recomputed from ``d_hat`` *every
    step*, so the band drifts downward (or re-expands) with the landscape
    continuously rather than jumping only at cycle boundaries -- which matters
    most when cycles are long (and for ``cycles = 1``, a single tracked cool,
    the only case where the boundary-only variant never re-centered at all).
    The EMA horizon is ``cycle_len / horizons_per_cycle``: with the default
    ``horizons_per_cycle = 2`` the center lags the move scale by about half a
    cycle, so a stale band never persists for a large fraction of a cycle.

    Budget knobs:
        total_steps: the annealing budget (temperatures emitted). ``None`` ->
            adaptive, ``clamp(steps_per_buffer * n, min_steps, max_steps)``.
        cycles: number of reheating cycles.
        horizons_per_cycle: EMA horizons per cycle (``H`` in the design notes);
            the move-scale EMA horizon is ``cycle_len / horizons_per_cycle``, so
            larger values track the landscape faster (at the cost of more noise
            and a stronger pull toward greedy cooling). Guessed default pending
            benchmarks, like ``cycles``.
        max_steps: hard cap on the adaptive budget (default 5000, keeping the
            n=100 random-buffer example bounded).
    """

    def __init__(
        self,
        *,
        total_steps: Optional[int] = None,
        cycles: int = 4,
        horizons_per_cycle: float = 2.0,
        steps_per_buffer: int = 30,
        min_steps: int = 500,
        max_steps: int = 5000,
        accept_hi: float = 0.8,
        accept_lo: float = 0.01,
    ):
        if not 0.0 < accept_lo < accept_hi < 1.0:
            raise ValueError("need 0 < accept_lo < accept_hi < 1")
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        if horizons_per_cycle <= 0.0:
            raise ValueError("horizons_per_cycle must be > 0")
        self._total_steps = total_steps
        self.cycles = cycles
        self.horizons_per_cycle = horizons_per_cycle
        self.steps_per_buffer = steps_per_buffer
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.accept_hi = accept_hi
        self.accept_lo = accept_lo
        # The reheat band is fixed by the two acceptance targets alone: a factor
        # `delta` above/below the center, with `sqrt(A*B)` converting the move
        # scale into the center temperature. Both are scale-independent.
        a = -math.log(accept_hi)
        b = -math.log(accept_lo)
        self._rt_ab = math.sqrt(a * b)
        self._delta = math.sqrt(b / a)
        # Sized in set_buffers (needs the buffer count and the peak load); _cycle_len
        # == 0 marks "not yet sized" so reset() can refuse to run uncalibrated.
        self.total_steps = total_steps or 0
        self._cycle_len = 0
        self._seed_center = 1.0

    @override
    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        if self._total_steps is None:
            self.total_steps = min(
                self.max_steps,
                max(self.min_steps, self.steps_per_buffer * len(buffers)),
            )
        else:
            self.total_steps = max(1, self._total_steps)
        self._cycle_len = max(1, self.total_steps // self.cycles)
        # Cool by a factor delta^2 across one cycle (band top to band bottom).
        self._alpha = self._delta ** (-2.0 / self._cycle_len)
        # Move-scale EMA horizon of cycle_len / horizons_per_cycle steps, so the
        # center (recomputed every step) lags the landscape by that fraction of a
        # cycle. Clamped to a valid EMA rate for short cycles (where the ratio can
        # reach or exceed 1); there it degrades to "center = latest scale".
        self._ema_beta = min(1.0, self.horizons_per_cycle / self._cycle_len)
        # Average this many nonzero samples before snapping center off the seed
        # (a cheap, low-variance bootstrap; at least one).
        self._snap_after = min(self._cycle_len // 4, 20) or 1
        # Seed center so the peak-load estimate lands at the band top.
        self._seed_center = default_initial_temperature(buffers) / self._delta

    @override
    def reset(self) -> Optional[float]:
        if self._cycle_len == 0:
            raise ValueError(
                "SelfCalibratingReheatingSchedule must be given buffers before "
                "use; run it through ImanishiXuSolverWithBuffers, or call "
                "set_buffers() first."
            )
        self._i = 0
        self._s = 0
        self._cycle = 0
        self._center = self._seed_center
        self._d_hat: Optional[float] = None
        self._sample_sum = 0.0
        self._n_samples = 0
        return self._temperature()

    def _temperature(self) -> float:
        # Position s within the cycle: s == 0 is the band top (center*delta),
        # cooling by alpha each step toward the band bottom (center/delta).
        return self._center * self._delta * self._alpha**self._s

    @override
    def update(self, accepted: bool, move_scale: float) -> Optional[float]:
        # Track the move scale, ignoring no-op reinsertions (move_scale == 0):
        # they dominate the sample and would collapse the center into a greedy
        # search. Before the first snap, average a few samples; after it, EMA.
        if move_scale > 0.0:
            if self._d_hat is None:
                self._sample_sum += move_scale
                self._n_samples += 1
                if self._n_samples >= self._snap_after:
                    self._d_hat = self._sample_sum / self._n_samples
            else:
                self._d_hat += self._ema_beta * (move_scale - self._d_hat)
        # Re-center from the current move scale every step, so the band tracks
        # the landscape continuously within a cycle rather than only at its
        # boundaries. Until the first snap ``d_hat`` is None and center stays at
        # the peak-load seed.
        if self._d_hat is not None:
            self._center = self._d_hat / self._rt_ab

        self._i += 1
        if self._i >= self.total_steps:
            return None
        self._s += 1
        # Cycle boundary: restart the cool at the band top. The last cycle
        # absorbs the budget remainder. (Center already tracks every step, so the
        # boundary only restarts the carrier phase.)
        if self._s >= self._cycle_len and self._cycle < self.cycles - 1:
            self._cycle += 1
            self._s = 0
        return self._temperature()


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
    def update(self, accepted: bool, move_scale: float) -> Optional[float]:
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


class ImanishiXuLayoutSolver(MemoryPlanSolver[LifetimeBoundBuffer]):
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
    ):
        super().__init__(size, alignment)
        self.initial = initial
        self.schedule = schedule
        self.random = random

    def plan_layout(
        self, buffers: Sequence[LifetimeBoundBuffer], log_lx_usage: bool = False
    ) -> list[LifetimeBoundBuffer]:
        _buffers = list(buffers)
        solver = ImanishiXuSolverWithBuffers(
            _buffers,
            self.limit,
            self.alignment,
            initial=self.initial,
            schedule=self.schedule,
            random=self.random,
        )
        solver.solve()
        solver.finalize()
        return _buffers


class ImanishiXuSolverWithBuffers:
    """Drives simulated annealing over a :class:`PermutationBasedLayoutSolver`.

    The layout is held as a *member* (``self.plan``), not a base class. This lets
    each reinsertion sweep run on a throwaway ``plan.copy()`` while the live plan
    only ever performs the single rotation that is actually accepted (and the
    cleanup swaps) -- so the live layout is never churned through a full sweep.
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
        self.quality_logs: list[list[float]] = []
        self.temperature_logs: list[list[float]] = []
        self.best_quality = self.plan.quality()
        self.best_permutation = copy.copy(self.initial)

        if isinstance(schedule, str):
            self.schedule: CoolingSchedule
            if schedule == "auto":
                self.schedule = SelfCalibratingReheatingSchedule()
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
        # optimal, so skip annealing outright. (Once annealing is under way the
        # inner loop's own check terminates it; this guard is only ever reached
        # with the untouched initial plan.)
        if self._is_optimal():
            return
        self.anneal()
        # Commit the best permutation seen, so finalize() writes it rather than
        # whatever state annealing happened to end in.
        if self.plan.permutation != self.best_permutation:
            self.plan = PermutationBasedLayoutSolver(
                self.buffers, list(self.best_permutation), self.size, self.alignment
            )

    def anneal(self) -> None:
        quality_log: list[float] = []
        temperature_log: list[float] = []

        temperature = self.schedule.reset()
        while temperature is not None:
            move, move_scale = self.annealing_step_rotate(temperature)
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

            temperature = self.schedule.update(move is not None, move_scale)

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

        def _top_or_inf(p: int) -> float:
            # Exclusive top (address + size) of buffer ``p``, or +inf when ``p``
            # is evicted (address is None). An evicted buffer sorts as if it sits
            # arbitrarily high, so it is treated as "above" any placed buffer and
            # is never reordered below one; two placed buffers compare by their
            # real tops, unchanged from before eviction used None.
            addr = plan.addresses[p]
            if addr is None:
                return math.inf
            return addr + self.buffers[p].size

        while i <= j:
            pi = perm[i]
            pi1 = perm[i + 1]

            if (not plan.overlaps(pi, pi1)) and _top_or_inf(pi) > _top_or_inf(pi1):
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

    def annealing_step_rotate(
        self, temperature: float
    ) -> tuple[Optional[tuple[int, int]], float]:
        """This is the inner loop of Algorithm 4 from the paper. The first return value is (i, j) iff
        we accepted a rotation inserting entry i of the permutation into position j != i; None if we
        accepted no rotation. We never accept a trivial rotation. The second return value is the move
        scale for this step -- the mean |Δquality| over the reinsertion positions probed, ignoring
        no-op positions -- which the schedule uses to size its temperatures online.

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
            # position. Eviction preserves this: moving x later only adds earlier-positioned
            # overlapping buffers to what it must stack on, so its top -- and hence whether it is
            # evicted -- is monotone in position, with an evicted address read as +inf.)
            upper_bound = (
                max((pos for pos, b in enumerate(allocated) if b), default=0) + 1
            )
            if upper_bound > n - 1:
                upper_bound = n - 1

        for p in range(1, upper_bound + 1):
            probe.swap(p - 1)  # bubble x from position p-1 to position p
            if p != i:
                qualities[p] = probe.quality()

        # Move scale streamed to the schedule: mean |Δquality| over the probed
        # reinsertion positions, ignoring no-op positions (which dominate the
        # set and would otherwise collapse the schedule's temperature). This is
        # the online analogue of the peak-load seed, in the right quality units.
        probed = [abs(q - quality_before) for q in qualities if q is not None]
        nonzero = [d for d in probed if d > 0.0]
        move_scale = sum(nonzero) / len(nonzero) if nonzero else 0.0

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
                return (i, j), move_scale

        # Nothing accepted: the live plan was never touched.
        return None, move_scale
