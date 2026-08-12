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


"""Cooling schedules for the simulated-annealing layout solver.

A :class:`CoolingSchedule` is a responsive temperature controller: the annealer
streams back each step's acceptance and move scale, so a schedule may adapt
online. This module holds the schedule ABC and its concrete implementations,
plus the peak-load helpers used to seed an initial temperature.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

from typing_extensions import override
from heapq import heappush, heappop

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer


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

    @property
    def drives_single_move(self) -> bool:
        """True if :meth:`reset` / :meth:`update` alone can drive a whole anneal.

        A schedule holding one temperature per move type cannot: it has no single
        starting temperature to return from :meth:`reset`, and ``None`` there
        already means "no steps". Such a schedule reports False so a single-move
        driver (:class:`SimulatedAnnealingLayoutSolver`) rejects it up front
        instead of reading that ``None`` as a zero-step anneal; its own caller
        drives it through ``temperature(move_type)`` instead. Default: True.
        """
        return True

    @abstractmethod
    def reset(self) -> Optional[float]:
        """Reinitialize and return the first temperature (None for no steps --
        only ever a genuinely empty anneal; see :attr:`drives_single_move`)."""

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
        """A schedule that starts at temperature `t_initial` and ends at `t_final`,
        cooling down by a constant factor every `steps_per_epoch` steps. There are
        `epochs` such epochs.

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


# Sentinel move type for the single-band (layout-only) case.
_SINGLE_MOVE = "default"


class SelfCalibratingReheatingSchedule(CoolingSchedule):
    """Self-calibrating simulated-annealing schedule with reheating cycles.

    The default schedule for both the layout-only annealer and the joint
    work-division + LX co-optimizer. It needs no tuning beyond the step budget:
    it sizes its temperatures to the instance online from the move scale streamed
    back, locates the productive temperature, and spends the budget on reheating
    cycles around it -- concentrating moves where they are useful but not frozen.

    Single band or per move type. A single acceptance band ``(accept_hi,
    accept_lo)`` is the default (the layout annealer's one move type). Passing
    ``bands={move_type: (hi, lo)}`` instead gives **one shared reheating carrier
    with an independent band + move-scale EMA per move type** (the co-optimizer's
    reorder / flip / region-recolor, with recolor coldest) -- so a move type's
    large deltas do not freeze the small ones, and a rare move calibrates on its
    own sample count. ``temperature(move_type)`` / ``update(.., move_type)`` and
    ``cycle_phase()`` expose the per-move surface; the single-move responsive API
    (``reset`` / ``update(accepted, scale)`` returning the next temperature, or
    ``None`` to stop) is unchanged and uses the default move type.

    NOTE: a *reasonable* self-calibrating default, not a tuned/provably-good one
    -- no representative models to benchmark against yet. Reheating-beats-a-single-
    cool and online-scale-learning are unvalidated bets; both are bounded by the
    solver's best-seen tracking, so they can waste budget but never worsen the
    result. Expect to revisit ``cycles`` and the bands once we can benchmark.

    Temperature scale (per move type ``m``). With ``A = -ln(hi_m)``,
    ``B = -ln(lo_m)``, a band centered on ``center_m`` accepts a mean-magnitude
    *worsening* move at ``hi_m`` at its top ``center_m * delta_m`` and ``lo_m`` at
    its bottom ``center_m / delta_m``, where ``delta_m = sqrt(B/A)`` and
    ``center_m = d_hat_m / sqrt(A*B)``. ``d_hat_m`` is an EMA of that move type's
    streamed scale, so its band tracks its move scale as the landscape flattens.

    Bootstrap. Before any move scale is known, ``center`` is seeded from the
    peak-load estimate (:func:`default_initial_temperature`, placed at the band
    top) via :meth:`set_buffers`, or from an explicit ``seed_center`` passed in
    the ctor (the co-optimizer, whose scorer is not in byte units). Only the few
    pre-snap steps use it, and best-seen tracking absorbs them.

    Reheating. The budget is split into ``cycles`` equal cycles (the last
    absorbing any remainder) on a single shared carrier; each cools geometrically
    from band top to band bottom, ``center_m`` recomputed from ``d_hat_m`` every
    step so bands drift with the landscape continuously. The EMA horizon is
    ``cycle_len / horizons_per_cycle``. The carrier saturates at the band bottom,
    so the last cycle's remainder steps hold there rather than cooling below the
    ``accept_lo`` the band was built from (see :meth:`_carrier`).

    Budget/band knobs:
        total_steps: annealing budget (temperatures emitted). ``None`` ->
            adaptive, ``clamp(steps_per_buffer * n, min_steps, max_steps)``, sized
            in :meth:`set_buffers`.
        cycles / horizons_per_cycle / max_steps: as before (guessed defaults).
        accept_hi / accept_lo: the single default band (ignored if ``bands`` set).
        bands: ``{move_type: (accept_hi, accept_lo)}`` for the multi-move case.
        seed_center: explicit pre-snap center (co-optimizer). With ``total_steps``
            also given, the schedule is fully sized in the ctor and needs no
            :meth:`set_buffers` call -- and is superseded by the peak-load seed if
            :meth:`set_buffers` is called anyway.
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
        bands: Optional[dict[str, tuple[float, float]]] = None,
        seed_center: Optional[float] = None,
    ):
        self._bands = bands or {_SINGLE_MOVE: (accept_hi, accept_lo)}
        for name, (hi, lo) in self._bands.items():
            if not 0.0 < lo < hi < 1.0:
                raise ValueError(f"{name}: need 0 < accept_lo < accept_hi < 1")
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
        # Per-move band geometry (scale-independent): `delta` a factor above/below
        # the center, `rt_ab` = sqrt(A*B) converting a move scale into the center.
        self._delta: dict[str, float] = {}
        self._rt_ab: dict[str, float] = {}
        for name, (hi, lo) in self._bands.items():
            a = -math.log(hi)
            b = -math.log(lo)
            self._delta[name] = math.sqrt(b / a)
            self._rt_ab[name] = math.sqrt(a * b)
        # ``_cycle_len == 0`` marks "not yet sized" so reset() refuses to run
        # uncalibrated. Sized here iff both the budget and an explicit seed are
        # given (co-optimizer path, no buffers); otherwise in set_buffers.
        self.total_steps = 0
        self._cycle_len = 0
        self._seed_center: dict[str, float] = {}
        if total_steps is not None and seed_center is not None:
            self.total_steps = max(1, total_steps)
            self._size()
            self._seed_center = {n: seed_center for n in self._bands}

    def _size(self) -> None:
        self._cycle_len = max(1, self.total_steps // self.cycles)
        # Cool by delta^2 across one cycle (band top to bottom), per move type.
        self._alpha = {n: d ** (-2.0 / self._cycle_len) for n, d in self._delta.items()}
        # EMA horizon of cycle_len / horizons_per_cycle steps; clamped to a valid
        # rate for short cycles (degrades to "center = latest scale").
        self._ema_beta = min(1.0, self.horizons_per_cycle / self._cycle_len)
        # Average this many nonzero samples before a band snaps off its seed.
        self._snap_after = min(self._cycle_len // 4, 20) or 1

    @override
    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        if self._total_steps is None:
            self.total_steps = min(
                self.max_steps,
                max(self.min_steps, self.steps_per_buffer * len(buffers)),
            )
        else:
            self.total_steps = max(1, self._total_steps)
        self._size()
        # Seed each band so the peak-load estimate lands at its top. Unlike
        # ``total_steps`` above, an explicit ctor ``seed_center`` does NOT survive
        # this: set_buffers is only ever called by the layout annealer, whose
        # quality is in buffer bytes, so the peak-derived seed is the one on the
        # right scale there. (A seed for some other scorer would be worse than
        # useless, and only the pre-snap steps use either.)
        peak = default_initial_temperature(buffers)
        self._seed_center = {n: peak / self._delta[n] for n in self._bands}

    @property
    @override
    def drives_single_move(self) -> bool:
        # One band is the layout annealer's case: reset()/update() carry the whole
        # anneal. Two or more means per-move temperatures, which that API cannot
        # express -- see reset().
        return len(self._bands) == 1

    @override
    def reset(self) -> Optional[float]:
        if self._cycle_len == 0:
            raise ValueError(
                "SelfCalibratingReheatingSchedule must be sized before use: give "
                "total_steps + seed_center, run it through "
                "SimulatedAnnealingLayoutSolver, or call set_buffers() first."
            )
        self._i = 0
        self._s = 0
        self._cycle = 0
        self._center = dict(self._seed_center)
        self._d_hat: dict[str, Optional[float]] = dict.fromkeys(self._bands, None)
        self._sample_sum = dict.fromkeys(self._bands, 0.0)
        self._n_samples = dict.fromkeys(self._bands, 0)
        # A single-move schedule has one unambiguous starting temperature (the
        # layout annealer uses it); a multi-move schedule does not -- its caller
        # queries temperature(move_type) per type, so there is nothing to return.
        # This ``None`` collides with the ABC's "no steps", which is why
        # drives_single_move is False above: a single-move driver refuses the
        # schedule rather than mistaking it for an empty anneal.
        if self.drives_single_move:
            return self.temperature(next(iter(self._bands)))
        return None

    @property
    def finished(self) -> bool:
        """True once the budget is spent (the co-optimizer's stop condition; the
        responsive :meth:`update` also signals it by returning ``None``)."""
        return self._i >= self.total_steps

    def _carrier(self) -> int:
        """Carrier position for the current step, clamped to one cycle length.

        ``cycle_len`` floors, so the last cycle absorbs the remainder
        (``total_steps % cycles`` steps beyond ``cycle_len``) and the raw step
        counter ``_s`` can run past a full cycle there. Clamping holds those steps
        at the cycle's cold end instead, which keeps both readers honest: the band
        bottom stays the coldest temperature the schedule will emit -- the whole
        point of building the band from ``accept_lo`` -- and the phase stays inside
        ``[0, 1]``.
        """
        return min(self._s, self._cycle_len)

    def cycle_phase(self) -> float:
        """Scale-invariant carrier phase ``s / cycle_len`` in ``[0, 1]`` -- 0 at a
        cycle's hot top, 1 at its cold bottom -- shared across move types (drives
        the co-optimizer's cycle-phase proposal mix).

        Closed at 1, not half-open: the last cycle's remainder steps saturate
        there (see :meth:`_carrier`). Every earlier cycle stops at
        ``1 - 1 / cycle_len``.
        """
        return self._carrier() / self._cycle_len

    def temperature(self, move_type: str = _SINGLE_MOVE) -> float:
        """Temperature for ``move_type`` at the current carrier position ``s``:
        ``center_m * delta_m * alpha_m ** s`` -- band top at ``s == 0``, band
        bottom ``center_m / delta_m`` at ``s == cycle_len``, and never below that
        bottom because the carrier saturates there (see :meth:`_carrier`). The
        band itself still moves with ``center_m``."""
        return (
            self._center[move_type]
            * self._delta[move_type]
            * (self._alpha[move_type] ** self._carrier())
        )

    @override
    def update(
        self, accepted: bool, move_scale: float, move_type: str = _SINGLE_MOVE
    ) -> Optional[float]:
        # Track the move scale, ignoring no-op moves (move_scale == 0): they
        # dominate the sample and would collapse the band into a greedy search.
        # Before the first snap, average a few samples; after it, EMA. Then
        # re-center this move type's band from its current scale.
        if move_scale > 0.0:
            dh = self._d_hat[move_type]
            if dh is None:
                self._sample_sum[move_type] += move_scale
                self._n_samples[move_type] += 1
                if self._n_samples[move_type] >= self._snap_after:
                    self._d_hat[move_type] = (
                        self._sample_sum[move_type] / self._n_samples[move_type]
                    )
            else:
                self._d_hat[move_type] = dh + self._ema_beta * (move_scale - dh)
            snapped = self._d_hat[move_type]
            if snapped is not None:
                self._center[move_type] = snapped / self._rt_ab[move_type]

        self._i += 1
        if self._i >= self.total_steps:
            return None
        self._s += 1
        # Cycle boundary: restart the shared carrier at the band top. The last
        # cycle absorbs the budget remainder. (Centers track every step, so the
        # boundary only restarts the carrier phase.)
        if self._s >= self._cycle_len and self._cycle < self.cycles - 1:
            self._cycle += 1
            self._s = 0
        return self.temperature(move_type)
