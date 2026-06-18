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

"""End-to-end tests for the Imanishi/Xu simulated-annealing layout solver."""

import copy
import math
import os
import random as rnd
import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
)
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    buffer_quality,
)
from torch_spyre._inductor.scratchpad.imanishi_xu import (
    SelfCalibratingCoolingSchedule,
    ExponentialCoolingSchedule,
    ImanishiXuLayoutSolver,
    ImanishiXuSolverWithBuffers,
    ReheatingSchedule,
    default_initial_temperature,
    peak_memory_load,
)

# Heavy randomized anneals over many seeds, larger problems and longer
# schedules. Skipped by default (slow); opt in with the env var.
_STRESS = os.environ.get("TORCH_SPYRE_STRESS_SCRATCHPAD") == "1"


def _random_buffers(rng, n, horizon=12, max_size=200):
    """Half-open lifetimes, some in-place children (parent.end == child.start+1)."""
    buffers = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        uses = [start] if end == start + 1 else [start, end - 1]
        buffers.append(LifetimeBoundBuffer(f"b{i}", size, uses))
    for child_i in range(1, n):
        if rng.random() < 0.25:
            parent = buffers[rng.randrange(child_i)]
            child = buffers[child_i]
            new_start = parent.uses[-1]
            new_last = max(child.uses[-1], parent.uses[-1])
            child.uses = [new_start] if new_start == new_last else [new_start, new_last]
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def _short_schedule():
    return ExponentialCoolingSchedule(
        t_initial=100.0, t_final=1.0, steps_per_epoch=5, epochs=4
    )


def _assert_feasible(buffers, capacity):
    """Committed buffers fit below capacity and never address-overlap a
    time-overlapping peer (an in-place pair may share its base address)."""
    committed = [b for b in buffers if b.address is not None]
    for b in committed:
        assert b.address + b.size <= capacity, f"{b.name} exceeds capacity"
    for a in range(len(committed)):
        for c in range(a + 1, len(committed)):
            u, v = committed[a], committed[c]
            if not u.overlaps_in_time(v):
                continue
            if u.name in v.in_place_parents or v.name in u.in_place_parents:
                continue  # in-place pair may share an address
            assert u.address + u.size <= v.address or v.address + v.size <= u.address, (
                f"{u.name}@{u.address}+{u.size} overlaps {v.name}@{v.address}+{v.size}"
            )


def _committed_total(buffers):
    # The annealer optimizes the use-weighted quality, so the committed total it
    # is compared against must use the same weighting (not raw size).
    return sum(buffer_quality(b) for b in buffers if b.address is not None)


class CoolingScheduleTests(TestCase):
    def test_exponential_schedule_sequence(self):
        # alpha = (1/8) ** (1/3) = 0.5; cools once per epoch (at i=2, 4).
        s = ExponentialCoolingSchedule(
            t_initial=8.0, t_final=1.0, steps_per_epoch=2, epochs=4
        )
        traj = [s.reset()]
        t = traj[0]
        while t is not None:
            t = s.update(True)  # ignores acceptance
            traj.append(t)
        self.assertEqual(traj, [8.0, 8.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, None])

    def test_reheating_schedule_trajectory(self):
        # Cool (halving) until the windowed acceptance rate drops below 0.5,
        # locating T1, then cycle the band [T1/2, T1*2] twice.
        s = ReheatingSchedule(
            t0=100.0, alpha=0.5, window=4, stall_rate=0.5, delta=2.0, restarts=2
        )
        scripted = [True, True, True, True, False, False, False]
        traj = [s.reset()]
        t = traj[0]
        i = 0
        while t is not None:
            accepted = scripted[i] if i < len(scripted) else False
            t = s.update(accepted)
            traj.append(t)
            i += 1
        # Stall at T1=1.5625 (rate 0.25 < 0.5); reheat band top/bottom = 3.125 /
        # 0.78125, two cycles.
        self.assertEqual(
            traj,
            [
                100.0,
                50.0,
                25.0,
                12.5,
                6.25,
                3.125,
                1.5625,
                3.125,
                1.5625,
                3.125,
                1.5625,
                None,
            ],
        )

    def test_reheating_no_restarts_stops_at_stall(self):
        s = ReheatingSchedule(
            t0=8.0, alpha=0.5, window=2, stall_rate=0.5, delta=2.0, restarts=0
        )
        s.reset()
        self.assertIsNotNone(s.update(False))  # window not full yet
        self.assertIsNone(s.update(False))  # window full, rate 0 < 0.5 -> stop

    def test_auto_schedule_adaptive_budget(self):
        def n_buffers(n):
            return [LifetimeBoundBuffer(f"b{i}", 1, [0]) for i in range(n)]

        # n=100 (the random-buffer example size): 30*n = 3000, under the 5000 cap.
        s = SelfCalibratingCoolingSchedule()
        s.set_buffers(n_buffers(100))
        self.assertEqual(s.total_steps, 3000)
        self.assertLessEqual(s.total_steps, 5000)  # the explicit budget ceiling
        self.assertEqual(s.warmup_steps, 300)  # round(0.1 * total)
        # Large n is capped; tiny n hits the floor.
        capped = SelfCalibratingCoolingSchedule()
        capped.set_buffers(n_buffers(1000))
        self.assertEqual(capped.total_steps, 5000)
        floored = SelfCalibratingCoolingSchedule()
        floored.set_buffers(n_buffers(5))
        self.assertEqual(floored.total_steps, 500)
        # Explicit overrides win.
        ex = SelfCalibratingCoolingSchedule(total_steps=42, warmup_steps=7)
        ex.set_buffers(n_buffers(100))
        self.assertEqual((ex.total_steps, ex.warmup_steps), (42, 7))

    def test_auto_schedule_calibration_endpoints(self):
        s = SelfCalibratingCoolingSchedule(total_steps=4, accept_hi=0.8, accept_lo=0.01)
        # mean over nonzero deltas = 20; t0/t_end target accept_hi/accept_lo.
        s.calibrate([10.0, 20.0, 30.0, 0.0])
        t0 = 20.0 / -math.log(0.8)
        temps = []
        t = s.reset()
        while t is not None:
            temps.append(t)
            t = s.update(True)
        self.assertEqual(len(temps), 4)  # total_steps temperatures, then stop
        self.assertAlmostEqual(temps[0], t0)
        self.assertTrue(all(temps[k] > temps[k + 1] for k in range(len(temps) - 1)))

    def test_auto_schedule_degenerate_deltas(self):
        # No sampled move changed quality -> a safe (positive) flat temperature.
        s = SelfCalibratingCoolingSchedule(total_steps=3)
        s.calibrate([0.0, 0.0])
        self.assertIsNotNone(s.reset())

    def test_auto_schedule_uncalibrated_reset_errors(self):
        s = SelfCalibratingCoolingSchedule(total_steps=10)
        s.set_buffers([LifetimeBoundBuffer("a", 1, [0])])
        with self.assertRaises(ValueError):
            s.reset()  # never calibrated

    def test_default_schedule_is_auto_feasible_and_deterministic(self):
        # The solver's default schedule is the auto-calibrated exponential; with
        # the default (seeded) RNG, two runs of the same instance must agree.
        for seed in range(15):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            b1, b2 = copy.deepcopy(buffers), copy.deepcopy(buffers)
            ImanishiXuLayoutSolver(cap, 128).plan_layout(b1)  # default schedule
            ImanishiXuLayoutSolver(cap, 128).plan_layout(b2)
            self.assertEqual(
                [b.address for b in b1], [b.address for b in b2], f"seed={seed}"
            )
            _assert_feasible(b1, cap)

    def test_peak_memory_load(self):
        # a:[0,2) b:[1,3) c:[2,4); peak live set is {b,c} at tick 2 = 50.
        buffers = [
            LifetimeBoundBuffer("a", 10, [0, 1]),
            LifetimeBoundBuffer("b", 20, [1, 2]),
            LifetimeBoundBuffer("c", 30, [2, 3]),
        ]
        self.assertEqual(peak_memory_load(buffers), 50)
        self.assertAlmostEqual(default_initial_temperature(buffers), 50 / 300.0)

    def test_reheating_t0_derived_from_buffers(self):
        buffers = [
            LifetimeBoundBuffer("a", 10, [0, 1]),
            LifetimeBoundBuffer("b", 20, [1, 2]),
            LifetimeBoundBuffer("c", 30, [2, 3]),
        ]
        s = ReheatingSchedule(
            alpha=0.5, window=2, stall_rate=0.5, delta=2.0, restarts=1
        )
        s.set_buffers(buffers)
        self.assertAlmostEqual(s.t0, 50 / 300.0)
        self.assertAlmostEqual(s.reset(), 50 / 300.0)

    def test_reheating_explicit_t0_not_overridden(self):
        s = ReheatingSchedule(
            t0=99.0, alpha=0.5, window=2, stall_rate=0.5, delta=2.0, restarts=1
        )
        s.set_buffers([LifetimeBoundBuffer("a", 10, [0, 1])])
        self.assertEqual(s.t0, 99.0)

    def test_reheating_no_t0_errors(self):
        s = ReheatingSchedule(
            alpha=0.5, window=2, stall_rate=0.5, delta=2.0, restarts=1
        )
        with self.assertRaises(ValueError):
            s.reset()


class ImanishiXuTests(TestCase):
    def _run(self, buffers, capacity, *, initial, seed, alignment=128):
        solver = ImanishiXuLayoutSolver(
            capacity,
            alignment,
            initial=initial,
            schedule=_short_schedule(),
            random=rnd.Random(seed),
        )
        return solver.plan_layout(buffers)

    def test_solve_skips_annealing_when_initial_already_complete(self):
        # The capacity fits all three buffers in any order, so the initial
        # first_fit layout is already globally optimal. solve()'s up-front check
        # must return before calibrating or running a single anneal.
        buffers = [
            LifetimeBoundBuffer("a", 64, [0, 1]),
            LifetimeBoundBuffer("b", 64, [0, 1]),
            LifetimeBoundBuffer("c", 64, [0, 1]),
        ]
        cap = 10_000
        solver = ImanishiXuSolverWithBuffers(
            buffers,
            cap,
            128,
            initial="first_fit",
            schedule=_short_schedule(),
            random=rnd.Random(0),
        )
        self.assertTrue(solver._is_optimal())  # precondition: initial is complete
        solver.solve()
        # anneal() appends exactly one log per call, so an empty list proves no
        # anneal ran.
        self.assertEqual(solver.quality_logs, [])
        solver.finalize()
        self.assertTrue(all(b.address is not None for b in buffers))
        _assert_feasible(buffers, cap)

    def test_anneal_stops_once_all_buffers_allocated(self):
        # Order [a, b, c] leaves c stacked above capacity (only 2 of 3 fit), but
        # placing c before b lets it drop to address 0 so all three fit. From
        # this order every buffer's best reinsertion reaches that complete
        # layout, so the first annealing step lands it regardless of the RNG --
        # and the cooling loop must then break immediately rather than run the
        # schedule out.
        buffers = [
            LifetimeBoundBuffer("a", 64, [0, 1]),
            LifetimeBoundBuffer("b", 64, [1, 4]),
            LifetimeBoundBuffer("c", 64, [3, 4]),
        ]
        cap = 128
        schedule = ExponentialCoolingSchedule(
            t_initial=8.0, t_final=1.0, steps_per_epoch=2, epochs=4
        )  # 8 cooling steps if never interrupted
        solver = ImanishiXuSolverWithBuffers(
            buffers,
            cap,
            1,
            initial=[0, 1, 2],
            schedule=schedule,
            random=rnd.Random(0),
        )
        self.assertEqual(solver.plan.count_allocated(), 2)  # c does not yet fit
        solver.anneal()
        self.assertEqual(solver.plan.count_allocated(), 3)  # reached completeness
        # Broke after the first iteration instead of running all 8 steps.
        self.assertEqual(len(solver.quality_logs[0]), 1)

    def test_finalized_layout_is_feasible(self):
        for seed in range(60):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            self._run(buffers, cap, initial="first_fit", seed=seed)
            _assert_feasible(buffers, cap)

    def test_annealing_never_worse_than_initial(self):
        # Starting from a known permutation, the tracked best (and thus the
        # finalized committed total) can only improve on the initial layout.
        for seed in range(60):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            initial = list(range(n))
            rng.shuffle(initial)
            initial_quality = PermutationBasedLayoutSolver(
                copy.deepcopy(buffers), list(initial), cap, 128
            ).quality()

            self._run(buffers, cap, initial=initial, seed=seed)
            self.assertGreaterEqual(_committed_total(buffers), initial_quality, seed)
            _assert_feasible(buffers, cap)

    def test_deterministic_with_seed(self):
        rng = rnd.Random(0)
        n = 7
        base = _random_buffers(rng, n)
        cap = max(b.size for b in base) * 3

        first = copy.deepcopy(base)
        self._run(first, cap, initial="first_fit", seed=42)
        second = copy.deepcopy(base)
        self._run(second, cap, initial="first_fit", seed=42)

        self.assertEqual([b.address for b in first], [b.address for b in second])

    def test_reheating_schedule_end_to_end(self):
        for seed in range(40):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            initial = list(range(n))
            rng.shuffle(initial)
            initial_quality = PermutationBasedLayoutSolver(
                copy.deepcopy(buffers), list(initial), cap, 128
            ).quality()

            # t0 omitted: the solver derives it from the peak load.
            schedule = ReheatingSchedule(
                alpha=0.85,
                window=15,
                stall_rate=0.2,
                delta=4.0,
                restarts=3,
            )
            solver = ImanishiXuLayoutSolver(
                cap, 128, initial=initial, schedule=schedule, random=rnd.Random(seed)
            )
            solver.plan_layout(buffers)

            _assert_feasible(buffers, cap)
            self.assertGreaterEqual(_committed_total(buffers), initial_quality, seed)


@unittest.skipUnless(
    _STRESS, "set TORCH_SPYRE_STRESS_SCRATCHPAD=1 to run scratchpad stress tests"
)
class ImanishiXuStressTests(TestCase):
    """Heavy version of ImanishiXuTests: many seeds, larger problems and a
    longer cooling schedule. Not run by default."""

    def test_many_anneals_feasible_and_not_worse(self):
        for seed in range(500):
            rng = rnd.Random(seed)
            n = rng.randint(2, 14)
            buffers = _random_buffers(rng, n, horizon=20, max_size=300)
            cap = max(b.size for b in buffers) * rng.randint(2, 5)
            initial = list(range(n))
            rng.shuffle(initial)
            initial_quality = PermutationBasedLayoutSolver(
                copy.deepcopy(buffers), list(initial), cap, 128
            ).quality()

            schedule = ExponentialCoolingSchedule(
                t_initial=200.0, t_final=0.5, steps_per_epoch=8, epochs=6
            )
            solver = ImanishiXuLayoutSolver(
                cap,
                128,
                initial=initial,
                schedule=schedule,
                random=rnd.Random(seed * 7 + 1),
            )
            solver.plan_layout(buffers)

            _assert_feasible(buffers, cap)
            self.assertGreaterEqual(_committed_total(buffers), initial_quality, seed)
