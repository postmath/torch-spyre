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

"""Unit tests for the multi-move surface of SelfCalibratingReheatingSchedule
(Plan §5.1 / §5.2): one shared reheating carrier, an independent band + move-scale
EMA per move type. (The single-move responsive path is covered in
test_simulated_annealing.py.)"""

import math
import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad.cooling_schedules import (
    SelfCalibratingReheatingSchedule,
)

# Progressively colder bands (reorder warmest, recolor coldest floor).
_BANDS = {
    "reorder": (0.6, 0.02),
    "flip": (0.3, 0.005),
    "recolor": (0.1, 0.001),
}


def _sched(total_steps=4000, cycles=1):
    s = SelfCalibratingReheatingSchedule(
        bands=_BANDS, total_steps=total_steps, cycles=cycles, seed_center=1.0
    )
    s.reset()
    return s


class ConstructionTest(TestCase):
    def test_rejects_bad_bands(self):
        for bad in ({"m": (0.5, 0.6)}, {"m": (1.0, 0.1)}, {"m": (0.5, 0.0)}):
            with self.assertRaises(ValueError):
                SelfCalibratingReheatingSchedule(
                    bands=bad, total_steps=100, seed_center=1.0
                )

    def test_no_bands_is_single_default_move(self):
        # bands=None -> one default-move band from accept_hi/accept_lo, usable via
        # the no-argument temperature()/update() (single-move layout API).
        s = SelfCalibratingReheatingSchedule(total_steps=20, seed_center=1.0)
        s.reset()
        self.assertIsInstance(s.temperature(), float)
        self.assertIsInstance(s.update(True, 5.0), float)

    def test_only_a_multi_move_schedule_declines_the_single_move_api(self):
        # A multi-band reset() returns None for its own reason (no single starting
        # temperature), colliding with the ABC's "no steps". drives_single_move is
        # how a single-move driver tells the two apart before it starts; the
        # rejection itself is covered in test_simulated_annealing.py.
        multi = SelfCalibratingReheatingSchedule(
            bands=_BANDS, total_steps=100, seed_center=1.0
        )
        self.assertFalse(multi.drives_single_move)
        self.assertIsNone(multi.reset())
        # One band -- named or default -- is still the single-move case.
        for bands in (None, {"reorder": (0.6, 0.02)}):
            single = SelfCalibratingReheatingSchedule(
                bands=bands, total_steps=100, seed_center=1.0
            )
            self.assertTrue(single.drives_single_move)
            self.assertIsNotNone(single.reset())


class BandOrderingTest(TestCase):
    def test_colder_band_is_colder_at_the_same_scale(self):
        # Set every band's snapped scale directly (carrier at s == 0). At the band
        # top the temperature is scale / -ln(accept_hi), so a colder band (smaller
        # hi) is colder.
        s = _sched()
        for m in _BANDS:
            s._d_hat[m] = 1000.0
            s._center[m] = 1000.0 / s._rt_ab[m]
        self.assertGreater(s.temperature("reorder"), s.temperature("flip"))
        self.assertGreater(s.temperature("flip"), s.temperature("recolor"))

    def test_band_top_temperature_matches_accept_target(self):
        # exp(-d_hat / T_top) == accept_hi at the band top (s == 0).
        s = SelfCalibratingReheatingSchedule(
            bands={"reorder": (0.6, 0.02)}, total_steps=4000, seed_center=1.0
        )
        s.reset()
        s._d_hat["reorder"] = 500.0
        s._center["reorder"] = 500.0 / s._rt_ab["reorder"]
        self.assertAlmostEqual(
            math.exp(-500.0 / s.temperature("reorder")), 0.6, places=9
        )


class EmaCalibrationTest(TestCase):
    def test_each_type_tracks_its_own_scale(self):
        # Feed >20 (>= any snap_after) equal samples per type; the EMA lands on
        # that scale. An unsampled type keeps its unseeded (None) EMA.
        s = _sched(total_steps=9000)
        for _ in range(25):
            s.update(False, 10.0, "reorder")
            s.update(False, 1000.0, "flip")
        self.assertAlmostEqual(s._d_hat["reorder"], 10.0)
        self.assertAlmostEqual(s._d_hat["flip"], 1000.0)
        self.assertIsNone(s._d_hat["recolor"])

    def test_zero_scale_samples_ignored(self):
        s = _sched(total_steps=9000)
        for _ in range(50):
            s.update(True, 0.0, "reorder")
        self.assertIsNone(s._d_hat["reorder"])  # no-ops never snap the EMA


class CarrierTest(TestCase):
    def test_cycle_phase_advances_and_reheats(self):
        s = SelfCalibratingReheatingSchedule(
            bands={"m": (0.6, 0.02)}, total_steps=100, cycles=4, seed_center=1.0
        )
        s.reset()
        phases, temps = [s.cycle_phase()], [s.temperature("m")]
        for _ in range(60):
            s.update(False, 100.0, "m")  # constant scale -> stable center
            phases.append(s.cycle_phase())
            temps.append(s.temperature("m"))
        self.assertTrue(all(0.0 <= p < 1.0 for p in phases))
        # Phase resets and temperature rises again at least once (a reheat).
        self.assertTrue(any(phases[i + 1] < phases[i] for i in range(len(phases) - 1)))
        self.assertTrue(any(temps[i + 1] > temps[i] for i in range(len(temps) - 1)))

    def test_carrier_saturates_in_the_last_cycle(self):
        # cycle_len floors, so the last cycle runs total_steps % cycles steps
        # beyond a full cycle (here 10 // 4 = 2, remainder 2). The clamped carrier
        # holds those steps at the cycle's cold end: the phase stays in the
        # documented [0, 1] (an overshoot would flip the co-optimizer's hotness
        # weights negative) and the temperature never dips below the band bottom.
        s = SelfCalibratingReheatingSchedule(
            bands={"m": (0.6, 0.02)}, total_steps=10, cycles=4, seed_center=1.0
        )
        s.reset()
        self.assertEqual(s._cycle_len, 2)
        phases, temps = [], []
        while not s.finished:
            phases.append(s.cycle_phase())
            temps.append(s.temperature("m"))
            s.update(True, 0.0, "m")  # no-op moves: center stays at the seed
        self.assertEqual(len(phases), 10)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in phases), phases)
        # Three cycles of (0, 0.5), then the last one overrunning by two steps.
        self.assertEqual(phases, [0.0, 0.5] * 4 + [1.0, 1.0])
        # With center pinned at the seed, the band is a fixed [bottom, top]: the
        # two remainder steps sit exactly on the bottom, not below it.
        top = 1.0 * s._delta["m"]
        bottom = 1.0 / s._delta["m"]
        self.assertAlmostEqual(min(temps), bottom)
        for t in temps:
            self.assertGreaterEqual(t, bottom - 1e-12)
            self.assertLessEqual(t, top + 1e-12)
        self.assertAlmostEqual(temps[-1], bottom)
        self.assertAlmostEqual(temps[-2], bottom)

    def test_temperature_stays_in_band_across_budget_and_cycle_mixes(self):
        # Same invariant over remainder shapes (r = 0, 1, 2, and cycles >
        # total_steps), including a moving center: a live d_hat rescales the band
        # every step, but the temperature stays within delta of that center.
        for total_steps, cycles in ((10, 4), (17, 4), (12, 4), (7, 3), (3, 4)):
            s = SelfCalibratingReheatingSchedule(
                bands={"m": (0.6, 0.02)},
                total_steps=total_steps,
                cycles=cycles,
                seed_center=1.0,
            )
            s.reset()
            while not s.finished:
                center = s._center["m"]
                t = s.temperature("m")
                where = (total_steps, cycles, s._s)
                self.assertLessEqual(t, center * s._delta["m"] + 1e-12, where)
                self.assertGreaterEqual(t, center / s._delta["m"] - 1e-12, where)
                self.assertLessEqual(s.cycle_phase(), 1.0, where)
                s.update(True, 5.0, "m")

    def test_cycle_phase_stays_below_one_when_cycles_divide_the_budget(self):
        # No remainder -> no overrun, so the phase never reaches 1 and the range
        # is the half-open one the carrier gives naturally.
        s = SelfCalibratingReheatingSchedule(
            bands={"m": (0.6, 0.02)}, total_steps=100, cycles=4, seed_center=1.0
        )
        s.reset()
        phases = []
        while not s.finished:
            phases.append(s.cycle_phase())
            s.update(True, 1.0, "m")
        self.assertAlmostEqual(max(phases), 24 / 25)  # (cycle_len - 1) / cycle_len

    def test_update_returns_none_exactly_at_budget(self):
        s = SelfCalibratingReheatingSchedule(
            bands={"m": (0.6, 0.02)}, total_steps=50, seed_center=1.0
        )
        t = s.reset()
        count = 0
        while t is not None:
            count += 1
            t = s.update(True, 1.0, "m")
        self.assertEqual(count, 50)
        self.assertTrue(s.finished)

    def test_reset_restores_initial_state(self):
        s = _sched(total_steps=200)
        for _ in range(30):
            s.update(True, 5.0, "flip")
        s.reset()
        self.assertFalse(s.finished)
        self.assertEqual(s.cycle_phase(), 0.0)
        self.assertIsNone(s._d_hat["reorder"])
        self.assertIsNone(s._d_hat["flip"])

    def test_reset_without_sizing_errors(self):
        # No total_steps and no set_buffers -> not sized -> reset must refuse.
        with self.assertRaises(ValueError):
            SelfCalibratingReheatingSchedule(bands=_BANDS, seed_center=1.0).reset()


if __name__ == "__main__":
    unittest.main()
