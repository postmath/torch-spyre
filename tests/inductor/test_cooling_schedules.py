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
