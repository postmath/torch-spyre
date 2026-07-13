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

"""Phase-3 validation for the minimal end-to-end SA co-optimization engine.

The two Plan §8.3 gates:

* **determinism** -- two runs on identical input give bit-for-bit identical
  ``chosen_division`` + ``address`` (Plan §7.5); and
* **>= baseline on the shared scorer** -- the returned state never scores worse
  than the seed (index-0 divisions + FirstFit ``pi``), the seed-from-baseline +
  keep-best guarantee (Plan §8.1).

Plus the engine's output contract: every buffer gets a ``chosen_division`` and an
``address`` (``None`` == spilled), with ``spill_reasons`` populated for the
misses. Runs over the real-shaped captured graphs at several capacities, so
residency pressure (spills / eligibility toggles) is actually exercised.
"""

import copy
import math
import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver

from tests.inductor.fake_cooptimization_substrate import load_captures


def _seed_footprint(buffers):
    """Total per-core footprint of the seed (index-0) divisions -- the scale used
    to pick exercise capacities."""
    return sum(
        math.ceil(b.size / b.core_divisions[0].output_partition) for b in buffers
    )


def _capacities(buffers):
    """A spread of scratchpad capacities: unbounded, roomy, and two tight ones
    that force spills / eligibility pressure."""
    tot = _seed_footprint(buffers)
    return [1 << 30, tot, max(1, tot // 2), max(1, tot // 4)]


def _all_cases():
    for case, graphs in load_captures().items():
        for gi, g in enumerate(graphs):
            yield case, gi, g.buffers


class OutputContractTest(TestCase):
    def test_every_buffer_gets_division_and_address(self):
        for case, gi, buffers in _all_cases():
            for cap in _capacities(buffers):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(cap, 128, seed=0)
                out = solver.plan_layout_and_core_divs(bufs)
                tag = f"{case}[{gi}] cap={cap}"
                self.assertEqual(len(out), len(bufs), tag)
                for b in out:
                    self.assertIsNotNone(b.chosen_division, f"{tag} {b.name}")
                    self.assertTrue(0 <= b.chosen_division < len(b.core_divisions), tag)
                    # A spilled buffer (no address) must carry a spill reason;
                    # a resident one must not.
                    if b.address is None:
                        self.assertIn(b.name, solver.spill_reasons, f"{tag} {b.name}")
                    else:
                        self.assertNotIn(b.name, solver.spill_reasons, tag)

    def test_empty_graph(self):
        solver = SaCoOptimizingSolver(1024, 128, seed=0)
        self.assertEqual(solver.plan_layout_and_core_divs([]), [])


class BaselineGuaranteeTest(TestCase):
    """The returned state never scores worse than the seed (lower is better)."""

    def test_never_worse_than_baseline(self):
        for case, gi, buffers in _all_cases():
            for cap in _capacities(buffers):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(cap, 128, seed=0)
                solver.plan_layout_and_core_divs(bufs)
                tag = f"{case}[{gi}] cap={cap}"
                self.assertLessEqual(solver.best_score, solver.baseline_score, tag)


class DeterminismTest(TestCase):
    """Two runs on identical input are bit-for-bit identical (Plan §6.5/§7.5)."""

    def _run(self, buffers, cap, seed):
        bufs = copy.deepcopy(buffers)
        solver = SaCoOptimizingSolver(cap, 128, seed=seed)
        out = solver.plan_layout_and_core_divs(bufs)
        return (
            [b.chosen_division for b in out],
            [b.address for b in out],
            solver.best_score,
            dict(solver.spill_reasons),
        )

    def test_same_seed_bit_identical(self):
        for case, gi, buffers in _all_cases():
            for cap in _capacities(buffers):
                a = self._run(buffers, cap, seed=0)
                b = self._run(buffers, cap, seed=0)
                self.assertEqual(a, b, f"{case}[{gi}] cap={cap}")

    def test_different_seeds_still_valid(self):
        # A different seed may explore differently, but each run must still be a
        # valid, >=-baseline solution (guards against seed-dependent breakage).
        for case, gi, buffers in _all_cases():
            cap = _seed_footprint(buffers)
            for seed in (1, 7):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(cap, 128, seed=seed)
                solver.plan_layout_and_core_divs(bufs)
                self.assertLessEqual(
                    solver.best_score, solver.baseline_score, f"{case} seed={seed}"
                )


class ImprovementSmokeTest(TestCase):
    """At a tight capacity the search should usually *improve* on the seed for at
    least one captured graph -- evidence the moves actually do something, beyond
    the (trivially satisfied) >=-baseline guarantee. Not asserted per-graph (a
    graph whose seed is already optimal legitimately ties)."""

    def test_some_graph_improves_under_pressure(self):
        improved = False
        for case, gi, buffers in _all_cases():
            tot = _seed_footprint(buffers)
            for cap in (max(1, tot // 2), max(1, tot // 4)):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(cap, 128, seed=0)
                solver.plan_layout_and_core_divs(bufs)
                if solver.best_score < solver.baseline_score:
                    improved = True
        self.assertTrue(improved, "SA never improved on the seed on any graph")


if __name__ == "__main__":
    unittest.main()
