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
import json
import math
import os
import subprocess
import sys
import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver

from tests.inductor.fake_cooptimization_substrate import (
    FakeCoreDivision,
    FakeCoreDivisionBuffer,
    load_captures,
)


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


def _div(partition):
    """A core division with the given output partition (1 == trivial/whole)."""
    return FakeCoreDivision(
        output_splits=({1: partition} if partition > 1 else {}),
        reduction_splits={},
    )


def _cdbuf(name, parents, matches):
    """A minimal buffer with a 3-entry menu (index 0 trivial, 1 split-2, 2
    split-4) and the given parent-compatibility pairs."""
    return FakeCoreDivisionBuffer(
        name=name,
        size=1024,
        uses=[0, 1],
        first_use_is_read=False,
        in_place_parents=[],
        placement=True,
        residency_reason=None,
        boundary_cost=0,
        spill_write_cost=1024,
        parents=parents,
        core_divisions=[_div(1), _div(2), _div(4)],
        cd_parent_matches=matches,
    )


def _flood(buffers, anchor_name, tiling):
    """Run ``_flood_region`` on a hand-built graph; return name -> chosen index."""
    solver = SaCoOptimizingSolver(1 << 30, 128, seed=0)
    solver._bufs = buffers
    solver._precompute_topology()
    result = solver._flood_region(solver._name_to_idx[anchor_name], tiling)
    return {buffers[i].name: d for i, d in result.items()}


class FloodRegionTest(TestCase):
    """The cd_parent_matches flood (Plan §7.2), on controlled synthetic graphs."""

    def test_chain_propagates_full_region(self):
        # A -> B -> C, every edge compatible at index 1: the split propagates end
        # to end.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 1)]}),
            _cdbuf("C", ["B"], {"B": [(1, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1), {"A": 1, "B": 1, "C": 1})

    def test_deterministic_tie_break_picks_smallest(self):
        # A's index 1 is compatible with both B-1 and B-2; the flood takes the
        # smallest, independent of pair list order.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 2), (1, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1)["B"], 1)

    def test_boundary_stops_flood(self):
        # The A->B edge carries no compatible pair for A's tiling 1 (only for 2),
        # so B is outside the region -- a boundary emerges for free.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(2, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1), {"A": 1})

    def test_upward_flood_reaches_parents(self):
        # Anchor the child; the flood must also go up the inverse relation.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "B", 1), {"A": 1, "B": 1})

    def test_join_accepts_internal_seam(self):
        # Diamond A->{B,C}->D with B,C forced to different indices. D is reachable
        # from both but assigned once (first-wins by frontier index: from B);
        # the C->D edge becomes an accepted internal seam, and the flood never
        # fails.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 1)]}),
            _cdbuf("C", ["A"], {"A": [(1, 2)]}),
            _cdbuf("D", ["B", "C"], {"B": [(1, 1)], "C": [(2, 2)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1), {"A": 1, "B": 1, "C": 2, "D": 1})


class RegionRecolorTest(TestCase):
    """Region-recolor is exercised on the captures, finds real regions, helps,
    and is deterministic (Plan §4.3 / §8.3 Tier-0 instrumentation)."""

    def test_recolor_exercised_and_finds_multi_op_regions(self):
        max_region_overall = 0
        for case, gi, buffers in _all_cases():
            tot = _seed_footprint(buffers)
            solver = SaCoOptimizingSolver(max(1, tot // 2), 128, seed=0)
            solver.plan_layout_and_core_divs(copy.deepcopy(buffers))
            if solver._anchor_candidates:  # graph has at least one splittable op
                self.assertGreater(solver.moves_proposed["recolor"], 0, case)
                self.assertTrue(solver.recolor_region_sizes, case)
                max_region_overall = max(
                    max_region_overall, max(solver.recolor_region_sizes)
                )
        # Across the captures, the flood coordinates genuine multi-op regions
        # (not just singletons) -- evidence the bidirectional flood spans edges.
        self.assertGreater(max_region_overall, 1)

    def test_recolor_improves_best_somewhere(self):
        total_improved = 0
        for case, gi, buffers in _all_cases():
            tot = _seed_footprint(buffers)
            for cap in (max(1, tot // 2), max(1, tot // 4)):
                solver = SaCoOptimizingSolver(cap, 128, seed=0)
                solver.plan_layout_and_core_divs(copy.deepcopy(buffers))
                total_improved += solver.recolor_improved
        self.assertGreater(total_improved, 0, "recolor never improved on any graph")

    def test_recolor_instrumentation_deterministic(self):
        for case, gi, buffers in _all_cases():
            tot = _seed_footprint(buffers)

            def run():
                s = SaCoOptimizingSolver(max(1, tot // 2), 128, seed=0)
                s.plan_layout_and_core_divs(copy.deepcopy(buffers))
                return (
                    s.moves_proposed,
                    s.moves_accepted,
                    s.recolor_region_sizes,
                    s.recolor_improved,
                    s.recolor_anchor_partitions,
                    s.recolor_accepted_partitions,
                )

            self.assertEqual(run(), run(), f"{case}[{gi}]")

    def test_anchor_partition_instrumentation_is_consistent(self):
        # The proposed/accepted anchor-partition traces (the input to the future
        # "weight the tiling by output_partition?" decision): every proposed
        # anchor is a genuine split (>1, i.e. the _nontrivial_menu filter holds),
        # there is one entry per proposed recolor, and the accepted trace is a
        # sub-multiset of the proposed one, one per accepted recolor.
        from collections import Counter

        for case, gi, buffers in _all_cases():
            tot = _seed_footprint(buffers)
            solver = SaCoOptimizingSolver(max(1, tot // 2), 128, seed=0)
            solver.plan_layout_and_core_divs(copy.deepcopy(buffers))
            proposed = solver.recolor_anchor_partitions
            accepted = solver.recolor_accepted_partitions
            tag = f"{case}[{gi}]"
            self.assertEqual(len(proposed), solver.moves_proposed["recolor"], tag)
            self.assertEqual(len(accepted), solver.moves_accepted["recolor"], tag)
            self.assertTrue(all(p > 1 for p in proposed), tag)
            self.assertLessEqual(Counter(accepted), Counter(proposed), tag)


class ScheduleTest(TestCase):
    """Plan §5 / §8.3: the reheating schedule + cycle-phase mix beats the crude
    baseline, records per-move acceptance traces + within-group CVs, and stays
    deterministic and >= baseline."""

    def test_reheating_beats_crude_overall(self):
        # Aggregate over the captures at a tight capacity: the reheating schedule
        # must never do worse than the crude one (same seed, same budget), and do
        # strictly better on at least one graph.
        total_reheat = total_crude = 0
        strictly_better = False
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            r = SaCoOptimizingSolver(cap, 128, seed=0, schedule="reheating")
            r.plan_layout_and_core_divs(copy.deepcopy(buffers))
            c = SaCoOptimizingSolver(cap, 128, seed=0, schedule="crude")
            c.plan_layout_and_core_divs(copy.deepcopy(buffers))
            self.assertLessEqual(r.best_score, c.best_score, f"{case}[{gi}]")
            total_reheat += r.best_score
            total_crude += c.best_score
            strictly_better = strictly_better or r.best_score < c.best_score
        self.assertLess(total_reheat, total_crude)
        self.assertTrue(strictly_better, "reheating never beat crude on any graph")

    def test_per_move_acceptance_traces_recorded(self):
        # Every applicable move type is proposed and its accepted count is a valid
        # subset -- the §8.3 per-move-type acceptance traces.
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = SaCoOptimizingSolver(cap, 128, seed=0)
            s.plan_layout_and_core_divs(copy.deepcopy(buffers))
            for m in ("reorder", "flip", "recolor"):
                self.assertGreater(s.moves_proposed[m], 0, f"{case} {m}")
                self.assertLessEqual(
                    s.moves_accepted[m], s.moves_proposed[m], f"{case} {m}"
                )

    def test_within_group_cv_available_and_finite(self):
        # The §5.3 within-group-CV instrumentation is populated and non-negative
        # (drives the deferred variance-bucketing decision).
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = SaCoOptimizingSolver(cap, 128, seed=0)
            s.plan_layout_and_core_divs(copy.deepcopy(buffers))
            cv = s.move_scale_cv()
            self.assertEqual(set(cv), {"reorder", "flip", "recolor", "none"})
            for m, v in cv.items():
                self.assertGreaterEqual(v, 0.0, f"{case} {m}")
                self.assertTrue(math.isfinite(v), f"{case} {m}")

    def test_reheating_deterministic(self):
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)

            def run():
                s = SaCoOptimizingSolver(cap, 128, seed=0, schedule="reheating")
                out = s.plan_layout_and_core_divs(copy.deepcopy(buffers))
                return (
                    [b.chosen_division for b in out],
                    [b.address for b in out],
                    s.best_score,
                    s.moves_proposed,
                    s.moves_accepted,
                )

            self.assertEqual(run(), run(), f"{case}[{gi}]")

    def test_both_schedules_respect_baseline(self):
        for case, gi, buffers in _all_cases():
            for sched in ("reheating", "crude"):
                cap = max(1, _seed_footprint(buffers) // 2)
                s = SaCoOptimizingSolver(cap, 128, seed=0, schedule=sched)
                s.plan_layout_and_core_divs(copy.deepcopy(buffers))
                self.assertLessEqual(s.best_score, s.baseline_score, f"{case} {sched}")

    def test_rejects_bad_schedule(self):
        with self.assertRaises(ValueError):
            SaCoOptimizingSolver(1024, 128, schedule="nope")


# Snippet run in a subprocess to solve one captured graph and print its result;
# used by the cross-process determinism test below.
_SOLVE_SNIPPET = """
import copy, json, math
from tests.inductor.fake_cooptimization_substrate import load_captures
from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver
g = load_captures()["sdpa"][0]
cap = max(1, sum(math.ceil(b.size / b.core_divisions[0].output_partition)
                 for b in g.buffers) // 2)
s = SaCoOptimizingSolver(cap, 128, seed=0)
out = s.plan_layout_and_core_divs(copy.deepcopy(g.buffers))
print("RESULT " + json.dumps({
    "chosen": [b.chosen_division for b in out],
    "addr": [b.address for b in out],
    "best": s.best_score,
}))
"""

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _solve_with_hashseed(hs):
    """Solve the sdpa graph in a subprocess with ``PYTHONHASHSEED=hs``."""
    env = dict(os.environ, PYTHONHASHSEED=str(hs), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    proc = subprocess.run(
        [sys.executable, "-c", _SOLVE_SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


class CrossProcessDeterminismTest(TestCase):
    """The §6.5 CI determinism test done right: solve twice in *separate
    processes* under different ``PYTHONHASHSEED`` values. In-process determinism
    tests share one hash seed and so cannot catch set-iteration-order bugs (Plan
    §7.5) -- this one can (it caught the FirstFit seed nondeterminism)."""

    def test_pythonhashseed_independent(self):
        base = _solve_with_hashseed(0)
        for hs in (1, 2):
            self.assertEqual(_solve_with_hashseed(hs), base, f"PYTHONHASHSEED={hs}")


if __name__ == "__main__":
    unittest.main()
