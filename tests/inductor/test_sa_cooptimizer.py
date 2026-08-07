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
import random as rnd
import subprocess
import sys
import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad.sa_cooptimizer import (
    _DEFAULT_MOVE_BANDS,
    SaCoOptimizingSolver,
)
from torch_spyre._inductor.scratchpad.permutation_layout import (
    make_permutation_packer,
)

from tests.inductor.cooptimization_capture_loader import load_captures
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    CoreDivisionBuffer,
)
from tests.inductor.synthetic_cooptimization_graphs import synthetic_graphs


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
    """The captured real corpus (softmax/mlp/swiglu/sdpa). Empirical, schedule-
    quality assertions that are calibrated on real cost structure iterate this."""
    for case, graphs in load_captures().items():
        for gi, g in enumerate(graphs):
            yield case, gi, g.buffers


def _synthetic_cases():
    """Hand-built structural fixtures (long/short chains, wide join, multi-region,
    K-split, pins, big-n). They carry no ground-truth ``solved`` reference, so they
    exercise only the *shape-invariant* guarantees below -- never schedule-quality
    claims. See ``synthetic_cooptimization_graphs``."""
    for case, graphs in synthetic_graphs().items():
        for gi, g in enumerate(graphs):
            yield case, gi, g.buffers


# Large (25-100 buffer) real captures for manual experiments, kept OUT of CI:
# they are slow (~2s/solve at n~79) and exist to surface schedule-scaling findings
# (e.g. reheating regresses vs crude on the largest flash graphs -- see the
# Phase-5 finding note). Opt in with ``SA_COOPT_LARGE_CAPTURES=1``.
_LARGE_CAPTURES_ENV = "SA_COOPT_LARGE_CAPTURES"
_LARGE_CAPTURES_PATH = os.path.join(
    os.path.dirname(__file__), "cooptimization_captures_large.json"
)


def _large_captures_enabled() -> bool:
    return os.environ.get(_LARGE_CAPTURES_ENV) == "1"


def _large_cases():
    """The env-gated large experimental graphs (empty unless opted in)."""
    if not _large_captures_enabled():
        return
    for case, graphs in load_captures(_LARGE_CAPTURES_PATH).items():
        for gi, g in enumerate(graphs):
            yield case, gi, g.buffers


def _all_cases_incl_synthetic():
    """Real captures + synthetic fixtures (+ large experimental captures when
    ``SA_COOPT_LARGE_CAPTURES=1``): the fan-out for shape-invariant tests (output
    contract, >= baseline, determinism, region flood, recolor coverage) that must
    hold for *any* valid graph, not just the calibrated corpus."""
    yield from _all_cases()
    yield from _synthetic_cases()
    yield from _large_cases()


class OutputContractTest(TestCase):
    def test_every_buffer_gets_division_and_address(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(cap, 128, seed=0)
                out = solver.plan_layout_and_core_divisions(bufs)
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
        self.assertEqual(solver.plan_layout_and_core_divisions([]), [])


class BaselineGuaranteeTest(TestCase):
    """The returned state never scores worse than the seed (lower is better)."""

    def test_never_worse_than_baseline(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(cap, 128, seed=0)
                solver.plan_layout_and_core_divisions(bufs)
                tag = f"{case}[{gi}] cap={cap}"
                self.assertLessEqual(solver.best_score, solver.baseline_score, tag)


class SeedPermutationTest(TestCase):
    """Plan §7.4: ``pi`` is *ordered* over the buffers that can ever be resident.

    A fixed pin can never be resident for any ``(pi, W)``, so it must not occupy a
    prefix slot and displace an eligible buffer. It keeps its index -- ``pi`` stays
    a permutation of all ``n`` -- but sorts after everything the seed placed.
    """

    @staticmethod
    def _seed(buffers, cap):
        solver = SaCoOptimizingSolver(cap, 128, seed=0)
        solver.spill_reasons = {}
        solver._bufs = buffers
        solver._rng = rnd.Random(0)
        solver._precompute_topology()
        solver.chosen = [0] * len(buffers)
        solver.packer = solver._build_seed_packer()
        return solver

    def test_pins_sort_after_every_placed_buffer(self):
        checked = 0
        for case, gi, buffers in _all_cases_incl_synthetic():
            bufs = copy.deepcopy(buffers)
            if not any(b.residency_reason is not None for b in bufs):
                continue
            for cap in _capacities(bufs):
                solver = self._seed(copy.deepcopy(bufs), cap)
                pi = list(solver.packer.permutation)
                addrs = solver.packer.addresses
                pos = {idx: p for p, idx in enumerate(pi)}
                placed = [i for i in range(len(bufs)) if addrs[i] is not None]
                pinned = [
                    i
                    for i, b in enumerate(solver._bufs)
                    if b.residency_reason is not None
                ]
                if not placed or not pinned:
                    continue
                checked += 1
                tag = f"{case}[{gi}] cap={cap}"
                self.assertLess(
                    max(pos[i] for i in placed),
                    min(pos[i] for i in pinned),
                    f"{tag}: a pinned buffer sits before a placed one in pi",
                )
        self.assertGreater(checked, 0, "no pinned graph exercised")

    def test_pi_remains_a_permutation_of_every_buffer(self):
        """Pins are re-ordered, never dropped: the packer's ``eligible`` mask is
        index-aligned with the buffer list, so ``pi`` must keep all ``n`` slots."""
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                solver = self._seed(copy.deepcopy(buffers), cap)
                pi = list(solver.packer.permutation)
                self.assertEqual(
                    sorted(pi), list(range(len(buffers))), f"{case}[{gi}] cap={cap}"
                )

    def test_pins_are_never_placed_by_the_seed(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                solver = self._seed(copy.deepcopy(buffers), cap)
                for i, b in enumerate(solver._bufs):
                    if b.residency_reason is not None:
                        self.assertIsNone(
                            solver.packer.addresses[i],
                            f"{case}[{gi}] cap={cap} {b.name}: pinned but placed",
                        )


class DeterminismTest(TestCase):
    """Two runs on identical input are bit-for-bit identical (Plan §6.5/§7.5)."""

    def _run(self, buffers, cap, seed):
        bufs = copy.deepcopy(buffers)
        solver = SaCoOptimizingSolver(cap, 128, seed=seed)
        out = solver.plan_layout_and_core_divisions(bufs)
        return (
            [b.chosen_division for b in out],
            [b.address for b in out],
            solver.best_score,
            dict(solver.spill_reasons),
        )

    def test_same_seed_bit_identical(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                a = self._run(buffers, cap, seed=0)
                b = self._run(buffers, cap, seed=0)
                self.assertEqual(a, b, f"{case}[{gi}] cap={cap}")

    def test_different_seeds_still_valid(self):
        # A different seed may explore differently, but each run must still be a
        # valid, >=-baseline solution (guards against seed-dependent breakage).
        for case, gi, buffers in _all_cases_incl_synthetic():
            cap = _seed_footprint(buffers)
            for seed in (1, 7):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(cap, 128, seed=seed)
                solver.plan_layout_and_core_divisions(bufs)
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
                solver.plan_layout_and_core_divisions(bufs)
                if solver.best_score < solver.baseline_score:
                    improved = True
        self.assertTrue(improved, "SA never improved on the seed on any graph")


def _div(partition):
    """A core division with the given output partition (1 == trivial/whole)."""
    return CoreDivision(
        output_splits=({1: partition} if partition > 1 else {}),
        reduction_splits={},
    )


def _cdbuf(name, parents, matches, size=1024, uses=(0, 1)):
    """A minimal buffer with a 3-entry menu (index 0 trivial, 1 split-2, 2
    split-4) and the given parent-compatibility pairs. ``size`` / ``uses`` are
    overridable for the fixtures that need layout pressure (the flood tests do
    not care)."""
    return CoreDivisionBuffer(
        name=name,
        size=size,
        uses=list(uses),
        first_use_is_read=False,
        in_place_parents=[],
        residency_reason=None,
        core_divisions=[_div(1), _div(2), _div(4)],
        parents=parents,
        cd_parent_matches=matches,
        boundary=BufferType.Intermediate,
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
        for case, gi, buffers in _all_cases_incl_synthetic():
            tot = _seed_footprint(buffers)
            solver = SaCoOptimizingSolver(max(1, tot // 2), 128, seed=0)
            solver.plan_layout_and_core_divisions(copy.deepcopy(buffers))
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
                solver.plan_layout_and_core_divisions(copy.deepcopy(buffers))
                total_improved += solver.recolor_improved
        self.assertGreater(total_improved, 0, "recolor never improved on any graph")

    def test_recolor_instrumentation_deterministic(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            tot = _seed_footprint(buffers)

            def run():
                s = SaCoOptimizingSolver(max(1, tot // 2), 128, seed=0)
                s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
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

        for case, gi, buffers in _all_cases_incl_synthetic():
            tot = _seed_footprint(buffers)
            solver = SaCoOptimizingSolver(max(1, tot // 2), 128, seed=0)
            solver.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            proposed = solver.recolor_anchor_partitions
            accepted = solver.recolor_accepted_partitions
            tag = f"{case}[{gi}]"
            self.assertEqual(len(proposed), solver.moves_proposed["recolor"], tag)
            self.assertEqual(len(accepted), solver.moves_accepted["recolor"], tag)
            self.assertTrue(all(p > 1 for p in proposed), tag)
            self.assertLessEqual(Counter(accepted), Counter(proposed), tag)


def _chain(n=8):
    """A ``B0 -> ... -> B{n-1}`` chain, every edge compatible index-for-index, with
    varied sizes and staggered lifetimes -- so layout moves genuinely shift
    addresses and packer quality (equal-sized buffers sharing one lifetime are
    permutation-insensitive, which would make the assertions below vacuous)."""
    bufs = []
    for i in range(n):
        parents = [f"B{i - 1}"] if i else []
        matches = {f"B{i - 1}": [(0, 0), (1, 1), (2, 2)]} if i else {}
        bufs.append(
            _cdbuf(f"B{i}", parents, matches, size=1024 * (1 + i % 4), uses=[i, i + 3])
        )
    return bufs


def _chain_caps(buffers):
    """Roomy plus two spill-forcing capacities for a hand-built fixture."""
    tot = sum(b.size for b in buffers)
    return [tot, max(1, tot // 2), max(1, tot // 3)]


def _seeded(buffers, capacity, **kwargs):
    """A solver primed to the seed state (index-0 divisions, FirstFit ``pi``) on
    hand-built buffers: the prefix of ``plan_layout_and_core_divisions`` up to the
    anneal, so a unit test can drive the move / snapshot machinery directly."""
    solver = SaCoOptimizingSolver(capacity, 128, seed=0, **kwargs)
    solver._bufs = buffers
    solver._rng = rnd.Random(0)
    solver._precompute_topology()
    solver.chosen = [0] * len(buffers)
    solver.packer = solver._build_seed_packer()
    solver._flippable_ops = solver._flippable()
    return solver


def _live_state(solver):
    """The observable joint state: layout addresses, packer quality, divisions."""
    return (
        list(solver.packer.addresses),
        solver.packer.quality(),
        list(solver.chosen),
    )


def _snap_state(snap):
    """The same observables read off a snapshot tuple instead of the live state."""
    return (list(snap[0].addresses), snap[0].quality(), list(snap[1]))


class SnapshotRestoreTest(TestCase):
    """The two restore contracts. ``_adopt`` transfers ownership -- the hot
    rejection path, where the snapshot was taken this iteration and dies with it,
    so a second O(n) packer copy would be pure overhead. ``_restore_copy`` leaves
    the snapshot intact, which is what a *retained* snapshot (``_best_snap``)
    requires: the engine keeps mutating the live packer in place afterwards, and
    only refreshes ``_best_snap`` on an improvement, so aliasing would leave the
    recorded best layout describing a state ``_best_score`` never scored."""

    def _mutate(self, solver):
        """A division change (resize + eligibility ripple) plus a reinsertion --
        between them they move addresses, quality and ``chosen``."""
        solver._atomic_flip(2, 2)
        solver.packer.rotate(0, 5)

    def test_adopt_round_trips_state(self):
        for cap in _chain_caps(_chain()):
            solver = _seeded(_chain(), cap)
            before = _live_state(solver)
            snap = solver._snapshot()
            self._mutate(solver)
            self.assertNotEqual(_live_state(solver), before, f"cap={cap}")
            solver._adopt(snap)  # snap is dead after this, by contract
            self.assertEqual(_live_state(solver), before, f"cap={cap}")

    def test_restore_copy_round_trips_state(self):
        for cap in _chain_caps(_chain()):
            solver = _seeded(_chain(), cap)
            before = _live_state(solver)
            snap = solver._snapshot()
            self._mutate(solver)
            self.assertNotEqual(_live_state(solver), before, f"cap={cap}")
            solver._restore_copy(snap)
            self.assertEqual(_live_state(solver), before, f"cap={cap}")

    def test_retained_snapshot_isolated_from_live_packer(self):
        # The one that pins the aliasing defect: restoring from a snapshot the
        # engine keeps must not hand the live state those same objects.
        for cap in _chain_caps(_chain()):
            solver = _seeded(_chain(), cap)
            retained = solver._snapshot()
            recorded = _snap_state(retained)
            solver._restore_copy(retained)
            self._mutate(solver)
            tag = f"cap={cap}"
            # Non-vacuous: the mutation really did change the live state.
            self.assertNotEqual(_live_state(solver), recorded, tag)
            self.assertEqual(_snap_state(retained), recorded, tag)


class InnerLayoutLoopTest(TestCase):
    """``_inner_layout_loop`` must leave a layout no worse than the one it was
    handed -- the promise its docstring makes and the nested mode relies on."""

    def test_annealed_inner_loop_never_worsens(self):
        # ``inner_annealed`` accepts worsening reinsertions by Metropolis; at a
        # huge quality temperature *every* one is accepted, so the walk is a pure
        # downhill drift and only counting the entry layout as a best-seen
        # candidate keeps the result from regressing.
        for cap in _chain_caps(_chain()):
            solver = _seeded(_chain(), cap, nested=True, inner_annealed=True)
            entry_q = solver.packer.quality()
            steps = solver._inner_layout_loop(40, 1e12)
            self.assertEqual(steps, 40, f"cap={cap}")
            self.assertGreaterEqual(solver.packer.quality(), entry_q, f"cap={cap}")

    def test_greedy_inner_loop_never_worsens(self):
        # The default (greedy-cold) path reverts every worsening step in place,
        # so it needs no entry snapshot -- assert the same guarantee holds.
        for cap in _chain_caps(_chain()):
            solver = _seeded(_chain(), cap, nested=True)
            entry_q = solver.packer.quality()
            solver._inner_layout_loop(40, 1.0)
            self.assertGreaterEqual(solver.packer.quality(), entry_q, f"cap={cap}")


class StepBudgetTest(TestCase):
    """``clamp(steps_per_buffer * n, min_steps, max_steps)`` -- the same shape the
    layout-only annealer's schedule uses, so neither engine grows without bound."""

    @staticmethod
    def _budget(solver, n):
        """The budget ``_anneal`` would compute for ``n`` buffers."""
        return min(
            solver._max_steps,
            max(solver._min_steps, solver._steps_per_buffer * n),
        )

    def test_rate_applies_between_the_floor_and_the_ceiling(self):
        s = SaCoOptimizingSolver(1 << 20, 128)
        self.assertEqual(self._budget(s, 100), s._steps_per_buffer * 100)

    def test_floor_applies_to_tiny_graphs(self):
        s = SaCoOptimizingSolver(1 << 20, 128)
        self.assertEqual(self._budget(s, 1), s._min_steps)

    def test_ceiling_caps_large_graphs(self):
        s = SaCoOptimizingSolver(1 << 20, 128)
        binds_at = s._max_steps // s._steps_per_buffer
        self.assertEqual(self._budget(s, binds_at * 4), s._max_steps)
        # Inert across the validated corpus: the largest captured graph is n=79,
        # far below where the ceiling starts binding. If this ever fails, the
        # clamp has begun changing committed benchmark numbers.
        self.assertGreater(binds_at, 79)

    def test_ceiling_is_higher_than_the_layout_only_annealer(self):
        """The joint engine searches divisions too, so it wants a larger budget
        at the same buffer count (and must not silently inherit the smaller one).
        """
        from torch_spyre._inductor.scratchpad.cooling_schedules import (
            SelfCalibratingReheatingSchedule,
        )

        layout_only = SelfCalibratingReheatingSchedule().max_steps
        self.assertGreater(SaCoOptimizingSolver(1 << 20, 128)._max_steps, layout_only)


class ScheduleTest(TestCase):
    """Plan §5 / §8.3: the reheating schedule + cycle-phase mix beats the crude
    baseline, records per-move acceptance traces + within-group CVs, and stays
    deterministic and >= baseline."""

    def test_reheating_beats_crude_overall(self):
        # Aggregate over the captures at a tight capacity: the reheating schedule
        # must beat the crude one *in total* (same seed, same budget) and strictly
        # better on at least one graph. Per-graph domination is deliberately NOT
        # asserted: the §5.1 claim is that reheating wins on average, not on every
        # graph -- both are heuristics at a fixed budget/seed, and a large graph
        # can regress (e.g. reheating scores ~3% worse than crude on the 43-buffer
        # flash_attention capture) while the aggregate still favors reheating. That
        # per-graph regression is a Phase-5 schedule-tuning signal, not a bug, so
        # the test tracks the honest aggregate claim.
        #
        # Pinned to ``reorder_move="random"``, the move the §5.1 claim was
        # measured against. Under the (now default) best-first sweep the ordering
        # *reverses*: the sweep lifts ``crude`` far more than ``reheating`` --
        # 41.44M -> 35.10M on flash_attention against 41.38M -> 41.26M -- so crude
        # wins the aggregate. See ``test_sweep_reverses_the_schedule_ordering``,
        # which pins that finding and explains the (geometric, not feedback)
        # cause. Retuning the schedule for the sweep is the open Phase-5
        # follow-up; until then this keeps testing the claim on the configuration
        # it was made for rather than silently inverting it.
        total_reheat = total_crude = 0
        strictly_better = False
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            r = SaCoOptimizingSolver(
                cap, 128, seed=0, schedule="reheating", reorder_move="random"
            )
            r.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            c = SaCoOptimizingSolver(
                cap, 128, seed=0, schedule="crude", reorder_move="random"
            )
            c.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            total_reheat += r.best_score
            total_crude += c.best_score
            strictly_better = strictly_better or r.best_score < c.best_score
        self.assertLess(total_reheat, total_crude)
        self.assertTrue(strictly_better, "reheating never beat crude on any graph")

    def test_sweep_reverses_the_schedule_ordering(self):
        """Under the default best-first sweep, ``crude`` beats ``reheating`` in
        aggregate -- the opposite of ``test_reheating_beats_crude_overall``.

        Pinned deliberately. The sweep is a much stronger reorder, and ``crude``
        (fixed weights, one geometric cool to ``t0 / 1000``) converts that into a
        large win while ``reheating`` barely moves.

        The cause is the band's *geometry*, not any acceptance feedback --
        ``SelfCalibratingReheatingSchedule.update`` ignores its ``accepted``
        argument. A band ``(hi, lo)`` only fixes the temperature range in units of
        the streamed move scale ``d_hat``: top ``d_hat / -ln(hi)``, bottom
        ``d_hat / -ln(lo)``. The shipped ``reorder`` band (0.6, 0.02) thus cycles
        between 1.96*d_hat and 0.256*d_hat and never reaches a cold, near-greedy
        phase, which is precisely what a strong reorder move exploits. Retuning is
        the open follow-up; this test fails the day it lands, which is exactly
        when both assertions should be revisited.
        """
        total_reheat = total_crude = 0
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            r = SaCoOptimizingSolver(cap, 128, seed=0, schedule="reheating")
            r.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            c = SaCoOptimizingSolver(cap, 128, seed=0, schedule="crude")
            c.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            total_reheat += r.best_score
            total_crude += c.best_score
        self.assertLess(total_crude, total_reheat)

    def test_reorder_acceptance_rate_overshoots_its_band(self):
        """The realized ``reorder`` acceptance rate sits far above the band's
        ``accept_hi``, under every reorder variant.

        Not the cause of the reversal above -- the band is geometry, not a
        control loop, so nothing is trying to hold the rate inside it. It is
        pinned because it characterizes the *objective*: only spilled buffers are
        priced, making the score a coarse step function of ``pi`` that most
        rotations leave exactly unchanged. A zero delta is accepted
        unconditionally, so the floor on the acceptance rate is
        ``P(delta <= 0)``, which no temperature can lower. That is also why
        ranking the sweep by the packer's continuous ``quality()`` beats ranking
        it by the objective itself."""
        accept_hi = _DEFAULT_MOVE_BANDS["reorder"][0]
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            for move in ("random", "sweep_quality"):
                s = SaCoOptimizingSolver(cap, 128, seed=0, reorder_move=move)
                s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
                rate = s.moves_accepted["reorder"] / s.moves_proposed["reorder"]
                self.assertGreater(rate, accept_hi, f"{case} {move} rate={rate}")

    def test_per_move_acceptance_traces_recorded(self):
        # Every applicable move type is proposed and its accepted count is a valid
        # subset -- the §8.3 per-move-type acceptance traces.
        for case, gi, buffers in _all_cases_incl_synthetic():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = SaCoOptimizingSolver(cap, 128, seed=0)
            s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            for m in ("reorder", "flip", "recolor"):
                self.assertGreater(s.moves_proposed[m], 0, f"{case} {m}")
                self.assertLessEqual(
                    s.moves_accepted[m], s.moves_proposed[m], f"{case} {m}"
                )

    def test_within_group_cv_available_and_finite(self):
        # The §5.3 within-group-CV instrumentation is populated and non-negative
        # (drives the deferred variance-bucketing decision).
        for case, gi, buffers in _all_cases_incl_synthetic():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = SaCoOptimizingSolver(cap, 128, seed=0)
            s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            cv = s.move_scale_cv()
            self.assertEqual(set(cv), {"reorder", "flip", "recolor", "none"})
            for m, v in cv.items():
                self.assertGreaterEqual(v, 0.0, f"{case} {m}")
                self.assertTrue(math.isfinite(v), f"{case} {m}")

    def test_reheating_deterministic(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            cap = max(1, _seed_footprint(buffers) // 2)

            def run():
                s = SaCoOptimizingSolver(cap, 128, seed=0, schedule="reheating")
                out = s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
                return (
                    [b.chosen_division for b in out],
                    [b.address for b in out],
                    s.best_score,
                    s.moves_proposed,
                    s.moves_accepted,
                )

            self.assertEqual(run(), run(), f"{case}[{gi}]")

    def test_both_schedules_respect_baseline(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for sched in ("reheating", "crude"):
                cap = max(1, _seed_footprint(buffers) // 2)
                s = SaCoOptimizingSolver(cap, 128, seed=0, schedule=sched)
                s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
                self.assertLessEqual(s.best_score, s.baseline_score, f"{case} {sched}")

    def test_rejects_bad_schedule(self):
        with self.assertRaises(ValueError):
            SaCoOptimizingSolver(1024, 128, schedule="nope")


# Snippet run in a subprocess to solve one graph (captured *or* synthetic, chosen
# by CASE) and print its result; used by the cross-process determinism test below.
_SOLVE_SNIPPET = """
import copy, json, math
from tests.inductor.cooptimization_capture_loader import load_captures
from tests.inductor.synthetic_cooptimization_graphs import synthetic_graphs
from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver
case = {case!r}
src = load_captures() if case in load_captures() else synthetic_graphs()
g = src[case][0]
cap = max(1, sum(math.ceil(b.size / b.core_divisions[0].output_partition)
                 for b in g.buffers) // 2)
s = SaCoOptimizingSolver(cap, 128, seed=0)
out = s.plan_layout_and_core_divisions(copy.deepcopy(g.buffers))
print("RESULT " + json.dumps({{
    "chosen": [b.chosen_division for b in out],
    "addr": [b.address for b in out],
    "best": s.best_score,
}}))
"""

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _solve_with_hashseed(hs, case="sdpa"):
    """Solve ``case`` in a subprocess with ``PYTHONHASHSEED=hs``."""
    env = dict(os.environ, PYTHONHASHSEED=str(hs), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    proc = subprocess.run(
        [sys.executable, "-c", _SOLVE_SNIPPET.format(case=case)],
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

    # ``sdpa`` is the richest captured graph (pins + reductions); ``big_chain`` is
    # the largest synthetic one (many regions / n~48), so between them they stress
    # the most set-ordered decisions (flood order, candidate lists, best-seen ties).
    def test_pythonhashseed_independent(self):
        for case in ("sdpa", "big_chain"):
            base = _solve_with_hashseed(0, case)
            for hs in (1, 2):
                self.assertEqual(
                    _solve_with_hashseed(hs, case), base, f"{case} PYTHONHASHSEED={hs}"
                )


@unittest.skipUnless(
    _large_captures_enabled(),
    f"large-capture experiments; set {_LARGE_CAPTURES_ENV}=1 to run",
)
class LargeCaptureExperimentTest(TestCase):
    """Opt-in (non-CI) experiments over the large 25-100 buffer captures. These
    exist to *surface findings*, not to gate CI -- so they assert only the
    shape-invariant guarantees (determinism, >= baseline) and otherwise *report*
    rather than fail. Run with ``SA_COOPT_LARGE_CAPTURES=1``."""

    def test_large_graphs_valid_and_deterministic(self):
        # The engine must stay correct on big n: contract + determinism + baseline.
        for case, gi, buffers in _large_cases():
            cap = max(1, _seed_footprint(buffers) // 2)

            def run():
                s = SaCoOptimizingSolver(cap, 128, seed=0)
                out = s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
                return (
                    [b.chosen_division for b in out],
                    [b.address for b in out],
                    s.best_score,
                )

            a, b = run(), run()
            self.assertEqual(a, b, f"{case}[{gi}] nondeterministic")
            s = SaCoOptimizingSolver(cap, 128, seed=0)
            s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            self.assertLessEqual(s.best_score, s.baseline_score, f"{case}[{gi}]")

    def test_report_reheating_vs_crude_by_size(self):
        # The flash_big-style finding, reproducible across small + large: reheating
        # ties/wins on small-to-mid graphs but regresses on the largest flash
        # graphs. Reports the per-graph deltas; asserts nothing about quality.
        print("\n  case               n   reheat        crude         delta")
        for case, gi, buffers in list(_all_cases()) + list(_large_cases()):
            cap = max(1, _seed_footprint(buffers) // 2)
            r = SaCoOptimizingSolver(cap, 128, seed=0, schedule="reheating")
            r.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            c = SaCoOptimizingSolver(cap, 128, seed=0, schedule="crude")
            c.plan_layout_and_core_divisions(copy.deepcopy(buffers))
            d = r.best_score - c.best_score
            tag = "better" if d < 0 else ("worse" if d > 0 else "tie")
            print(
                f"  {case:16} {len(buffers):3d}  {r.best_score:12d}  "
                f"{c.best_score:12d}  {d:+d} {tag}"
            )


_SWEEP_ARMS = [
    dict(reorder_move="sweep_quality"),
    dict(reorder_move="sweep_score"),
    dict(reorder_move="sweep_quality", sweep_biased_i=False),
    dict(reorder_move="sweep_score", sweep_cleanup=True),
]


def _armed(buffers, cap, seed=0, **kwargs):
    """A solver stopped just short of annealing: topology precomputed, seed state
    built, per-step bookkeeping initialized. Lets a test drive one move directly.
    """
    s = SaCoOptimizingSolver(cap, 128, seed=seed, **kwargs)
    s.spill_reasons = {}
    zeros = {"reorder": 0, "flip": 0, "recolor": 0, "none": 0}
    s.moves_proposed = dict(zeros)
    s.moves_accepted = dict(zeros)
    s._ms_n = dict(zeros)
    s._ms_sum = {k: 0.0 for k in zeros}
    s._ms_sqsum = {k: 0.0 for k in zeros}
    s.sweep_probes = s.sweep_evals = s.sweep_steps = 0
    s._bufs = buffers
    s._rng = rnd.Random(seed)
    s._precompute_topology()
    s.chosen = [0] * len(buffers)
    s.packer = s._build_seed_packer()
    s._best_score = s._score()
    s._best_snap = s._snapshot()
    return s


def _score_after_rotate(s, i, j):
    """The objective reached by rotating position ``i`` to ``j``, leaving ``s``
    exactly as it was found."""
    snap = s._snapshot()
    s.packer.rotate(i, j)
    value = s._score()
    s._adopt(snap)  # a fresh snapshot is taken per call, so this transfer is safe
    return value


class SweepReorderMoveTest(TestCase):
    """The layout-only annealer's best-first reinsertion move, ported to the joint
    objective (``reorder_move="sweep_quality" | "sweep_score"``)."""

    def test_rejects_unknown_reorder_move(self):
        with self.assertRaises(ValueError):
            SaCoOptimizingSolver(1024, 128, reorder_move="best_first")

    def test_sweep_arms_hold_the_shape_invariants(self):
        # The same contract every arm owes: a division and a spill reason or an
        # address for every buffer, never worse than the seed, bit-reproducible.
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                for arm in _SWEEP_ARMS:
                    tag = f"{case}[{gi}] cap={cap} {arm}"
                    runs = []
                    for _ in range(2):
                        s = SaCoOptimizingSolver(cap, 128, seed=0, **arm)
                        out = s.plan_layout_and_core_divisions(copy.deepcopy(buffers))
                        self.assertLessEqual(s.best_score, s.baseline_score, tag)
                        for b in out:
                            self.assertIsNotNone(b.chosen_division, f"{tag} {b.name}")
                            if b.address is None:
                                self.assertIn(
                                    b.name, s.spill_reasons, f"{tag} {b.name}"
                                )
                            else:
                                self.assertNotIn(b.name, s.spill_reasons, tag)
                        runs.append((s.best_score, [b.chosen_division for b in out]))
                    self.assertEqual(runs[0], runs[1], f"{tag} not deterministic")

    def test_probe_walk_leaves_the_packer_consistent(self):
        """The sweep walks the *live* packer and restores from the step snapshot,
        so a bookkeeping slip would show up as incremental state that disagrees
        with a packer rebuilt from scratch on the same permutation."""
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            for arm in _SWEEP_ARMS:
                s = _armed(copy.deepcopy(buffers), cap, **arm)
                if len(s._bufs) < 2:
                    continue
                cur = s._score()
                for step in range(40):
                    cur, _, _ = s._step_reorder_sweep(1000.0, cur)
                    tag = f"{case}[{gi}] {arm} step={step}"
                    # The returned running score must be the state's real score.
                    self.assertEqual(cur, s._score(), tag)
                    # And the incrementally-maintained placement must match a
                    # from-scratch rebuild on the permutation it ended up with.
                    sizes = [
                        s._per_core_size(i, s.chosen[i]) for i in range(len(s._bufs))
                    ]
                    fresh = make_permutation_packer(
                        s._lifetime_buffers(sizes),
                        list(s.packer.permutation),
                        s.limit,
                        s.alignment,
                        eligible=[s._eligible(i) for i in range(len(s._bufs))],
                    )
                    self.assertEqual(
                        list(fresh.addresses), list(s.packer.addresses), tag
                    )
                    self.assertEqual(fresh.quality(), s.packer.quality(), tag)

    def test_cold_sweep_lands_on_the_best_reinsertion(self):
        """At a temperature that accepts nothing uphill, ``sweep_score`` must end
        on the best-scoring reinsertion it probed -- that is what "best-first"
        buys over one random sample."""
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = _armed(copy.deepcopy(buffers), cap, reorder_move="sweep_score")
            n = len(s._bufs)
            if n < 2:
                continue
            cur = s._score()
            for step in range(25):
                perm = s.packer.permutation
                allocated = [s.packer.is_fully_allocated(perm[k]) for k in range(n)]
                # Replay the source pick against a clone of the RNG so the brute
                # force below targets the same buffer the step will lift.
                probe_rng = copy.deepcopy(s._rng)
                saved, s._rng = s._rng, probe_rng
                i = s._choose_reinsertion_source(allocated)
                s._rng = saved
                upper = s._sweep_upper_bound(i, allocated)
                brute = {
                    j: _score_after_rotate(s, i, j) for j in range(upper + 1) if j != i
                }
                before = cur
                cur, accepted, _ = s._step_reorder_sweep(1e-12, cur)
                tag = f"{case}[{gi}] step={step} i={i}"
                best = min(brute.values())
                if best <= before:
                    self.assertTrue(accepted, tag)
                    self.assertEqual(cur, best, tag)
                else:
                    # Everything reachable is uphill and the temperature is cold,
                    # so the step must decline and leave the score where it was.
                    self.assertFalse(accepted, tag)
                    self.assertEqual(cur, before, tag)

    def test_monotonicity_bound_hides_no_better_position(self):
        """The sweep inherits the layout-only annealer's bound: an unallocated
        buffer is only probed up to the last allocated position + 1. Check on real
        graphs that nothing past the bound would have scored better."""
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = _armed(copy.deepcopy(buffers), cap, reorder_move="sweep_score")
            n = len(s._bufs)
            if n < 2:
                continue
            cur = s._score()
            for step in range(25):
                perm = s.packer.permutation
                allocated = [s.packer.is_fully_allocated(perm[k]) for k in range(n)]
                for i in range(n):
                    upper = s._sweep_upper_bound(i, allocated)
                    if upper >= n - 1:
                        continue  # unbounded; nothing was skipped
                    inside = min(
                        (
                            _score_after_rotate(s, i, j)
                            for j in range(upper + 1)
                            if j != i
                        ),
                        default=cur,
                    )
                    for j in range(upper + 1, n):
                        self.assertGreaterEqual(
                            _score_after_rotate(s, i, j),
                            inside,
                            f"{case}[{gi}] step={step} i={i} j={j} beat the bound",
                        )
                cur, _, _ = s._step_reorder_sweep(1000.0, cur)


if __name__ == "__main__":
    unittest.main()
