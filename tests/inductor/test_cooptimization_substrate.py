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

"""Phase-0 validation for the co-optimization substrate seam (Plan §8.3).

Two things are checked here:

* **type-checks / conformance** -- the fake substrate's replayed objects
  structurally satisfy the §7 coupling-surface protocols the SA engine codes
  against (:class:`CoreDivisionProtocol`, :class:`CoreDivisionBufferProtocol`),
  and a trivial concrete solver satisfies :class:`CoOptimizingSolver`; and
* **fake round-trips** -- every captured graph loads into well-formed buffers,
  the solved reference is self-consistent with those buffers, and the §8.2 seed
  (``chosen_division = 0`` for every op) is a valid state.
"""

import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad.cooptimization_substrate import (
    CoOptimizingSolver,
    CoreDivisionBufferProtocol,
    CoreDivisionProtocol,
)

from tests.inductor.fake_cooptimization_substrate import (
    SEED_DIVISION_INDEX,
    FakeCoreDivisionBuffer,
    load_captures,
)


class _NoOpSolver(CoOptimizingSolver):
    """Minimal concrete engine: seeds every buffer, spills everything.

    Exercises the ABC contract (must implement ``plan_layout_and_core_divs``)
    without depending on any real solving machinery.
    """

    def plan_layout_and_core_divs(self, buffers, log_lx_usage=False):
        for b in buffers:
            b.chosen_division = SEED_DIVISION_INDEX
            b.address = None  # spilled
            self.spill_reasons[b.name] = "no-op solver spills everything"
        return list(buffers)


class CoOptimizationSubstrateConformanceTest(TestCase):
    """The fake structurally satisfies the §7 coupling-surface protocols."""

    def setUp(self):
        self.cases = load_captures()

    def test_captures_load_and_are_non_empty(self):
        self.assertGreater(len(self.cases), 0)
        for case, graphs in self.cases.items():
            self.assertGreater(len(graphs), 0, f"{case} has no graphs")
            for gi, g in enumerate(graphs):
                self.assertGreater(len(g.buffers), 0, f"{case}[{gi}] has no buffers")

    def test_divisions_conform_to_protocol(self):
        for case, graphs in self.cases.items():
            for gi, g in enumerate(graphs):
                for b in g.buffers:
                    for cd in b.core_divisions:
                        self.assertIsInstance(
                            cd,
                            CoreDivisionProtocol,
                            f"{case}[{gi}] {b.name}: division not conforming",
                        )

    def test_buffers_conform_to_protocol(self):
        for case, graphs in self.cases.items():
            for gi, g in enumerate(graphs):
                for b in g.buffers:
                    self.assertIsInstance(
                        b,
                        CoreDivisionBufferProtocol,
                        f"{case}[{gi}] {b.name}: buffer not conforming",
                    )


class CoOptimizationSubstrateRoundTripTest(TestCase):
    """Each captured graph is a well-formed, self-consistent SA problem."""

    def setUp(self):
        self.cases = load_captures()

    def test_buffers_are_well_formed(self):
        for case, graphs in self.cases.items():
            for gi, g in enumerate(graphs):
                for b in g.buffers:
                    ctx = f"{case}[{gi}] {b.name}"
                    # Liveness must be usable by the packer.
                    self.assertGreater(len(b.uses), 0, f"{ctx}: empty uses")
                    self.assertEqual(b.uses, sorted(b.uses), f"{ctx}: unsorted uses")
                    self.assertLess(b.start_time, b.end_time, f"{ctx}: bad lifetime")
                    # Every buffer carries at least the seed division (index 0).
                    self.assertGreater(
                        len(b.core_divisions), 0, f"{ctx}: no candidate divisions"
                    )
                    # residency_reason and the placement pin agree.
                    if b.residency_reason is not None:
                        self.assertFalse(
                            b.residency_allowed, f"{ctx}: pinned but placement True"
                        )

    def test_edges_reference_known_buffers(self):
        for case, graphs in self.cases.items():
            for gi, g in enumerate(graphs):
                names = set(g.by_name())
                for b in g.buffers:
                    ctx = f"{case}[{gi}] {b.name}"
                    for p in b.parents:
                        self.assertIn(p, names, f"{ctx}: parent {p!r} unknown")
                    # cd_parent_matches keys are a subset of declared parents,
                    # and every matched pair indexes valid menu entries.
                    for p, pairs in b.cd_parent_matches.items():
                        self.assertIn(p, b.parents, f"{ctx}: match parent {p!r}")
                        parent = g.by_name()[p]
                        for pj, cj in pairs:
                            self.assertTrue(
                                0 <= pj < len(parent.core_divisions),
                                f"{ctx}: parent idx {pj} out of range",
                            )
                            self.assertTrue(
                                0 <= cj < len(b.core_divisions),
                                f"{ctx}: child idx {cj} out of range",
                            )

    def test_solved_reference_is_consistent(self):
        for case, graphs in self.cases.items():
            for gi, g in enumerate(graphs):
                by_name = g.by_name()
                self.assertLessEqual(
                    set(g.solved), set(by_name), f"{case}[{gi}]: solved/name mismatch"
                )
                for name, sol in g.solved.items():
                    ctx = f"{case}[{gi}] {name}"
                    b = by_name[name]
                    cd_idx = sol["chosen_division"]
                    if cd_idx is not None:
                        self.assertTrue(
                            0 <= cd_idx < len(b.core_divisions),
                            f"{ctx}: chosen_division {cd_idx} out of range",
                        )
                    # A resident buffer has an address and was residency-allowed.
                    if sol["resident"]:
                        self.assertIsNotNone(
                            sol["address"], f"{ctx}: resident, no addr"
                        )
                        self.assertTrue(
                            b.residency_allowed, f"{ctx}: resident but pinned out"
                        )

    def test_seed_is_a_valid_state(self):
        # Plan §8.2: the SA seed is chosen_division=0 for every op. Index 0 must
        # exist for every buffer, and writing it must round-trip through the
        # solver-output fields added to the fake buffer.
        for case, graphs in self.cases.items():
            for gi, g in enumerate(graphs):
                for b in g.buffers:
                    self.assertGreater(len(b.core_divisions), SEED_DIVISION_INDEX)
                    b.chosen_division = SEED_DIVISION_INDEX
                    self.assertEqual(b.chosen_division, SEED_DIVISION_INDEX)
                    self.assertIsNone(b.address)


class CoOptimizingSolverABCTest(TestCase):
    """The local ``CoOptimizingSolver`` ABC behaves as the Phase-3 base."""

    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            CoOptimizingSolver(1024)  # type: ignore[abstract]

    def test_concrete_subclass_solves_and_writes_outputs(self):
        graph = load_captures()["softmax"][0]
        solver = _NoOpSolver(size=4096, alignment=128)
        self.assertEqual(solver.limit, 4096)
        self.assertEqual(solver.alignment, 128)
        self.assertEqual(solver.spill_reasons, {})

        out = solver.plan_layout_and_core_divs(graph.buffers)
        self.assertEqual(len(out), len(graph.buffers))
        for b in out:
            self.assertEqual(b.chosen_division, SEED_DIVISION_INDEX)
            self.assertIsNone(b.address)
            self.assertIn(b.name, solver.spill_reasons)

    def test_spill_cost_matches_substrate_formula(self):
        b = FakeCoreDivisionBuffer(
            name="x",
            size=2048,
            uses=[0, 1],
            first_use_is_read=False,
            in_place_parents=[],
            placement=True,
            residency_reason=None,
            boundary_cost=0,
            spill_write_cost=2048,
            parents=[],
            core_divisions=[],
            cd_parent_matches={},
        )
        # num_children * size + spill_write_cost
        self.assertEqual(CoOptimizingSolver._spill_cost(b, 3), 3 * 2048 + 2048)
        self.assertEqual(CoOptimizingSolver._spill_cost(b, 0), 2048)


if __name__ == "__main__":
    unittest.main()
