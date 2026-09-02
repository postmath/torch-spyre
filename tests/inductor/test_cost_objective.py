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

"""The cached cost-model objective, and its wiring into the SA co-optimizer.

The property that matters most here is that caching cannot change the answer.
``score`` maintains a running total by recomputing only the bundles a move could
have dirtied; ``score_from_scratch`` recomputes everything. They must agree
*exactly*, not approximately -- the engine's determinism guarantee is bit-for-bit,
and a float accumulation would drift the two apart over a long anneal. Most of
these tests are that invariant under different kinds of state change.
"""

import copy
import json
import os
import random
import unittest
from unittest import TestCase, mock

from torch_spyre._inductor.cost_model import op_from_dict
from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective
from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver

FIXTURE = os.path.join(os.path.dirname(__file__), "cooptimization_op_features.json")


def _graph(name="simple_attn"):
    with open(FIXTURE) as fh:
        g = json.load(fh)["graphs"][name]
    names = list(g["buffers"])
    features = {
        n: [None if f is None else op_from_dict(f) for f in b["features"]]
        for n, b in g["buffers"].items()
    }
    _name_output_args_after_their_buffer(features)
    return names, features


def _name_output_args_after_their_buffer(features):
    """Respell each op's output arg with the buffer it writes.

    The fixture predates ``extract_op_features`` naming that arg after the buffer
    (it carries the old ``"op7"`` for a record whose buffer is ``"buf7"``), and
    regenerating it needs a Spyre machine. Without this the fixture-driven tests
    below never cross the seam the naming exists for -- a resident buffer's write
    would silently stay charged, which is the bug the naming fixed.
    """
    for buf, menu in features.items():
        for feat in menu:
            if feat is None:
                continue
            for arg in feat.args:
                if arg.role == "output":
                    arg.name = buf


def _objective(names, features, bundles=None):
    if bundles is None:
        bundles = [names[: len(names) // 2], names[len(names) // 2 :]]
    return BundleCostObjective(names, features, bundles)


class CacheAgreesWithFullRecomputeTest(TestCase):
    """The invariant the whole design rests on."""

    def test_agrees_across_a_random_walk(self):
        names, features = _graph()
        obj = _objective(names, features)
        menu = {n: len(features[n]) for n in names}
        rng = random.Random(0)
        chosen = [0] * len(names)
        resident = frozenset(names[:4])
        for step in range(200):
            i = rng.randrange(len(names))
            chosen[i] = rng.randrange(menu[names[i]])
            if rng.random() < 0.5:
                resident = resident ^ {names[rng.randrange(len(names))]}
            self.assertEqual(
                obj.score(chosen, resident),
                obj.score_from_scratch(chosen, resident),
                f"incremental total drifted at step {step}",
            )

    def test_agrees_when_only_residency_moves(self):
        # Division fixed: exercises the residency half of the dirty map, which
        # fans out further than the division half (an argument is read by ops in
        # bundles other than the one that writes it).
        names, features = _graph()
        obj = _objective(names, features)
        rng = random.Random(1)
        chosen = [0] * len(names)
        resident = frozenset()
        for _ in range(100):
            resident = resident ^ {names[rng.randrange(len(names))]}
            self.assertEqual(
                obj.score(chosen, resident), obj.score_from_scratch(chosen, resident)
            )

    def test_agrees_when_only_divisions_move(self):
        names, features = _graph()
        obj = _objective(names, features)
        menu = {n: len(features[n]) for n in names}
        rng = random.Random(2)
        chosen = [0] * len(names)
        resident = frozenset(names[:3])
        for _ in range(100):
            i = rng.randrange(len(names))
            chosen[i] = rng.randrange(menu[names[i]])
            self.assertEqual(
                obj.score(chosen, resident), obj.score_from_scratch(chosen, resident)
            )

    def test_invalidate_does_not_change_the_score(self):
        # invalidate() drops the diff baseline, not the cached values, so the
        # next score must recompute to the same number.
        names, features = _graph()
        obj = _objective(names, features)
        chosen = [0] * len(names)
        resident = frozenset(names[:5])
        before = obj.score(chosen, resident)
        obj.invalidate()
        self.assertEqual(obj.score(chosen, resident), before)

    def test_revisiting_a_state_gives_the_same_score(self):
        names, features = _graph()
        obj = _objective(names, features)
        chosen = [0] * len(names)
        a = obj.score(chosen, frozenset(names[:2]))
        obj.score([1 if i == 0 else c for i, c in enumerate(chosen)], frozenset())
        self.assertEqual(obj.score(chosen, frozenset(names[:2])), a)


class ResidencyFreesTheProducingWriteTest(TestCase):
    """Residency frees an intermediate's write, not just its reads.

    This is the seam the rest of the module does not touch: the resident set is
    spelled the way the *solver* spells it -- buffer names -- and an op's output
    arg has to be spelled the same way or its write stays charged. Both
    memory-only objectives price that write (``spill_cost``'s
    ``+1 if is_intermediate``), so the cost objective has to as well.
    """

    def test_freeing_the_write_lowers_the_score(self):
        names, features = _graph("softmax")
        chosen = [0] * len(names)
        resident = frozenset(names)
        reads_only = BundleCostObjective(
            names, features, [names], intermediates=frozenset()
        ).score_from_scratch(chosen, resident)
        both = BundleCostObjective(
            names, features, [names], intermediates=frozenset(names)
        ).score_from_scratch(chosen, resident)
        self.assertLess(both, reads_only)

    def test_a_boundary_buffer_keeps_its_write(self):
        # A graph output's write-out survives residency: the clone that pins it
        # is inserted after the solve, so nothing in the featurized graph pays
        # for it. Excluding one buffer from ``intermediates`` must therefore
        # cost strictly more than including it.
        names, features = _graph("softmax")
        chosen = [0] * len(names)
        resident = frozenset(names)
        boundary = names[-1]
        with_it = BundleCostObjective(
            names, features, [names], intermediates=frozenset(names)
        ).score_from_scratch(chosen, resident)
        without = BundleCostObjective(
            names, features, [names], intermediates=frozenset(names) - {boundary}
        ).score_from_scratch(chosen, resident)
        self.assertGreater(without, with_it)

    def test_residency_of_a_write_only_buffer_dirties_its_bundle(self):
        # The write side only reaches the incremental path if the dirty map
        # routes a residency change to the bundle that WRITES the buffer, which
        # it does through the output arg's name. One bundle per buffer makes the
        # routing observable.
        names, features = _graph("softmax")
        obj = BundleCostObjective(
            names, features, [[n] for n in names], intermediates=frozenset(names)
        )
        chosen = [0] * len(names)
        base = obj.score(chosen, frozenset())
        moved = obj.score(chosen, frozenset({names[0]}))
        self.assertNotEqual(moved, base)
        self.assertEqual(moved, obj.score_from_scratch(chosen, frozenset({names[0]})))


class StructureTest(TestCase):
    def test_names_outside_the_solver_are_dropped(self):
        # Bundles legitimately contain names the solver does not own; a bundle
        # emptied by that filter must disappear rather than score 0 and dilute
        # the dirty map.
        #
        # Note the filter keys on membership, NOT on an ``arg`` prefix: a
        # clone-eligible graph input really is one of the solver's buffers, so
        # prefix-matching would wrongly discard it. Which graph owns one is a
        # property of the capture, not of this code -- ``rms_norm`` is the one
        # that does in the current fixture (``simple_attn``, the default
        # elsewhere here, no longer has a clone-eligible input).
        names, features = _graph("rms_norm")
        self.assertIn("arg0_1", names, "fixture no longer exercises the arg case")
        obj = _objective(
            names, features, bundles=[["not_a_buffer", "also_absent"], names]
        )
        self.assertEqual(len(obj._bundles), 1)

    def test_score_is_a_nonnegative_integer(self):
        # Fixed point, not float: see the determinism note in the module.
        names, features = _graph()
        obj = _objective(names, features)
        value = obj.score([0] * len(names), frozenset())
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 0)

    def test_dirty_tracking_beats_full_recompute(self):
        # Not a performance assertion so much as proof the mechanism engages: a
        # single-buffer change must not touch every bundle.
        names, features = _graph()
        bundles = [[n] for n in names]  # one bundle each, to make the count clear
        obj = BundleCostObjective(names, features, bundles)
        chosen = [0] * len(names)
        obj.score(chosen, frozenset())
        baseline = obj.lookups
        chosen[0] = 1
        obj.score(chosen, frozenset())
        self.assertLess(obj.lookups - baseline, len(bundles))


class SolverWiringTest(TestCase):
    """The solver delegates to the objective, and rolls it back correctly."""

    class _StubObjective:
        """Counts calls and scores off the division vector alone, so the test
        does not depend on the cost model's numbers."""

        def __init__(self):
            self.calls = 0
            self.invalidations = 0

        def score(self, chosen, resident):
            self.calls += 1
            return sum(chosen) * 1000 + len(resident)

        def invalidate(self):
            self.invalidations += 1

    @staticmethod
    def _buffers():
        from tests.inductor.cooptimization_capture_loader import load_captures

        return next(iter(load_captures().values()))[0].buffers

    @staticmethod
    def _solve_with(objective, buffers):
        """Solve ``buffers`` with ``objective`` standing in for the one the live
        Inductor graph would have built.

        The engine takes no arguments, so the seam is ``_bundle_objective`` -- the
        one function that reads ``V.graph``. Patching it is also what lets this
        exercise the objective-driven path at all: off-hardware there is no live
        graph, so an unpatched solver falls back to the memory-only objective.
        """
        from torch_spyre._inductor.scratchpad import sa_cooptimizer

        with mock.patch.object(
            sa_cooptimizer, "_bundle_objective", return_value=objective
        ):
            solver = SaCoOptimizingSolver(copy.deepcopy(buffers), 1 << 14, 128)
            out = solver.plan_layout_and_core_divisions()
        return solver, out

    def test_solver_uses_the_objective_and_invalidates_on_rollback(self):
        stub = self._StubObjective()
        self._solve_with(stub, self._buffers())
        self.assertGreater(stub.calls, 0, "solver never consulted the objective")
        self.assertGreater(
            stub.invalidations, 0, "solver never told the objective state rolled back"
        )

    def test_solver_stays_deterministic_with_an_objective(self):
        bufs = self._buffers()

        def run():
            solver, out = self._solve_with(self._StubObjective(), bufs)
            return [b.chosen_division for b in out], solver.best_score

        self.assertEqual(run(), run())

    def test_without_a_live_graph_the_objective_is_absent(self):
        # The fallback seam: no ``V.graph`` -> no objective -> the memory-only
        # score. ``test_sa_cooptimizer.MemoryOnlyFallbackTest`` pins what that
        # objective computes; this pins that the solver really takes that path
        # when the features are unavailable.
        solver = SaCoOptimizingSolver(copy.deepcopy(self._buffers()), 1 << 14, 128)
        self.assertIsNone(solver._cost_objective)


if __name__ == "__main__":
    unittest.main()
