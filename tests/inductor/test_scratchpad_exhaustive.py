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

"""Tests for the exhaustive branch-and-bound layout solver."""

import math
import random as rnd
from itertools import permutations
from unittest import TestCase

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.exhaustive_solver import (
    ExhaustiveLayoutSolver,
    TimeBudget,
    _interacts,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)


def _placed_weight(buffers):
    """Total size of buffers that were assigned an address."""
    return sum(b.size for b in buffers if b.address is not None)


def _assert_valid_packing(test, buffers, limit):
    """Assert the returned addresses form a legal, non-overlapping packing.

    Overlap is checked per (time-column, address-range). The in-place exception
    is allowed: a child sharing its parent's exact address at the single shared
    column is not a violation.
    """
    placed = [b for b in buffers if b.address is not None]
    parents = {b.name: set(b.in_place_parents) for b in buffers}
    for b in placed:
        test.assertGreaterEqual(b.address, 0)
        test.assertLessEqual(b.address + b.size, limit, f"{b.name} exceeds limit")
    for i, a in enumerate(placed):
        for c in placed[i + 1 :]:
            if not _interacts(a, c):
                continue
            # Address ranges disjoint -> fine.
            if a.address + a.size <= c.address or c.address + c.size <= a.address:
                continue
            # Otherwise the only legal overlap is a single shared column between
            # an in-place parent/child pair sharing an address.
            shared_start = max(a.start_time, c.start_time)
            shared_end = min(a.end_time, c.end_time)
            inplace_pair = c.name in parents[a.name] or a.name in parents[c.name]
            test.assertTrue(
                inplace_pair and (shared_end - shared_start) == 1,
                f"illegal overlap between {a.name}@{a.address} and "
                f"{c.name}@{c.address}",
            )


def _random_buffers(random, n, *, size_max=30, time_max=10, inplace_prob=0.25):
    """Generate n random buffers, occasionally chaining an in-place child.

    An in-place child splits an existing buffer's lifetime: the parent's end is
    pulled in to the split tick and the child runs from there, with
    ``parent.end_time == child.start_time + 1`` and ``child.size <= parent.size``
    as required by the in-place contract.
    """
    buffers: list[LifetimeBoundBuffer] = []
    for k in range(n):
        candidates = [
            b
            for b in buffers
            if b.end_time - b.start_time >= 2 and not _is_parent(buffers, b)
        ]
        if candidates and random.random() < inplace_prob:
            parent = random.choice(candidates)
            old_end = parent.end_time
            # The child starts on the parent's (new) last column: the contract
            # is parent.end_time == child.start_time + 1, so they share exactly
            # one column.
            split = random.randrange(parent.start_time + 1, old_end)
            parent.end_time = split + 1
            buffers.append(
                LifetimeBoundBuffer(
                    f"b{k}",
                    random.randrange(1, parent.size + 1),
                    split,
                    old_end,
                    in_place_parents=[parent.name],
                )
            )
            continue
        start = random.randrange(time_max)
        end = start + random.randrange(1, time_max - start + 1)
        buffers.append(
            LifetimeBoundBuffer(f"b{k}", random.randrange(1, size_max + 1), start, end)
        )
    return buffers


def _is_parent(buffers, buf):
    return any(buf.name in b.in_place_parents for b in buffers)


def _brute_force_optimum(solver, buffers):
    """Maximum placed weight over every drop permutation of the active buffers.

    Uses the solver's own drop semantics (via ``_evaluate_order``) so it is a
    faithful oracle for the canonical-representative search.
    """
    solver._prepare(buffers)
    n = len(solver._bufs)
    best = 0
    for order in permutations(range(n)):
        wt, _ = solver._evaluate_order(list(order))
        best = max(best, wt)
    return best


class TestExhaustiveDrop(TestCase):
    """Basic, hand-checked placements."""

    def test_empty(self):
        self.assertEqual(ExhaustiveLayoutSolver(100).plan_layout([]), [])

    def test_single_fits(self):
        buffers = [LifetimeBoundBuffer("a", 10, 0, 5)]
        ExhaustiveLayoutSolver(100, alignment=1).plan_layout(buffers)
        self.assertEqual(buffers[0].address, 0)

    def test_single_too_large_evicted(self):
        buffers = [LifetimeBoundBuffer("a", 101, 0, 5)]
        ExhaustiveLayoutSolver(100, alignment=1).plan_layout(buffers)
        self.assertIsNone(buffers[0].address)

    def test_non_overlapping_reuse_address(self):
        buffers = [
            LifetimeBoundBuffer("a", 20, 0, 5),
            LifetimeBoundBuffer("b", 20, 5, 10),
        ]
        ExhaustiveLayoutSolver(40, alignment=1).plan_layout(buffers)
        self.assertEqual({b.name: b.address for b in buffers}, {"a": 0, "b": 0})

    def test_evicts_to_maximize_weight(self):
        # All three share a column in a capacity-10 cache. "big" (7) fits only
        # by itself, but the two 5s stack exactly to 10. Optimum keeps the pair
        # (weight 10) and evicts "big" (which alone would be 7).
        buffers = [
            LifetimeBoundBuffer("big", 7, 0, 1),
            LifetimeBoundBuffer("s1", 5, 0, 1),
            LifetimeBoundBuffer("s2", 5, 0, 1),
        ]
        ExhaustiveLayoutSolver(10, alignment=1).plan_layout(buffers)
        placed = {b.name for b in buffers if b.address is not None}
        self.assertEqual(placed, {"s1", "s2"})
        _assert_valid_packing(self, buffers, 10)

    def test_alignment_enforced(self):
        buffers = [
            LifetimeBoundBuffer("a", 10, 0, 4),
            LifetimeBoundBuffer("b", 10, 0, 4),
        ]
        ExhaustiveLayoutSolver(512, alignment=128).plan_layout(buffers)
        addrs = sorted(b.address for b in buffers)
        self.assertEqual(addrs, [0, 128])


class TestExhaustiveInPlace(TestCase):
    """Cases where the optimum strictly requires the in-place exception."""

    def test_inplace_needed_to_place_child(self):
        # Parent fills the whole cache for [0, 3); child lives [2, 5) and equals
        # the cache height. Without in-place the child cannot coexist with the
        # parent's last column and is evicted. In-place lets it reuse the
        # parent's address, pinning both.
        h = 512
        parent = LifetimeBoundBuffer("p", h, 0, 3)
        child = LifetimeBoundBuffer("c", h, 2, 5, in_place_parents=["p"])

        solver = ExhaustiveLayoutSolver(h, alignment=128)
        solver.plan_layout([parent, child])
        self.assertEqual(parent.address, 0)
        self.assertEqual(child.address, 0)

        # Sanity: without the in-place declaration, only one of them fits.
        p2 = LifetimeBoundBuffer("p", h, 0, 3)
        c2 = LifetimeBoundBuffer("c", h, 2, 5)
        ExhaustiveLayoutSolver(h, alignment=128).plan_layout([p2, c2])
        self.assertEqual(_placed_weight([p2, c2]), h)

    def test_inplace_chain(self):
        # A chain p -> q -> r, each handing its address to the next, plus a
        # full-height bystander 'x' that overlaps the chain's columns. The cache
        # holds two stacked full-height buffers. Only by collapsing the whole
        # chain onto one address (via in-place) can all of p, q, r AND x be
        # pinned; any non-in-place arrangement evicts at least one.
        h = 256
        p = LifetimeBoundBuffer("p", h // 2, 0, 3)
        q = LifetimeBoundBuffer("q", h // 2, 2, 5, in_place_parents=["p"])
        r = LifetimeBoundBuffer("r", h // 2, 4, 7, in_place_parents=["q"])
        x = LifetimeBoundBuffer("x", h // 2, 0, 7)

        solver = ExhaustiveLayoutSolver(h, alignment=128)
        solver.plan_layout([p, q, r, x])
        self.assertEqual(_placed_weight([p, q, r, x]), 4 * (h // 2))
        # p, q, r collapse to a single shared address; x takes the other slot.
        self.assertEqual(p.address, q.address)
        self.assertEqual(q.address, r.address)
        self.assertNotEqual(x.address, p.address)
        _assert_valid_packing(self, [p, q, r, x], h)


class TestExhaustiveOptimality(TestCase):
    """Cross-checks against brute force, the class count, and first-fit."""

    def test_class_count_equals_leaves(self):
        # With the upper-bound prune disabled and ample capacity (nothing is
        # ever evicted), the search must visit exactly prod(1 + d_i) leaves.
        random = rnd.Random(20260612)
        for trial in range(40):
            n = random.randrange(1, 8)
            buffers = _random_buffers(random, n, time_max=6)
            # node_budget=None: the class count requires the full enumeration.
            solver = ExhaustiveLayoutSolver(10**9, alignment=1, node_budget=None)
            expected = solver.class_count(buffers)

            solver.disable_upper_bound = True
            solver.plan_layout(buffers)
            self.assertEqual(
                solver.complete_orders,
                expected,
                f"trial {trial}: n={n}, buffers={buffers}",
            )

    def test_matches_brute_force(self):
        random = rnd.Random(424242)
        for trial in range(120):
            n = random.randrange(1, 8)
            buffers = _random_buffers(random, n)
            limit = random.randrange(10, 80)

            oracle = _brute_force_optimum(ExhaustiveLayoutSolver(limit, 1), buffers)

            # node_budget=None so the search is provably optimal, not truncated.
            solver = ExhaustiveLayoutSolver(limit, alignment=1, node_budget=None)
            solver.plan_layout(buffers)
            self.assertEqual(
                _placed_weight(buffers),
                oracle,
                f"trial {trial}: limit={limit}, buffers={buffers}",
            )
            _assert_valid_packing(self, buffers, limit)

    def test_matches_or_beats_first_fit(self):
        random = rnd.Random(7)
        for trial in range(80):
            n = random.randrange(1, 10)
            buffers = _random_buffers(random, n)
            limit = random.randrange(20, 120)
            alignment = random.choice([1, 8, 16])

            ff_buffers = [
                LifetimeBoundBuffer(
                    b.name,
                    b.size,
                    b.start_time,
                    b.end_time,
                    None,
                    list(b.in_place_parents),
                )
                for b in buffers
            ]
            FirstFitLayoutSolver(limit, alignment).plan_layout(ff_buffers)
            ff_weight = _placed_weight(ff_buffers)

            # node_budget=None: exact optimum is required to guarantee >= ff.
            ExhaustiveLayoutSolver(limit, alignment, node_budget=None).plan_layout(
                buffers
            )
            self.assertGreaterEqual(
                _placed_weight(buffers),
                ff_weight,
                f"trial {trial}: limit={limit}, align={alignment}, buffers={buffers}",
            )
            _assert_valid_packing(self, buffers, limit)

    def test_node_budget_aborts_but_returns_valid(self):
        random = rnd.Random(99)
        buffers = _random_buffers(random, 7)
        limit = 60
        solver = ExhaustiveLayoutSolver(limit, alignment=1, node_budget=3)
        solver.plan_layout(buffers)
        self.assertTrue(solver._aborted)
        # Even when aborted early, the warm start guarantees a valid packing.
        _assert_valid_packing(self, buffers, limit)

    def test_time_budget_scales_node_cap_with_n(self):
        # The node cap shrinks as n grows (per-node cost rises), and a longer
        # target proportionally raises it.
        one_sec = TimeBudget(1.0)
        self.assertGreater(one_sec.node_cap(10), one_sec.node_cap(20))
        self.assertGreater(one_sec.node_cap(20), one_sec.node_cap(30))
        # Twice the target is ~twice the cap (modulo integer rounding).
        self.assertAlmostEqual(
            TimeBudget(2.0).node_cap(15), 2 * one_sec.node_cap(15), delta=1
        )
        self.assertGreaterEqual(one_sec.node_cap(1000), 1)  # never zero

        # A TimeBudget resolves to a concrete per-n cap at solve time, and the
        # result is a valid packing regardless.
        random = rnd.Random(5)
        buffers = _random_buffers(random, 8)
        active_n = sum(1 for b in buffers if b.end_time > b.start_time)
        budget = TimeBudget(1.0)
        solver = ExhaustiveLayoutSolver(50, alignment=1, node_budget=budget)
        solver.plan_layout(buffers)
        self.assertEqual(solver._node_cap, budget.node_cap(active_n))
        _assert_valid_packing(self, buffers, 50)

    def test_class_count_formula(self):
        # A 3-clique (all mutually overlapping) has 3! = 6 classes; a 3-chain
        # path A-B-C has (1+1)(1+1)(1+0) = 4; an independent set has 1.
        clique = [
            LifetimeBoundBuffer("a", 1, 0, 3),
            LifetimeBoundBuffer("b", 1, 0, 3),
            LifetimeBoundBuffer("c", 1, 0, 3),
        ]
        self.assertEqual(math.prod([3, 2, 1]), 6)
        self.assertEqual(ExhaustiveLayoutSolver(10**9).class_count(clique), 6)

        path = [
            LifetimeBoundBuffer("a", 1, 0, 2),
            LifetimeBoundBuffer("b", 1, 1, 3),
            LifetimeBoundBuffer("c", 1, 2, 4),
        ]
        self.assertEqual(ExhaustiveLayoutSolver(10**9).class_count(path), 4)

        independent = [
            LifetimeBoundBuffer("a", 1, 0, 1),
            LifetimeBoundBuffer("b", 1, 1, 2),
            LifetimeBoundBuffer("c", 1, 2, 3),
        ]
        self.assertEqual(ExhaustiveLayoutSolver(10**9).class_count(independent), 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
