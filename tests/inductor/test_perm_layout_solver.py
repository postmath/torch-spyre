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

"""Tests for the capacity-bounded allocation plans."""

import random
from unittest import TestCase
from typing import TYPE_CHECKING

from torch_spyre._inductor.scratchpad.plan_solver import (
    PermutationBasedLayoutSolver,
    LifetimeBoundBuffer,
    ReferencePermutationBasedLayoutSolver,
)

ALIGNMENT = 128


def _random_buffers(rng, n, horizon=12, max_size=200):
    """Generate ``n`` random buffers, occasionally wiring in-place pairs.

    Lifetimes are half-open ``[start, end)`` and non-empty (end > start).
    """
    buffers = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        buffers.append(_buf(f"b{i}", size, start, end))
    # Turn a few buffers into in-place children of an earlier buffer: the child
    # starts at the parent's last live tick (parent.end - 1, so that
    # parent.end == child.start + 1) and clamps its size to fit.
    for child_i in range(1, n):
        if rng.random() < 0.25:
            parent_i = rng.randrange(child_i)
            parent = buffers[parent_i]
            child = buffers[child_i]
            child.start_time = parent.end_time - 1
            child.end_time = max(child.end_time, parent.end_time)
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def _oracle_graph(plan):
    """Independent brute-force neighbour graph from the final addresses.

    Implemented differently from the production sweep: for every ordered pair
    (b, c) with b entirely below c, b is a direct below-neighbour iff there is
    some shared tick at which no third buffer d sits strictly between them
    (``b.top <= d.addr`` and ``d.top <= c.addr``). In-place partners sharing an
    address are never "entirely below" each other, so the explicit edges added
    first are the only links between them.
    """
    n = len(plan.buffers)
    addr = plan.addresses
    bufs = plan.buffers
    below = {i: set() for i in range(n)}
    above = {i: set() for i in range(n)}
    for reuser, reused in plan.inplace_reuse.items():
        below[reuser].add(reused)
        above[reused].add(reuser)

    def top(i):
        return addr[i] + bufs[i].size

    for c in range(n):
        for b in range(n):
            if b == c or top(b) > addr[c]:
                continue
            lo = max(bufs[b].start_time, bufs[c].start_time)
            hi = min(bufs[b].end_time, bufs[c].end_time)
            for t in range(lo, hi):
                between = any(
                    d not in (b, c)
                    and bufs[d].start_time <= t < bufs[d].end_time
                    and top(b) <= addr[d]
                    and top(d) <= addr[c]
                    for d in range(n)
                )
                if not between:
                    below[c].add(b)
                    above[b].add(c)
                    break
    return below, above


def _buf(name, size, start, end, in_place_parents=None):
    return LifetimeBoundBuffer(
        name=name,
        size=size,
        start_time=start,
        end_time=end,
        in_place_parents=in_place_parents or [],
    )


# Both concrete plans share PermutationBasedLayoutSolverBase, so the Step 1 skeleton
# behaviour (field setup, helpers, finalize) is identical and tested for both.
if TYPE_CHECKING:
    MixinBase = TestCase
else:
    MixinBase = object


class SkeletonTestsMixin(MixinBase):
    plan_class: type = None  # type: ignore[assignment]

    def make_plan(self, buffers, permutation, capacity, alignment=ALIGNMENT):
        return self.plan_class(buffers, permutation, capacity, alignment)

    def test_init_stores_fields(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 64, 1, 3)]
        plan = self.make_plan(buffers, [1, 0], capacity=256, alignment=ALIGNMENT)

        self.assertIs(plan.buffers, buffers)
        self.assertEqual(plan.permutation, [1, 0])
        self.assertEqual(plan.capacity, 256)
        self.assertEqual(plan.alignment, ALIGNMENT)
        self.assertEqual(plan._name_to_idx, {"a": 0, "b": 1})
        # addresses has one slot per buffer; its contents depend on _build.
        self.assertEqual(len(plan.addresses), 2)

    def test_permutation_is_copied(self):
        buffers = [_buf("a", 64, 0, 1)]
        perm = [0]
        plan = self.make_plan(buffers, perm, capacity=128)
        perm.append(99)
        self.assertEqual(plan.permutation, [0])

    def test_invalid_permutation_rejected(self):
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 0, 1)]
        with self.assertRaises(AssertionError):
            self.make_plan(buffers, [0, 0], capacity=128)
        with self.assertRaises(AssertionError):
            self.make_plan(buffers, [0], capacity=128)

    def test_align_up(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=128, alignment=128)
        self.assertEqual(plan._align_up(0), 0)
        self.assertEqual(plan._align_up(1), 128)
        self.assertEqual(plan._align_up(128), 128)
        self.assertEqual(plan._align_up(129), 256)

    def test_top(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=256)
        plan.addresses[0] = 128
        self.assertEqual(plan._top(0), 192)

    def test_is_fully_allocated(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=100)
        # Fits exactly at the boundary.
        plan.addresses[0] = 36
        self.assertTrue(plan._is_fully_allocated(0))
        # One byte over capacity.
        plan.addresses[0] = 37
        self.assertFalse(plan._is_fully_allocated(0))

    def test_quality_accessor(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=128)
        plan.total_allocated_size = 42
        plan.total_allocated_count = 3
        self.assertEqual(plan.quality(), 42)
        self.assertEqual(plan.count_allocated(), 3)

    def test_finalize_writes_back_only_fully_allocated(self):
        buffers = [
            _buf("fits", 64, 0, 1),
            _buf("over_cap", 64, 0, 1),
            _buf("also_over", 64, 0, 1),
        ]
        plan = self.make_plan(buffers, [0, 1, 2], capacity=128)
        # Every buffer has a notional address; only those fitting below capacity
        # are committed. 100 + 64 = 164 > 128 and 200 + 64 = 264 > 128.
        plan.addresses = [0, 100, 200]

        plan.finalize()

        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)
        self.assertIsNone(buffers[2].address)


class ReferenceSolverSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = ReferencePermutationBasedLayoutSolver


class PermutationBasedLayoutSolverSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = PermutationBasedLayoutSolver


def _addr(plan, name):
    return plan.addresses[plan._name_to_idx[name]]


class ReferencePlacementTests(TestCase):
    """Step 2: O(n^2) placement in ReferencePermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return ReferencePermutationBasedLayoutSolver(
            buffers, permutation, capacity, alignment
        )

    def test_disjoint_lifetimes_all_at_zero(self):
        # No two buffers are ever alive together, so each reuses address 0.
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 2, 3), _buf("c", 64, 4, 5)]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual([_addr(plan, n) for n in "abc"], [0, 0, 0])
        self.assertEqual(plan.quality(), 192)

    def test_overlapping_lifetimes_stack(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)  # stacked on top of a
        self.assertEqual(plan.quality(), 114)

    def test_permutation_order_changes_layout(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [1, 0])  # b placed first
        self.assertEqual(_addr(plan, "b"), 0)
        self.assertEqual(_addr(plan, "a"), 50)  # a stacked on top of b

    def test_alignment_rounds_up(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], alignment=128)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 128)  # ceil(64/128)*128

    def test_freed_low_space_reused_by_later_disjoint_buffer(self):
        # a dies before c starts; c does not overlap a, so it drops back to 0
        # even though b (overlapping both) sits above.
        buffers = [
            _buf("a", 64, 0, 2),
            _buf("b", 64, 1, 5),
            _buf("c", 64, 3, 5),
        ]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)
        # c overlaps b (not a) -> stacks only on b.
        self.assertEqual(_addr(plan, "c"), 128)

    def test_in_place_child_reuses_parent_address(self):
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 0)  # reuses parent's address
        self.assertEqual(plan.quality(), 192)

    def test_in_place_parent_placed_after_child_reuses_address(self):
        # Symmetric case: child allocated first, parent reuses its address.
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [1, 0])  # child first
        self.assertEqual(_addr(plan, "c"), 0)
        self.assertEqual(_addr(plan, "p"), 0)  # parent reuses child's address

    def test_in_place_blocked_when_child_larger_than_parent(self):
        parent = _buf("p", 64, 0, 5)
        child = _buf("c", 128, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 64)  # cannot reuse; stacks on top

    def test_in_place_blocked_by_intruding_buffer(self):
        # The collision case from the design discussion: Z coexists with the
        # child but not the parent, so reusing the parent's address would
        # overlap Z. Placement must fall back to stacking.
        parent = _buf("p", 50, 0, 5)
        child = _buf("c", 30, 4, 10, in_place_parents=["p"])
        z = _buf("z", 20, 6, 10)
        plan = self.plan([parent, child, z], [0, 2, 1])  # order: p, z, c
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "z"), 0)  # z does not overlap p
        # c overlaps both p (top 50) and z (top 20); p is topmost and is the
        # in-place partner, but reusing addr 0 would hit z -> stack at 50.
        self.assertEqual(_addr(plan, "c"), 50)

    def test_over_capacity_buffer_not_counted_but_still_addressed(self):
        # b stacks above a and crosses the capacity line: it keeps an address
        # (so later buffers stack on it) but is excluded from total_size.
        buffers = [_buf("a", 64, 0, 3), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)  # 64 + 64 = 128 > 100
        self.assertEqual(plan.quality(), 64)  # only a counts

    def test_finalize_after_build(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        plan.finalize()
        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)  # over capacity, not committed


class NeighborGraphTests(TestCase):
    """Neighbour-graph construction in PermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def _below(self, plan, name):
        return {
            plan.buffers[i].name for i in plan.below_neighbors[plan._name_to_idx[name]]
        }

    def _above(self, plan, name):
        return {
            plan.buffers[i].name for i in plan.above_neighbors[plan._name_to_idx[name]]
        }

    def test_simple_stack_edges(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])  # a@0, b@64
        self.assertEqual(self._below(plan, "b"), {"a"})
        self.assertEqual(self._above(plan, "a"), {"b"})
        self.assertEqual(self._below(plan, "a"), set())
        self.assertEqual(self._above(plan, "b"), set())

    def test_air_gap_neighbor(self):
        # low spans the whole range; tall is briefly stacked on it and forces
        # high up to 320. After tall dies, only low sits beneath high, with an
        # air gap in (64, 320). low must still be a below-neighbour of high.
        low = _buf("low", 64, 0, 10)
        tall = _buf("tall", 256, 0, 3)
        high = _buf("high", 64, 0, 10)
        plan = self.plan([low, tall, high], [0, 1, 2])
        self.assertEqual(_addr(plan, "low"), 0)
        self.assertEqual(_addr(plan, "tall"), 64)
        self.assertEqual(_addr(plan, "high"), 320)
        self.assertEqual(self._below(plan, "high"), {"low", "tall"})

    def test_in_place_edge_is_explicit(self):
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])  # c reuses p's address (0)
        self.assertEqual(self._below(plan, "c"), {"p"})
        self.assertEqual(self._above(plan, "p"), {"c"})

    # --- randomized differential checks ------------------------------------

    def _cases(self, seeds=300, max_n=8):
        for seed in range(seeds):
            rng = random.Random(seed)
            n = rng.randint(1, max_n)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            capacity = rng.choice([200, 600, 10_000])
            alignment = rng.choice([1, 64, 128])
            yield seed, buffers, perm, capacity, alignment

    def test_addresses_match_reference(self):
        for seed, buffers, perm, cap, align in self._cases():
            ref = ReferencePermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            self.assertEqual(fast.addresses, ref.addresses, f"seed={seed}")
            self.assertEqual(fast.quality(), ref.quality(), f"seed={seed}")

    def test_graph_matches_oracle(self):
        for seed, buffers, perm, cap, align in self._cases():
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            below, above = _oracle_graph(fast)
            self.assertEqual(fast.below_neighbors, below, f"seed={seed}")
            self.assertEqual(fast.above_neighbors, above, f"seed={seed}")

    def test_graph_is_symmetric(self):
        for seed, buffers, perm, cap, align in self._cases():
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            for c, lowers in fast.below_neighbors.items():
                for b in lowers:
                    self.assertIn(c, fast.above_neighbors[b], f"seed={seed}")
            for b, uppers in fast.above_neighbors.items():
                for c in uppers:
                    self.assertIn(b, fast.below_neighbors[c], f"seed={seed}")

    def test_below_neighbors_determine_address(self):
        # The invariant Step 4 relies on: each buffer's address is recovered
        # from its below-neighbours alone, identically to the full build.
        for seed, buffers, perm, cap, align in self._cases():
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            for idx in range(len(buffers)):
                addr, _ = fast._placement_decision(
                    idx, sorted(fast.below_neighbors[idx])
                )
                self.assertEqual(addr, fast.addresses[idx], f"seed={seed} idx={idx}")


class SwapTests(TestCase):
    """Incremental swap in PermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_overlapping_swap_relayouts(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 0, 2)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 64])
        delta = plan.swap(0)  # -> [b, a]
        self.assertEqual([_addr(plan, "b"), _addr(plan, "a")], [0, 50])
        self.assertEqual(delta, 0)  # both still fit

    def test_non_overlapping_swap_is_noop(self):
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 2, 3)]
        plan = self.plan(buffers, [0, 1])
        before = list(plan.addresses)
        delta = plan.swap(0)
        self.assertEqual(delta, 0)
        self.assertEqual(plan.addresses, before)
        self.assertEqual(plan.permutation, [1, 0])

    def test_swap_changes_total_size(self):
        # Only one of the two can fit fully below capacity; swapping which one
        # is placed first changes the total.
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual(plan.quality(), 30)  # a@0 fits, b@30 (->120) does not
        delta = plan.swap(0)  # -> [b, a]: b@0 fits, a@90 (->120) does not
        self.assertEqual(plan.quality(), 90)
        self.assertEqual(delta, 60)

    def test_swap_back_restores(self):
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        d1 = plan.swap(0)
        d2 = plan.swap(0)
        self.assertEqual(d1 + d2, 0)
        self.assertEqual(plan.quality(), 30)
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 30])

    def test_finalize_after_swaps_end_to_end(self):
        # Build, optimize via swaps, then commit: only buffers that fit below
        # capacity get an address written back.
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        plan.swap(0)  # -> [b, a]: b@0 fits, a@90 (-> 120) does not
        plan.finalize()
        self.assertEqual(buffers[1].address, 0)  # b committed
        self.assertIsNone(buffers[0].address)  # a over capacity, dropped

    def test_random_swap_sequences_match_reference(self):
        for seed in range(3000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            for step in range(rng.randint(1, 2 * n)):
                i = rng.randrange(n - 1)
                before = fast.quality()
                delta = fast.swap(i)
                tag = f"seed={seed} step={step}"

                # Ground truth: a fresh reference build of the new permutation.
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.addresses, ref.addresses, tag)
                self.assertEqual(fast.quality(), ref.quality(), tag)
                self.assertEqual(delta, fast.quality() - before, tag)

                # The incrementally maintained graph matches a from-scratch
                # rebuild of the same permutation, exactly.
                rebuilt = PermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.below_neighbors, rebuilt.below_neighbors, tag)
                self.assertEqual(fast.above_neighbors, rebuilt.above_neighbors, tag)
                self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)


class RotateTests(TestCase):
    """rotate(i, j) and the single-element sweep it enables."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_rotate_noop(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 0, 2)]
        plan = self.plan(buffers, [0, 1])
        before = list(plan.addresses)
        self.assertEqual(plan.rotate(1, 1), 0)
        self.assertEqual(plan.permutation, [0, 1])
        self.assertEqual(plan.addresses, before)

    def test_rotate_moves_element(self):
        # Three mutually overlapping buffers; move the first to the end.
        buffers = [_buf("a", 10, 0, 3), _buf("b", 20, 0, 3), _buf("c", 30, 0, 3)]
        plan = self.plan(buffers, [0, 1, 2])  # a@0, b@10, c@30
        plan.rotate(0, 2)  # -> [b, c, a]: b@0, c@20, a@50
        self.assertEqual(plan.permutation, [1, 2, 0])
        self.assertEqual([_addr(plan, n) for n in "abc"], [50, 0, 20])

    def test_random_rotations_match_reference(self):
        for seed in range(3000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            for step in range(rng.randint(1, 2 * n)):
                i, j = rng.randrange(n), rng.randrange(n)
                before = fast.quality()
                delta = fast.rotate(i, j)
                tag = f"seed={seed} step={step} i={i} j={j}"

                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.addresses, ref.addresses, tag)
                self.assertEqual(fast.quality(), ref.quality(), tag)
                self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
                self.assertEqual(delta, fast.quality() - before, tag)

                rebuilt = PermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.below_neighbors, rebuilt.below_neighbors, tag)
                self.assertEqual(fast.above_neighbors, rebuilt.above_neighbors, tag)
                self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)

    def test_single_element_sweep_matches_reference(self):
        # Sweep one element across every position (rotate it to 0, then bubble
        # it right), reading quality() at each stop. Each stop must match a
        # fresh build of that permutation, and a round trip must restore the
        # original state exactly -- the contract the annealing sweep relies on.
        for seed in range(500):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            orig_perm = list(fast.permutation)
            orig_addr = list(fast.addresses)
            i = rng.randrange(n)
            x = orig_perm[i]
            others = [b for b in orig_perm if b != x]

            qualities = {}
            fast.rotate(i, 0)  # x to the front
            qualities[0] = fast.quality()
            for p in range(1, n):
                fast.swap(p - 1)  # bubble x from p-1 to p
                qualities[p] = fast.quality()

            # Every recorded objective matches a fresh build of "x inserted at p".
            for p in range(n):
                test_perm = others[:p] + [x] + others[p:]
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, test_perm, cap, align
                )
                self.assertEqual(qualities[p], ref.quality(), f"seed={seed} p={p}")

            # Round trip restores the exact original state (no hysteresis).
            fast.rotate(n - 1, i)
            self.assertEqual(fast.permutation, orig_perm, f"seed={seed}")
            self.assertEqual(fast.addresses, orig_addr, f"seed={seed}")
            rebuilt = PermutationBasedLayoutSolver(buffers, orig_perm, cap, align)
            self.assertEqual(fast.below_neighbors, rebuilt.below_neighbors, f"{seed}")
            self.assertEqual(fast.above_neighbors, rebuilt.above_neighbors, f"{seed}")
            self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, f"{seed}")


class CopyTests(TestCase):
    """copy() makes an independent layout snapshot sharing static structures."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_static_shared_dynamic_independent(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 0, 3), _buf("c", 40, 1, 3)]
        plan = self.plan(buffers, [0, 1, 2])
        clone = plan.copy()
        # Static structures are shared by reference.
        self.assertIs(clone.buffers, plan.buffers)
        self.assertIs(clone.overlaps, plan.overlaps)
        self.assertIs(clone.inplace_partners, plan.inplace_partners)
        self.assertIs(clone._name_to_idx, plan._name_to_idx)
        # Dynamic state is equal but independent.
        self.assertEqual(clone.addresses, plan.addresses)
        self.assertEqual(clone.below_neighbors, plan.below_neighbors)
        self.assertIsNot(clone.permutation, plan.permutation)
        self.assertIsNot(clone.below_neighbors, plan.below_neighbors)
        self.assertIsNot(clone.below_neighbors[0], plan.below_neighbors[0])

    def test_mutating_copy_leaves_original_intact(self):
        for seed in range(2000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            plan = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            orig_perm = list(plan.permutation)
            orig_addr = list(plan.addresses)
            orig_below = {k: set(v) for k, v in plan.below_neighbors.items()}
            orig_quality = plan.quality()

            clone = plan.copy()
            for _ in range(rng.randint(1, 2 * n)):
                clone.swap(rng.randrange(n - 1))

            # Original is untouched by mutations on the clone.
            self.assertEqual(plan.permutation, orig_perm, seed)
            self.assertEqual(plan.addresses, orig_addr, seed)
            self.assertEqual(plan.below_neighbors, orig_below, seed)
            self.assertEqual(plan.quality(), orig_quality, seed)

            # The mutated clone is a valid plan: matches a fresh build.
            rebuilt = PermutationBasedLayoutSolver(
                buffers, list(clone.permutation), cap, align
            )
            self.assertEqual(clone.addresses, rebuilt.addresses, seed)
            self.assertEqual(clone.quality(), rebuilt.quality(), seed)
            self.assertEqual(clone.below_neighbors, rebuilt.below_neighbors, seed)
            self.assertEqual(clone.above_neighbors, rebuilt.above_neighbors, seed)
