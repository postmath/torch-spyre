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

from unittest import TestCase

from torch_spyre._inductor.scratchpad.plan_solver import (
    CappedAllocatorPlan,
    LifetimeBoundBuffer,
    ReferenceCappedAllocatorPlan,
)

ALIGNMENT = 128


def _buf(name, size, start, end, in_place_parents=None):
    return LifetimeBoundBuffer(
        name=name,
        size=size,
        start_time=start,
        end_time=end,
        in_place_parents=in_place_parents or [],
    )


# Both concrete plans share CappedAllocatorPlanBase, so the Step 1 skeleton
# behaviour (field setup, helpers, finalize) is identical and tested for both.
class SkeletonTestsMixin:
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
        # Unplaced.
        plan.addresses[0] = None
        self.assertFalse(plan._is_fully_allocated(0))
        # Fits exactly at the boundary.
        plan.addresses[0] = 36
        self.assertTrue(plan._is_fully_allocated(0))
        # One byte over capacity.
        plan.addresses[0] = 37
        self.assertFalse(plan._is_fully_allocated(0))

    def test_total_size_accessor(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=128)
        plan.total_allocated_size = 42
        self.assertEqual(plan.total_size(), 42)

    def test_finalize_writes_back_only_fully_allocated(self):
        buffers = [
            _buf("fits", 64, 0, 1),
            _buf("over_cap", 64, 0, 1),
            _buf("unplaced", 64, 0, 1),
        ]
        plan = self.make_plan(buffers, [0, 1, 2], capacity=128)
        plan.addresses = [0, 100, None]  # over_cap: 100 + 64 = 164 > 128

        plan.finalize()

        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)
        self.assertIsNone(buffers[2].address)


class ReferencePlanSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = ReferenceCappedAllocatorPlan


class CappedAllocatorPlanSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = CappedAllocatorPlan


def _addr(plan, name):
    return plan.addresses[plan._name_to_idx[name]]


class ReferencePlacementTests(TestCase):
    """Step 2: O(n^2) placement in ReferenceCappedAllocatorPlan."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return ReferenceCappedAllocatorPlan(buffers, permutation, capacity, alignment)

    def test_disjoint_lifetimes_all_at_zero(self):
        # No two buffers are ever alive together, so each reuses address 0.
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 2, 3), _buf("c", 64, 4, 5)]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual([_addr(plan, n) for n in "abc"], [0, 0, 0])
        self.assertEqual(plan.total_size(), 192)

    def test_overlapping_lifetimes_stack(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)  # stacked on top of a
        self.assertEqual(plan.total_size(), 114)

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
        child = _buf("c", 64, 5, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 0)  # reuses parent's address
        self.assertEqual(plan.total_size(), 192)

    def test_in_place_parent_placed_after_child_reuses_address(self):
        # Symmetric case: child allocated first, parent reuses its address.
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 5, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [1, 0])  # child first
        self.assertEqual(_addr(plan, "c"), 0)
        self.assertEqual(_addr(plan, "p"), 0)  # parent reuses child's address

    def test_in_place_blocked_when_child_larger_than_parent(self):
        parent = _buf("p", 64, 0, 5)
        child = _buf("c", 128, 5, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 64)  # cannot reuse; stacks on top

    def test_in_place_blocked_by_intruding_buffer(self):
        # The collision case from the design discussion: Z coexists with the
        # child but not the parent, so reusing the parent's address would
        # overlap Z. Placement must fall back to stacking.
        parent = _buf("p", 50, 0, 5)
        child = _buf("c", 30, 5, 10, in_place_parents=["p"])
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
        self.assertEqual(plan.total_size(), 64)  # only a counts

    def test_finalize_after_build(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        plan.finalize()
        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)  # over capacity, not committed
