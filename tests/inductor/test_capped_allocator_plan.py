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
        self.assertEqual(plan.addresses, [None, None])
        self.assertEqual(plan.total_allocated_size, 0)

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
