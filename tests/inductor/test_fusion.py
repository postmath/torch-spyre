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

"""Unit tests for the bundle-grouping policy shared by the fusion pass and the
pre-scheduling bundle estimator, and for the bundle-aware feature extraction
built on it.

:func:`group_contiguous_fusable` is exercised directly: it is generic over the
item type, so the grouping rule can be checked with plain integers, without a
scheduler, a graph or a device.

:func:`extract_features_by_bundle` is exercised with bare ``ComputedBuffer``
instances (allocated via ``__new__``, with only the attributes the grouping
consults) plus a stubbed ``extract_op_features``. Building real ones needs a
compile, so these cover the bundling and skip/drop logic that this module adds,
not the per-op feature extraction it delegates to.
"""

import unittest
from unittest import mock
import torch
from torch._inductor.ir import ComputedBuffer
from torch_spyre._inductor import dump_cost_model
from torch_spyre._inductor.constants import DEVICE_NAME
from torch_spyre._inductor.fusion import estimate_bundles, group_contiguous_fusable


def _is_even(x: int) -> bool:
    """Stand-in predicate: fusable items are even, boundaries are odd.

    Any predicate would do; this one keeps the expected groupings easy to read.
    """
    return x % 2 == 0


class TestGroupContiguousFusable(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(group_contiguous_fusable([], _is_even), [])

    def test_all_fusable_is_one_run(self):
        self.assertEqual(
            group_contiguous_fusable([0, 2, 4, 6], _is_even), [[0, 2, 4, 6]]
        )

    def test_single_fusable(self):
        self.assertEqual(group_contiguous_fusable([2], _is_even), [[2]])

    def test_all_boundary_each_its_own_group(self):
        self.assertEqual(group_contiguous_fusable([1, 3, 5], _is_even), [[1], [3], [5]])

    def test_lone_boundary_comes_back_as_single_element_group(self):
        # This is what makes the refactor behaviour-preserving in
        # ``spyre_fuse_nodes``: a boundary node is a length-1 run, which
        # ``_make_fused`` returns unchanged rather than wrapping.
        self.assertEqual(group_contiguous_fusable([1], _is_even), [[1]])

    def test_alternating(self):
        self.assertEqual(
            group_contiguous_fusable([0, 1, 2, 3, 4], _is_even),
            [[0], [1], [2], [3], [4]],
        )

    def test_boundary_at_start(self):
        self.assertEqual(
            group_contiguous_fusable([1, 0, 2, 4], _is_even), [[1], [0, 2, 4]]
        )

    def test_boundary_at_end(self):
        self.assertEqual(
            group_contiguous_fusable([0, 2, 4, 1], _is_even), [[0, 2, 4], [1]]
        )

    def test_boundary_at_both_ends(self):
        self.assertEqual(
            group_contiguous_fusable([1, 0, 2, 3], _is_even),
            [[1], [0, 2], [3]],
        )

    def test_runs_are_maximal_and_order_is_preserved(self):
        items = [0, 2, 1, 4, 6, 8, 3, 5, 10]
        self.assertEqual(
            group_contiguous_fusable(items, _is_even),
            [[0, 2], [1], [4, 6, 8], [3], [5], [10]],
        )
        # Every item appears exactly once, in the original order.
        groups = group_contiguous_fusable(items, _is_even)
        self.assertEqual([item for group in groups for item in group], items)


def _buffer(tag: str, device: str) -> ComputedBuffer:
    """A bare ``ComputedBuffer`` carrying only what the grouping consults.

    ``extract_op_features`` is stubbed out in these tests, so the buffer needs no
    layout or loop body -- just to satisfy the ``isinstance`` check and answer
    ``get_device()``.  ``tag`` identifies it in assertions.
    """
    op = ComputedBuffer.__new__(ComputedBuffer)
    op.tag = tag
    op.get_device = lambda: torch.device(device)
    return op


def _spyre(tag: str) -> ComputedBuffer:
    return _buffer(tag, DEVICE_NAME)


def _cpu(tag: str) -> ComputedBuffer:
    """A ComputedBuffer off-device: a bundle boundary that is still modellable."""
    return _buffer(tag, "cpu")


class _Extern:
    """Stands in for an extern/fallback op: not a ComputedBuffer at all."""

    def __init__(self, tag: str):
        self.tag = tag


def _tags(bundles) -> list[list[str]]:
    return [[f.name for f in bundle] for bundle in bundles]


class _Feat:
    """Stand-in for OpFeatures; only ``name`` is read back by the assertions."""

    def __init__(self, name: str):
        self.name = name


def _features_of(op) -> _Feat:
    return _Feat(op.tag)


def _unmodellable(_op):
    raise ValueError("cannot model this op")


class TestExtractFeaturesByBundle(unittest.TestCase):
    def test_bundles_align_with_estimate_bundles(self):
        ops = [_spyre("a"), _spyre("b"), _Extern("x"), _spyre("c")]
        with mock.patch.object(dump_cost_model, "extract_op_features", _features_of):
            bundles = dump_cost_model.extract_features_by_bundle(ops)

        # The extern op forms its own group, which yields no features and is
        # dropped; the two Spyre runs survive with their membership intact.
        self.assertEqual(_tags(bundles), [["a", "b"], ["c"]])

        # And that grouping is exactly what estimate_bundles reports, modulo the
        # groups dropped for having no modellable ops.
        estimated = [
            [op.tag for op in group if isinstance(op, ComputedBuffer)]
            for group in estimate_bundles(ops)
        ]
        self.assertEqual([g for g in estimated if g], _tags(bundles))

    def test_off_device_buffer_is_its_own_bundle(self):
        # A CPU ComputedBuffer is a boundary, but unlike an extern op it still
        # extracts, so it survives as a single-op bundle between the two runs.
        ops = [_spyre("a"), _cpu("m"), _spyre("b")]
        with mock.patch.object(dump_cost_model, "extract_op_features", _features_of):
            bundles = dump_cost_model.extract_features_by_bundle(ops)
        self.assertEqual(_tags(bundles), [["a"], ["m"], ["b"]])

    def test_empty_group_is_dropped(self):
        # "b" is the only op in its group and cannot be modelled, so that group
        # disappears rather than coming back as an empty bundle.
        ops = [_spyre("a"), _cpu("b"), _spyre("c")]

        def _fail_on_b(op):
            if op.tag == "b":
                raise ValueError("cannot model this op")
            return _Feat(op.tag)

        with mock.patch.object(dump_cost_model, "extract_op_features", _fail_on_b):
            bundles = dump_cost_model.extract_features_by_bundle(ops)
        self.assertEqual(_tags(bundles), [["a"], ["c"]])
        self.assertTrue(all(bundle for bundle in bundles))

    def test_unmodellable_op_is_skipped_within_a_surviving_bundle(self):
        ops = [_spyre("a"), _spyre("b"), _spyre("c")]

        def _fail_on_b(op):
            if op.tag == "b":
                raise ValueError("cannot model this op")
            return _Feat(op.tag)

        with mock.patch.object(dump_cost_model, "extract_op_features", _fail_on_b):
            bundles = dump_cost_model.extract_features_by_bundle(ops)
        self.assertEqual(_tags(bundles), [["a", "c"]])

    def test_all_unmodellable_gives_no_bundles(self):
        ops = [_spyre("a"), _spyre("b")]
        with mock.patch.object(dump_cost_model, "extract_op_features", _unmodellable):
            self.assertEqual(dump_cost_model.extract_features_by_bundle(ops), [])

    def test_no_operations(self):
        self.assertEqual(dump_cost_model.extract_features_by_bundle([]), [])

    def test_non_computed_buffer_ops_are_skipped(self):
        ops = [_Extern("x"), _Extern("y")]
        with mock.patch.object(dump_cost_model, "extract_op_features", _features_of):
            self.assertEqual(dump_cost_model.extract_features_by_bundle(ops), [])


class TestPredictByBundle(unittest.TestCase):
    def test_sums_one_prediction_per_bundle(self):
        ops = [_spyre("a"), _spyre("b"), _Extern("x"), _spyre("c")]
        scored = []

        def _predict(bundle):
            scored.append([f.name for f in bundle])
            return 10.0 * len(bundle)

        with (
            mock.patch.object(dump_cost_model, "extract_op_features", _features_of),
            mock.patch.object(dump_cost_model, "predict_ops", _predict),
        ):
            total = dump_cost_model.predict_by_bundle(ops)

        # Scored per bundle, not once over the flattened graph.
        self.assertEqual(scored, [["a", "b"], ["c"]])
        self.assertEqual(total, 30.0)

    def test_no_bundles_predicts_zero(self):
        self.assertEqual(dump_cost_model.predict_by_bundle([]), 0)


if __name__ == "__main__":
    unittest.main()
