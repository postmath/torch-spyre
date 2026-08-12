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

"""Validation for the shared co-optimization scorer + node oracle.

The safety guards the scorer rests on:

* the ``hbm_us`` conservation test (exact integer equality) -- the strongest
  tripwire that stripping HBM from the matmul node cost neither double-counts nor
  drops bytes; and
* multiplicity / node-cost units, pointwise -> None, PSUM reductions, fixed-point
  determinism, and the additive memory + node objective.
"""

import math
import unittest
from unittest import TestCase

from torch_spyre._inductor import work_division as wd
from torch_spyre._inductor.scratchpad import cooptimization_scorer as sc
from torch_spyre._inductor.scratchpad.cooptimization_scorer import (
    MatmulNode,
    MemoryEdge,
    PointwiseNode,
    ReductionNode,
    bytes_moved,
    cohort_multiplicity,
    memory_term_fixed,
    missed_bytes,
    node_cost_fixed,
    node_cost_us,
    node_term_fixed,
    score_fixed,
    to_fixed_us,
)


def _matmul_io_edges(op: MatmulNode, served_by_lx=(False, False, False)):
    """The three matmul operands (LHS [B,M,K], RHS [wb,K,N], out [B,M,N]) as
    memory edges whose byte totals reproduce the estimator's ``bytes_total``.

    Byte counts use the *pre-split* footprints and the cohort multiplicity of each
    operand under the op's split, so summing them equals ``_matmul_hbm_us``'s
    ``bytes_total`` exactly when unsplit (multiplicity 1, no cohort penalty) -- the
    conservation identity below.
    """
    (B, b), (M, m), (N, n), (K, k) = op.b_axis, op.m_axis, op.n_axis, op.k_axis
    db = sc.dtype_bytes()
    wb = 1 if op.shared_weight else B
    # dims present in each operand (using symbolic dim ids b=0,m=1,n=2,k=3); the
    # consumer split is the full op split, and an operand missing a split dim is
    # re-read by that cohort (cohort_multiplicity).
    splits = {0: b, 1: m, 2: n, 3: k}
    lhs = bytes_moved(B * M * K * db, cohort_multiplicity(splits, {0, 1, 3}))
    rhs = bytes_moved(wb * K * N * db, cohort_multiplicity(splits, {0, 2, 3}))
    out = bytes_moved(B * M * N * db, cohort_multiplicity(splits, {0, 1, 2}))
    return [
        MemoryEdge(lhs, served_by_lx[0]),
        MemoryEdge(rhs, served_by_lx[1]),
        MemoryEdge(out, served_by_lx[2]),
    ]


class FixedPointTests(TestCase):
    def test_round_half_up_deterministic(self):
        self.assertEqual(to_fixed_us(0.0), 0)
        # 1.0 µs at 1e6 scale = 1_000_000; half rounds up.
        self.assertEqual(to_fixed_us(1.0), 1_000_000)
        self.assertEqual(to_fixed_us(0.5e-6), 1)  # 0.5 unit -> 1
        self.assertEqual(to_fixed_us(0.4e-6), 0)  # 0.4 unit -> 0
        # Determinism: repeated calls agree exactly.
        self.assertEqual(to_fixed_us(3.14159), to_fixed_us(3.14159))

    def test_rejects_negative_and_infinite(self):
        for bad in (-1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                to_fixed_us(bad)


class MultiplicityTests(TestCase):
    def test_empty_and_shared_dims_give_one(self):
        # No consumer split -> multiplicity 1.
        self.assertEqual(cohort_multiplicity({}, {0, 1}), 1)
        # Splitting only dims the tensor also has -> slicing, not re-read -> 1.
        self.assertEqual(cohort_multiplicity({0: 4, 1: 2}, {0, 1}), 1)

    def test_split_on_missing_dim_multiplies(self):
        # Consumer splits dim 1 (size 4) which the tensor lacks -> 4x re-read.
        self.assertEqual(cohort_multiplicity({1: 4}, {0}), 4)
        # Two missing split dims multiply; a shared one is excluded.
        self.assertEqual(cohort_multiplicity({1: 4, 2: 3, 0: 5}, {0}), 12)

    def test_bytes_moved_is_product(self):
        self.assertEqual(bytes_moved(100, 1), 100)
        self.assertEqual(bytes_moved(100, 3), 300)


class MemoryTermTests(TestCase):
    def test_only_missed_bytes_counted(self):
        edges = [MemoryEdge(100, False), MemoryEdge(200, True), MemoryEdge(50, False)]
        self.assertEqual(missed_bytes(edges), 150)

    def test_units_bytes_to_microseconds(self):
        # 204800 bytes at 204.8 GB/s = exactly 1 µs = 1e6 fixed units.
        bw = sc.hbm_bytes_per_us()
        self.assertEqual(bw, 204800.0)
        self.assertEqual(memory_term_fixed([MemoryEdge(int(bw), False)]), 1_000_000)

    def test_all_lx_resident_is_zero(self):
        edges = [MemoryEdge(10_000, True), MemoryEdge(20_000, True)]
        self.assertEqual(memory_term_fixed(edges), 0)


class NodeOracleTests(TestCase):
    def test_pointwise_is_none_and_zero(self):
        self.assertIsNone(node_cost_us(PointwiseNode()))
        self.assertEqual(node_cost_fixed(PointwiseNode()), 0)

    def test_matmul_strips_exactly_the_hbm_component(self):
        # The node cost is the native estimate minus its hbm_us, so adding hbm_us
        # back recovers the full estimate (float identity).
        op = MatmulNode((1, 1), (128, 2), (256, 1), (64, 1), max_cores=32)
        total = wd._matmul_split_cost(
            op.b_axis, op.m_axis, op.n_axis, op.k_axis, op.max_cores, op.shared_weight
        )
        hbm = wd._matmul_hbm_us(op.b_axis, op.m_axis, op.n_axis, op.k_axis)
        node = node_cost_us(op)
        self.assertAlmostEqual(node + hbm, total, places=9)
        self.assertGreaterEqual(node, 0.0)

    def test_matmul_infeasible_split_raises(self):
        # 4*4*4*4 = 256 cores > 32.
        op = MatmulNode((4, 4), (16, 4), (16, 4), (16, 4), max_cores=32)
        with self.assertRaises(ValueError):
            node_cost_us(op)

    def test_reduction_psum_ring(self):
        # (split-1) * elems/core * coeff; unsplit costs nothing.
        self.assertEqual(node_cost_us(ReductionNode(1, 100)), 0.0)
        coeff = wd._PSUM_PER_CORE_ELEM_US
        self.assertAlmostEqual(node_cost_us(ReductionNode(4, 50)), 3 * 50 * coeff)

    def test_unknown_op_type_raises(self):
        with self.assertRaises(TypeError):
            node_cost_us(object())


class HbmConservationTests(TestCase):
    """The memory term for an all-HBM matmul equals the estimator's hbm_us
    exactly under integer accumulation.

    The strongest tripwire on the ``hbm_us`` strip: it proves the strip neither
    double-counts nor drops bytes, and makes any residual mismatch visible rather
    than a silent bias on the split direction."""

    def test_unsplit_conservation_exact(self):
        # Whole matmul, all operands in HBM: multiplicity 1, no cohort penalty, so
        # memory_term == to_fixed_us(hbm_us) with no residual.
        for op in (
            MatmulNode((1, 1), (128, 1), (256, 1), (64, 1), max_cores=32),
            MatmulNode((4, 1), (64, 1), (128, 1), (32, 1), max_cores=32),
            MatmulNode((1, 1), (512, 1), (64, 1), (256, 1), 32, shared_weight=True),
        ):
            hbm = wd._matmul_hbm_us(
                op.b_axis, op.m_axis, op.n_axis, op.k_axis, op.shared_weight
            )
            edges = _matmul_io_edges(op)
            self.assertEqual(
                memory_term_fixed(edges), to_fixed_us(hbm), f"conservation {op}"
            )

    def test_split_surfaces_cohort_rereads_the_estimator_omits(self):
        # An M-split makes each M-cohort re-read the M-free RHS/output -- traffic
        # the memory term charges via multiplicity but that the estimator's hbm_us
        # (penalized only on N-fanout) does not see. The mismatch is *visible*
        # (our term is strictly larger), not silently folded away.
        unsplit = MatmulNode((1, 1), (128, 1), (256, 1), (64, 1), max_cores=32)
        m_split = MatmulNode((1, 1), (128, 4), (256, 1), (64, 1), max_cores=32)
        base = missed_bytes(_matmul_io_edges(unsplit))
        split = missed_bytes(_matmul_io_edges(m_split))
        self.assertGreater(split, base)
        # The estimator's hbm_us is unchanged by an M-split (fanout is N only).
        self.assertEqual(
            wd._matmul_hbm_us((1, 1), (128, 4), (256, 1), (64, 1)),
            wd._matmul_hbm_us((1, 1), (128, 1), (256, 1), (64, 1)),
        )


class ScorerTests(TestCase):
    def test_score_is_memory_plus_node(self):
        edges = [MemoryEdge(204800, False), MemoryEdge(1000, True)]
        ops = [PointwiseNode(), MatmulNode((1, 1), (64, 1), (64, 1), (64, 1), 32)]
        self.assertEqual(
            score_fixed(edges, ops), memory_term_fixed(edges) + node_term_fixed(ops)
        )

    def test_deterministic_bit_for_bit(self):
        edges = [MemoryEdge(3 * 204800 + 7, False), MemoryEdge(999, False)]
        ops = [MatmulNode((2, 1), (128, 2), (256, 1), (64, 1), 32), PointwiseNode()]
        self.assertEqual(score_fixed(edges, ops), score_fixed(edges, ops))

    def test_all_resident_no_matmul_is_zero(self):
        edges = [MemoryEdge(10_000, True)]
        self.assertEqual(score_fixed(edges, [PointwiseNode()]), 0)


class ConstantsBridgeTests(TestCase):
    """The lazy bridge to the native cost model exposes the same constants the
    estimator uses (no drift between the memory term and hbm_us)."""

    def test_constants_match_work_division(self):
        self.assertEqual(sc.hbm_bytes_per_us(), float(wd._HBM_BW_GBS) * 1000.0)
        self.assertEqual(sc.dtype_bytes(), wd._DTYPE_BYTES)
        self.assertTrue(math.isfinite(sc.hbm_bytes_per_us()))


if __name__ == "__main__":
    unittest.main()
