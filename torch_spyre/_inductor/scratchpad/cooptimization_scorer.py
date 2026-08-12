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

"""Shared cost service for the joint work-division + LX-layout SA optimizer.

This is the single authoritative scorer: every engine and the SA Metropolis test
read the same objective, so cross-engine comparison is honest. The objective is
richer than the substrate's own HBM-traffic-only cost -- it adds a node/compute
term and the split-projection cohort multiplicity:

    Score = memory_term + node_term        (time units, integer fixed-point)

* **Memory term**: the time to move every tensor byte *not* served from LX. Per
  consuming edge it is ``bytes_moved(tensor, consumer_split)`` times a residency
  factor (0 on an LX hit, 1 on a miss), where ``bytes_moved`` folds in the
  division-dependent cohort ``multiplicity`` -- the count of consumer cores that
  re-read a tensor lacking their split dim. Converted to time by the HBM
  bandwidth constant.
* **Node term**: a context-free per-op oracle. Cheap memory-bound pointwise ops
  cost ``None`` (0); matmuls/bmms route to the native
  ``_matmul_split_cost`` estimator **with its ``hbm_us`` component stripped** (that
  traffic is already in the memory term -- no double-counting); cross-core
  reductions cost the PSUM ring overhead.

**Determinism / integer accumulation**: microsecond quantities are mapped to a
fixed-point integer scale by a *single* deterministic rounding step, so the
accumulated score is bit-for-bit reproducible with no float non-determinism.

**Two multiplicities -- do not conflate**: the memory-term ``multiplicity`` here
is the *division-dependent* cohort formula and belongs to the memory term only.
The packer's ``buffer_quality`` weight (``len(uses) + write_bonus``) is a *static*
op-access count and is a separate number; applying this formula to it would
double-count.

Substrate isolation: this module depends only on the native matmul estimator in
``work_division`` (lazily, so the memory-term / multiplicity core imports without
torch) -- never on the co-optimization substrate. Turning ``CoreDivisionBuffer``s
into memory edges / node ops belongs to the engine (:mod:`sa_cooptimizer`), which
keeps this module reusable by any engine and testable without one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

# --- fixed-point scale ------------------------------------------------------ #
# Microseconds are the universal currency; every µs quantity is converted to an
# integer on this scale by exactly one rounding step, so accumulation is pure
# integer. 1e6 gives picosecond resolution -- ample for the smallest memory
# terms -- while Python's arbitrary-precision ints keep large sums exact.
US_FIXED_POINT_SCALE = 1_000_000


def to_fixed_us(us: float) -> int:
    """Map a non-negative microsecond quantity to the fixed-point integer scale
    with a single deterministic round-half-up step.

    Round-half-up on non-negative inputs is order-independent and platform-stable
    (no banker's rounding), which is what the determinism guarantee needs. Node
    and memory costs are always ``>= 0``; an infinite cost (an infeasible split)
    is a caller error, flagged rather than silently mapped.
    """
    if not math.isfinite(us) or us < 0.0:
        raise ValueError(f"cost must be finite and non-negative, got {us!r}")
    return int(us * US_FIXED_POINT_SCALE + 0.5)


# --- lazy bridge to the native cost model (keeps the core torch-free) ------- #
_WD_CACHE: dict[str, object] = {}


def _wd():
    """Import and cache ``work_division`` on first use.

    Deferred so the memory-term / multiplicity / fixed-point helpers -- which need
    none of it -- import without pulling in torch. Only the node oracle's
    matmul/reduction paths and the physical constants touch it.
    """
    mod = _WD_CACHE.get("mod")
    if mod is None:
        from torch_spyre._inductor import work_division as mod  # noqa: PLC0415

        _WD_CACHE["mod"] = mod
    return mod


def hbm_bytes_per_us() -> float:
    """HBM bandwidth as bytes per microsecond, sourced from the native cost model
    (``_HBM_BW_GBS`` GB/s x 1000 = bytes/µs) so the memory term and the matmul
    estimator's ``hbm_us`` use the identical constant -- the conservation check in
    ``tests/inductor/test_cooptimization_scorer.py`` relies on this."""
    return float(_wd()._HBM_BW_GBS) * 1000.0


def dtype_bytes() -> int:
    """Element size in bytes (fp16), from the native cost model."""
    return int(_wd()._DTYPE_BYTES)


# ===========================================================================
# Memory term
# ===========================================================================


def cohort_multiplicity(
    consumer_splits: dict[int, int], tensor_dims: Iterable[int]
) -> int:
    """Access multiplicity of a tensor read by a consumer op:

        multiplicity = prod( S_O[d] for d in dims(O) if d not in dims(T) )

    ``consumer_splits`` maps the consumer op's split dims (the stride/coeff keys
    of a ``CoreDivision``'s ``output_splits`` / ``reduction_splits``) to their
    per-dim core counts; ``tensor_dims`` are the dim keys present in the tensor
    ``T``. A consumer split dim absent from ``T`` means each core along it re-reads
    the whole tensor, so its count multiplies the traffic. Splits on shared dims
    slice the tensor (no re-read) and do not. The empty product is 1.
    """
    dims = set(tensor_dims)
    return math.prod(count for dim, count in consumer_splits.items() if dim not in dims)


def bytes_moved(tensor_bytes: int, multiplicity: int) -> int:
    """Bytes transferred for one tensor read, cohort re-reads folded in.

    Decoupled geometry: this is a pure function of the tensor footprint and the
    consumer split, independent of residency. The memory term multiplies it by
    the 0/1 residency factor.
    """
    return tensor_bytes * multiplicity


@dataclass(frozen=True)
class MemoryEdge:
    """One tensor read by one consuming op. ``moved`` is :func:`bytes_moved`
    (footprint x cohort multiplicity); ``served_by_lx`` is True when the tensor is
    LX-resident *and* tiling-compatible on this edge, in which case it costs no
    HBM traffic. A tensor that misses LX -- via an incompatible tiling edge or an
    allocation spill -- is counted once here regardless of the cause."""

    moved: int
    served_by_lx: bool


def missed_bytes(edges: Iterable[MemoryEdge]) -> int:
    """Total bytes that hit HBM: the sum of ``moved`` over edges not served by LX.
    Pure integer accumulation."""
    return sum(e.moved for e in edges if not e.served_by_lx)


def memory_term_fixed(edges: Iterable[MemoryEdge]) -> int:
    """The memory term in fixed-point time units: HBM-missing bytes converted to
    microseconds by the bandwidth constant, with a single rounding step."""
    return to_fixed_us(missed_bytes(edges) / hbm_bytes_per_us())


# ===========================================================================
# Node oracle
# ===========================================================================
#
# ``node_cost(op, own_division) -> int | None`` as a context-free dispatch on op
# kind. The division is carried inside each op descriptor's ``(size, split)``
# axes, so the oracle needs nothing but the op itself. That rests on ops running
# sequentially, each fanning out over up to SENCORES at a time, so there are no
# concurrent co-tenants to model; if the backend ever pipelines ops against each
# other, cores become a shared time-multiplexed budget and this oracle needs a
# rewrite rather than a tweak.


@dataclass(frozen=True)
class PointwiseNode:
    """A memory-bound pointwise op (add/mul/relu/copy/...). Its per-core compute
    is negligible next to the HBM traffic the memory term already charges, so its
    node cost is ``None`` == 0. Compute-bound transcendentals (exp/gelu/tanh) are
    deliberately deferred to a future registry entry."""


@dataclass(frozen=True)
class MatmulNode:
    """A matmul / batch-matmul / convolution, scored by the native estimator with
    its HBM component removed. Each axis is a ``(size, split)`` pair; ``shared_weight``
    marks a loaded-once (projection / broadcast) RHS."""

    b_axis: tuple[int, int]
    m_axis: tuple[int, int]
    n_axis: tuple[int, int]
    k_axis: tuple[int, int]
    max_cores: int
    shared_weight: bool = False


@dataclass(frozen=True)
class ReductionNode:
    """A cross-core reduction: a ``reduction_split``-way split spreads the sum over
    that many cores, costing ``(split - 1)`` PSUM-ring hops per output element per
    core. ``output_elems_per_core`` is the post-split per-core output size.
    (A clean, unsplit reduction has ``reduction_split <= 1`` and costs nothing.)"""

    reduction_split: int
    output_elems_per_core: int
    shared_weight: bool = True


def _matmul_node_us(op: MatmulNode) -> float:
    """Matmul node cost in µs: the native estimate minus its ``hbm_us`` component.

    Stripping the *exact* ``_matmul_hbm_us`` the estimator added (not a re-derived
    formula) guarantees the traffic lives in the memory term once and only once.
    The remaining terms are compute, PSUM, and the schedule-shape penalties --
    none of which have a tensor-residency analog."""
    wd = _wd()
    total = wd._matmul_split_cost(
        op.b_axis, op.m_axis, op.n_axis, op.k_axis, op.max_cores, op.shared_weight
    )
    if not math.isfinite(total):
        raise ValueError(
            "matmul split is infeasible (cores_used > max_cores); the enumerated "
            "division menu should never offer such a split"
        )
    hbm = wd._matmul_hbm_us(
        op.b_axis, op.m_axis, op.n_axis, op.k_axis, op.shared_weight
    )
    # Floating subtraction of two finite non-negatives; clamp the tiny negative
    # that fp cancellation can produce when the estimate is essentially all-HBM.
    return max(0.0, total - hbm)


def _reduction_node_us(op: ReductionNode) -> float:
    """Cross-core reduction PSUM-ring cost in µs, mirroring the matmul estimator's
    PSUM term: ``max(0, split - 1) * output_elems_per_core * coeff``."""
    wd = _wd()
    coeff = (
        wd._PSUM_PER_CORE_ELEM_US if op.shared_weight else wd._BMM_PSUM_PER_CORE_ELEM_US
    )
    return max(0, op.reduction_split - 1) * op.output_elems_per_core * coeff


def node_cost_us(op: object) -> Optional[float]:
    """Node cost in microseconds, or ``None`` for a free (pointwise) op."""
    if isinstance(op, PointwiseNode):
        return None
    if isinstance(op, MatmulNode):
        return _matmul_node_us(op)
    if isinstance(op, ReductionNode):
        return _reduction_node_us(op)
    raise TypeError(f"unknown node op type: {type(op).__name__}")


def node_cost_fixed(op: object) -> int:
    """Node cost in fixed-point time units; ``0`` for a free (pointwise) op."""
    us = node_cost_us(op)
    return 0 if us is None else to_fixed_us(us)


def node_term_fixed(ops: Iterable[object]) -> int:
    """The node term: the summed per-op fixed-point node costs (each op is rounded
    once, then integer-summed)."""
    return sum(node_cost_fixed(op) for op in ops)


# ===========================================================================
# Unified scorer
# ===========================================================================


def score_fixed(edges: Iterable[MemoryEdge], ops: Iterable[object]) -> int:
    """The full objective in fixed-point time units: memory term + node term.
    Lower is better.

    The two terms are computed from independent, locally-decomposable
    contributions (per-edge :class:`MemoryEdge`, per-op node cost), so an
    incremental caller can maintain running sums and re-score a single-op change
    in O(edges/ops touched) rather than re-summing the graph. This one-shot form
    is the reference the incremental path is checked against.
    """
    return memory_term_fixed(edges) + node_term_fixed(ops)
