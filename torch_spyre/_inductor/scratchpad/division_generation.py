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

"""Per-candidate core-division machinery -- what stands in for a menu.

A solver can be handed an enumerated list of core-division candidates per op
plus a precomputed ``|D_p| x |D_c|`` compatibility table per edge, or it can
generate candidates as it goes and ask these functions the same questions one
candidate at a time. This module holds the per-candidate side:

* :func:`_core_division` classifies a symbol-keyed split map into the output /
  reduction split pair a :class:`CoreDivision` carries, and
  :func:`_division_splits` restores the complete map from it;
* :class:`ResidencyEdge` owns one producer-buffer -> consumer edge, both the
  geometry (does this pair of candidates slice the buffer identically) and the
  policy filters that decide a candidate can host a readable residency at all.

``allocator.py`` materializes the edge relation as the ``cd_parent_matches``
pair table every engine consumes today.
"""

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Optional

import sympy
from torch._inductor.dependencies import Dep, MemoryDep
from torch._inductor.ir import Operation

from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    iteration_space_from_op,
    op_read_writes,
    op_short_name,
    _is_matmul_op,
    _per_core_view_from_prep,
    _prepare_per_core_view,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision


def _core_division(op: Operation, splits: dict[sympy.Symbol, int]) -> CoreDivision:
    """Classify one symbol-keyed candidate for its producing operation."""
    rw = op_read_writes(op)
    write = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write is None:
        return CoreDivision()
    output = {
        s: int(v) for s, v in splits.items() if write.index.coeff(s) != 0 and v > 1
    }
    reduction = {
        s: int(v) for s, v in splits.items() if write.index.coeff(s) == 0 and v > 1
    }
    return CoreDivision(output_splits=output, reduction_splits=reduction)


def _division_splits(op: Operation, division: CoreDivision) -> dict[sympy.Symbol, int]:
    """Restore a complete symbol-keyed split map from a sparse division."""
    return {
        sym: int(division.output_splits.get(sym, division.reduction_splits.get(sym, 1)))
        for sym in iteration_space_from_op(op)
    }


def _view_for_div(
    op: Operation,
    dep: MemoryDep,
    buf_name: str,
    division: CoreDivision,
    prep_cache: dict,
):
    """One candidate division's per-core view of ``buf_name``.

    ``prep_cache`` holds the candidate-invariant (sympy-heavy) context, keyed by
    ``(op name, dep, buf_name)``: a producer's write-dep and a consumer's
    read-dep on the same buffer can be equal ``MemoryDep``s, so the op name
    keeps their preps distinct while a parent read by several consumers reuses
    its write-view prep.
    """
    key = (op.get_name(), dep, buf_name)
    if key not in prep_cache:
        prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
    return _per_core_view_from_prep(
        prep_cache[key], _division_splits(op, division), division.reduction_splits
    )


def _is_frame_changing_clone(op: Operation, buf_name: str) -> bool:
    """True if ``op`` is a clone whose output ``buf_name`` has an iteration
    dimension that none of its inputs carry -- i.e. it broadcasts a dim
    (e.g. GQA broadcasting K/V over the query-group axis). Such a clone reads
    its input in a different frame than it writes its output, so a per-core
    slice of the output cannot be produced from a core-local slice of the
    input; pinning the output mis-addresses (cf. the restickify barrier)."""
    if op_short_name(op) != "clone":
        return False
    rw = op_read_writes(op)
    write = next(
        (w for w in rw.writes if w.name == buf_name and hasattr(w, "index")), None
    )
    if write is None:
        return False
    read_syms: set = set()
    for r in rw.reads:
        if hasattr(r, "index"):
            read_syms |= set(r.index.free_symbols)
    # A write-only free symbol means the clone expands (broadcasts) that dim.
    return bool(set(write.index.free_symbols) - read_syms)


@dataclass
class ResidencyEdge:
    """One producer-buffer -> consumer edge, with its residency policy applied.

    Owns both halves of "can these two candidates share a residency": the
    *geometry* -- the same per-core slicing of the buffer, compared in the
    buffer's own device-dim frame, on the same total core count -- and the
    *policy* filters that decide a candidate can host a readable residency at
    all. Built once per edge by :func:`build_residency_edge`, which returns
    ``None`` for an edge excluded outright, so a caller that generates
    candidates instead of enumerating them cannot apply the geometry and forget
    the filters.

    Excluded outright (the producer then falls back to HBM, always correct): a
    producer that can never be resident, and a frame-changing (broadcasting)
    clone, whose per-core slice cannot be produced core-locally at all -- the
    single-frame view comparison misses that, and the broadcast read from HBM
    is globally correct. Excluded per candidate: see :meth:`parent_view` and
    :meth:`consumer_view`.
    """

    buf_name: str
    parent_op: Operation
    consumer_op: Operation
    write_dep: MemoryDep
    read_dep: MemoryDep
    # An SDSC carries only a matmul's primary split, so a multi-dim-split matmul
    # output cannot be coherently LX-pinned even when views match -- a consumer
    # would read per-core LX holding only a fragment. (Mirrors #2745's
    # ``get_ncores_for_buffers`` matmul guard for the greedy path.)
    parent_is_matmul: bool
    prep_cache: dict

    def parent_view(self, division: CoreDivision) -> Optional[PerCoreView]:
        """The producer's write-view under ``division``, or ``None`` when that
        candidate cannot host a readable residency: a partial-reduction write
        (output not final), an unrepresentable slicing, or a matmul output
        split across more than one device dim."""
        view, partial, repr_ok = _view_for_div(
            self.parent_op, self.write_dep, self.buf_name, division, self.prep_cache
        )
        if not repr_ok or partial:
            return None
        if self.parent_is_matmul and len(view.work_slice_dims) > 1:
            return None
        return view

    def consumer_view(self, division: CoreDivision) -> Optional[PerCoreView]:
        """The consumer's read-view under ``division``, or ``None`` when its
        slicing of the buffer is unrepresentable -- we never pin on a slicing
        we cannot verify."""
        view, _partial, repr_ok = _view_for_div(
            self.consumer_op, self.read_dep, self.buf_name, division, self.prep_cache
        )
        return view if repr_ok else None

    def compatible(
        self, parent_division: CoreDivision, consumer_division: CoreDivision
    ) -> bool:
        """Whether the two candidates induce the same per-core slicing of the
        buffer on the same total core count. Equal views alone are not enough:
        a producer on N and a consumer on M > N cores can share a slicing while
        the consumer's extra (broadcast-axis) cores hold no copy and would read
        stale LX."""
        if parent_division.cores_used != consumer_division.cores_used:
            return False
        parent_view = self.parent_view(parent_division)
        return parent_view is not None and parent_view == self.consumer_view(
            consumer_division
        )

    def match_pairs(
        self,
        parent_divisions: Sequence[CoreDivision],
        consumer_divisions: Sequence[CoreDivision],
    ) -> list[tuple[int, int]]:
        """Compatible ``(parent index, consumer index)`` pairs, with each side's
        view computed once per candidate rather than once per pair."""
        parent_views = [self.parent_view(cd) for cd in parent_divisions]
        consumer_views = [self.consumer_view(cd) for cd in consumer_divisions]
        return [
            (i, j)
            for i, parent_view in enumerate(parent_views)
            if parent_view is not None
            for j, consumer_view in enumerate(consumer_views)
            if consumer_view is not None
            and parent_view == consumer_view
            and parent_divisions[i].cores_used == consumer_divisions[j].cores_used
        ]


def build_residency_edge(
    buf_name: str,
    parent_op: Operation,
    consumer_op: Operation,
    consumer_reads: Iterable[Dep],
    residency_reason: Optional[str],
    prep_cache: dict,
) -> Optional[ResidencyEdge]:
    """The :class:`ResidencyEdge` for this producer-consumer pair, or ``None``
    when the edge can never host a residency."""
    if residency_reason is not None:
        return None
    if _is_frame_changing_clone(parent_op, buf_name):
        return None
    write_dep = next(
        (
            w
            for w in op_read_writes(parent_op).writes
            if w.name == buf_name and hasattr(w, "index")
        ),
        None,
    )
    read_dep = next(
        (r for r in consumer_reads if r.name == buf_name and hasattr(r, "index")),
        None,
    )
    if write_dep is None or read_dep is None:
        return None
    return ResidencyEdge(
        buf_name=buf_name,
        parent_op=parent_op,
        consumer_op=consumer_op,
        write_dep=write_dep,
        read_dep=read_dep,
        parent_is_matmul=_is_matmul_op(parent_op),
        prep_cache=prep_cache,
    )
