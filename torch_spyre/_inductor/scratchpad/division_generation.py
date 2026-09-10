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

import math
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Callable, Optional

import sympy
from torch._inductor.dependencies import Dep, MemoryDep
from torch._inductor.ir import Operation

from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    op_read_writes,
    _per_core_view_from_prep,
    _prepare_per_core_view,
    tile_ownership_view,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision, TileSpec


def _reduction_syms(
    op: Operation, splits: dict[sympy.Symbol, int]
) -> frozenset[sympy.Symbol]:
    """Get reduction symbols for an operation."""
    rw = op_read_writes(op)
    write = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write is None:
        return frozenset()
    return frozenset(s for s in splits if write.index.coeff(s) == 0)


def _core_division(
    op: Operation,
    splits: dict[sympy.Symbol, int],
    tiling: "TileSpec | None" = None,
    tile_splits: tuple[tuple[sympy.Symbol, int], ...] = (),
) -> CoreDivision:
    """Classify one symbol-keyed candidate for its producing operation.

    ``tiling`` is the coarse tiling the candidate was enumerated under, carried
    onto the division so the pair travels together, and ``tile_splits`` is
    that tiling resolved to ``op``'s loop symbols. Output/reduction
    classification asks only whether a symbol *appears* in the write index,
    which a tiling rescales but never eliminates, so it is tiling-invariant and
    the same test serves both frames.
    """
    sparse = {s: v for s, v in splits.items() if v > 1}
    return CoreDivision(
        splits=sparse,
        reduction_syms=_reduction_syms(op, sparse),
        tiling=tiling if tiling is not None else TileSpec(),
        tile_splits=tile_splits,
    )


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

    This is core ownership only, on the untiled buffer whatever the division's
    tiling; how the tiling owns the buffer is :func:`_tile_view_for_div`.
    """
    key = (op.get_name(), dep, buf_name)
    if key not in prep_cache:
        prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
    splits = division.splits
    syms = _reduction_syms(op, splits)
    return _per_core_view_from_prep(
        prep_cache[key],
        splits,
        {k: v for k, v in splits.items() if k in syms},
    )


_WHOLE_VIEW = PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=1)


def _tile_view_for_div(
    op: Operation,
    dep: MemoryDep,
    buf_name: str,
    division: CoreDivision,
    prep_cache: dict,
) -> Optional[PerCoreView]:
    """How ``division``'s coarse tiling owns ``buf_name``, tile by tile.

    The whole-buffer view when untiled, and ``None`` when the tiling cannot be
    represented. Kept apart from the per-core view so it is built once per
    tiling rather than once per division: comparing the two separately is the
    same test as comparing ownership per (tile, core), since a tile owns a slice
    of each tiled dim and a core a slice of that tile.
    """
    if not division.tile_splits:
        return _WHOLE_VIEW
    key = ("tile", op.get_name(), dep, buf_name, division.tile_splits)
    if key not in prep_cache:
        prep_key = (op.get_name(), dep, buf_name)
        if prep_key not in prep_cache:
            prep_cache[prep_key] = _prepare_per_core_view(op, dep, buf_name)
        prep_cache[key] = tile_ownership_view(
            prep_cache[prep_key], division.tile_splits
        )
    return prep_cache[key]


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

    A producer rejected for LX is excluded outright. Otherwise, check each
    producer-consumer edge independently. A broadcasting clone may read its
    input from HBM and still keep its completed output in LX for a matching
    consumer. Candidate-specific checks are in :meth:`parent_view` and
    :meth:`consumer_view`.

    ``read_deps`` holds every read the consumer makes of the buffer. One
    residency serves them all, so a candidate has to own the buffer the same
    way through each. ``a + a.permute(1, 0, 2)`` reads ``a`` once in step and
    once transposed: a division that splits dim 0 or dim 1 slices ``a`` along
    one of them for the first read and the other for the second, so it has no
    pair. One that splits only dim 2, which both reads walk alike, or does not
    split at all, still does.
    """

    buf_name: str
    parent_op: Operation
    consumer_op: Operation
    write_dep: MemoryDep
    read_deps: tuple[MemoryDep, ...]
    prep_cache: dict

    def parent_view(self, division: CoreDivision) -> Optional[PerCoreView]:
        """The producer's write-view under ``division``, or ``None`` when that
        candidate cannot host a readable residency: a partial-reduction write
        (output not final) or an unrepresentable slicing. Matching compares
        the complete per-core views, including all split dimensions."""
        view, partial, repr_ok = _view_for_div(
            self.parent_op, self.write_dep, self.buf_name, division, self.prep_cache
        )
        if not repr_ok or partial:
            return None
        return view

    def _common_read_view(
        self, view_of: Callable[[MemoryDep], Optional[PerCoreView]]
    ) -> Optional[PerCoreView]:
        """The view every read of the buffer has under ``view_of``, or ``None``
        when some read has none or two reads own the buffer differently."""
        first, *rest = (view_of(dep) for dep in self.read_deps)
        if first is None:
            return None
        if any(view is None or not first.same_partition(view) for view in rest):
            return None
        return first

    def consumer_view(self, division: CoreDivision) -> Optional[PerCoreView]:
        """The consumer's read-view under ``division``, or ``None`` when its
        slicing of the buffer is unrepresentable -- we never pin on a slicing
        we cannot verify -- or differs from one read of the buffer to another."""

        def core_view(dep: MemoryDep) -> Optional[PerCoreView]:
            view, _partial, repr_ok = _view_for_div(
                self.consumer_op, dep, self.buf_name, division, self.prep_cache
            )
            return view if repr_ok else None

        return self._common_read_view(core_view)

    def consumer_tile_view(self, division: CoreDivision) -> Optional[PerCoreView]:
        """How the consumer's reads walk the buffer tile by tile under
        ``division``, or ``None`` when that cannot be represented or differs
        from one read to another (see :func:`_tile_view_for_div`)."""
        return self._common_read_view(
            lambda dep: _tile_view_for_div(
                self.consumer_op, dep, self.buf_name, division, self.prep_cache
            )
        )

    @staticmethod
    def _cores_used(division: CoreDivision) -> int:
        return math.prod(division.splits.values())

    def compatible(
        self,
        parent_splits: dict[sympy.Symbol, int],
        consumer_splits: dict[sympy.Symbol, int],
    ) -> bool:
        """Whether the two candidates induce the same per-core slicing of the
        buffer on the same total core count. Equal views alone are not enough:
        a producer on N and a consumer on M > N cores can share a slicing while
        the consumer's extra (broadcast-axis) cores hold no copy and would read
        stale LX.

        :meth:`match_pairs` answers this over two menus and caches each side's
        view across the cross product; this is the single-pair form, for a
        caller holding one untiled candidate per side rather than a list.
        """
        parent = CoreDivision(splits=parent_splits)
        consumer = CoreDivision(splits=consumer_splits)
        if self._cores_used(parent) != self._cores_used(consumer):
            return False
        parent_view = self.parent_view(parent)
        return parent_view is not None and parent_view == self.consumer_view(consumer)

    def match_pairs(
        self,
        parent_divisions: Sequence[CoreDivision],
        consumer_divisions: Sequence[CoreDivision],
    ) -> list[tuple[int, int]]:
        """Compatible ``(parent index, consumer index)`` pairs, with each side's
        view computed once per candidate rather than once per pair.

        A pair must agree on core ownership and on tile ownership: the consumer
        of a tiled producer reads it one tile at a time, so tile ``t`` has to
        touch the same slice on both sides (see :func:`_tile_view_for_div`).
        Two untiled divisions have equal, whole-buffer tile views."""
        parent_views = [self.parent_view(cd) for cd in parent_divisions]
        consumer_views = [self.consumer_view(cd) for cd in consumer_divisions]
        parent_tiles = [
            _tile_view_for_div(
                self.parent_op, self.write_dep, self.buf_name, cd, self.prep_cache
            )
            for cd in parent_divisions
        ]
        consumer_tiles = [self.consumer_tile_view(cd) for cd in consumer_divisions]
        return [
            (i, j)
            for i, (parent_view, parent_tile) in enumerate(
                zip(parent_views, parent_tiles)
            )
            if parent_view is not None and parent_tile is not None
            for j, (consumer_view, consumer_tile) in enumerate(
                zip(consumer_views, consumer_tiles)
            )
            if consumer_view is not None
            and consumer_tile is not None
            and parent_view.same_partition(consumer_view)
            and parent_tile.same_partition(consumer_tile)
            and self._cores_used(parent_divisions[i])
            == self._cores_used(consumer_divisions[j])
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
    write_dep = next(
        (
            w
            for w in op_read_writes(parent_op).writes
            if w.name == buf_name and isinstance(w, MemoryDep)
        ),
        None,
    )

    def wrapped_hasattr(obj, attr):
        try:
            return hasattr(obj, attr)
        except NotImplementedError:
            return False

    read_deps = tuple(
        r for r in consumer_reads if r.name == buf_name and isinstance(r, MemoryDep)
    )
    if write_dep is None or not read_deps:
        return None
    return ResidencyEdge(
        buf_name=buf_name,
        parent_op=parent_op,
        consumer_op=consumer_op,
        write_dep=write_dep,
        read_deps=read_deps,
        prep_cache=prep_cache,
    )
