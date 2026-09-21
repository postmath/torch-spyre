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

"""Declarative coarse tiling, applied inside the scratchpad planning pass.

The tiling is stated as data (a
:class:`~torch_spyre._inductor.scratchpad.plan_solver.TileSpec` per op) and
*applied* to a real graph through a :class:`ScratchpadOptimizationPass`. The
tiling is an input here, not a search -- candidate enumeration and the solver
that chooses among tilings live elsewhere.

The pass mints hint ids and a group-id offset from bases derived off the graph
(never a reserved constant), so a tiling applied here cannot collide with a
hint-driven group already stamped pre-stickification at pass 430. It reuses the
existing ``coarse_tile`` machinery verbatim; the only new work is lowering a
``TileSpec`` to per-op ``DimHint``s and deriving groups as consecutive runs of
ops that share a spec.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Optional

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Reduction

from ..errors import Unsupported
from ..pass_utils import op_out_coords
from ..propagate_hints import DimHint
from ..wsr.coarse_tile import (
    coarse_tile_post_stickify,
    plan_coarse_tile_groups,
    reduction_loop_vars,
    validate_coarse_tile_groups,
)
from .allocator import ScratchpadOptimizationPass
from .plan_solver import TileSpec


def tile_spec_to_dim_hints(
    op: ComputedBuffer,
    spec: TileSpec,
    hint_ids: Sequence[int],
) -> list[DimHint]:
    """Lower a :class:`TileSpec` into per-op :class:`DimHint`s.

    Each :class:`TileAxis` becomes one ``DimHint`` carrying the axis's split
    count and the op's *own* loop variable for that axis, paired with the group's
    ``hint_id`` for that level. ``hint_ids`` has one entry per axis, outermost
    first, matching the group's ``levels``.

    The output-axis case is exactly ``_dims_to_hints`` (span overflow): resolve
    the loop var from ``op_out_coords(op)[host_dim]``. The reduction-axis case is
    the inverse of :func:`reduction_loop_vars` -- ``host_dim`` positionally
    indexes the op's ordered reduction loop variables.
    """
    if len(hint_ids) != len(spec.axes):
        raise ValueError(
            f"tile_spec_to_dim_hints: {len(hint_ids)} hint_ids for "
            f"{len(spec.axes)} axes on {op.get_name()}"
        )
    out_coords = op_out_coords(op)
    red_vars: list[sympy.Symbol] | None = None
    hints: list[DimHint] = []
    for axis, hint_id in zip(spec.axes, hint_ids):
        if axis.is_reduction:
            if not isinstance(op.data, Reduction):
                raise Unsupported(
                    f"coarse tiling: reduction axis host_dim={axis.host_dim} "
                    f"requested on non-Reduction op {op.get_name()}."
                )
            if red_vars is None:
                red_vars = reduction_loop_vars(op)
            if axis.host_dim >= len(red_vars):
                raise Unsupported(
                    f"coarse tiling: reduction host_dim={axis.host_dim} is out "
                    f"of bounds for {len(red_vars)} reduction loop variables on "
                    f"{op.get_name()}."
                )
            loop_var = red_vars[axis.host_dim]
        else:
            if axis.host_dim >= len(out_coords):
                raise Unsupported(
                    f"coarse tiling: host_dim={axis.host_dim} is out of bounds "
                    f"for {len(out_coords)} output coordinates on {op.get_name()}."
                )
            coord = out_coords[axis.host_dim]
            free_symbols = coord.free_symbols
            if len(free_symbols) != 1:
                raise Unsupported(
                    f"coarse tiling: host_dim={axis.host_dim} output coordinate "
                    f"{coord} on {op.get_name()} has {len(free_symbols)} free "
                    "symbols; expected exactly one loop var."
                )
            loop_var = next(iter(free_symbols))
        hints.append(
            DimHint(
                dim_names=["_coarse_tile"],
                split_count=axis.count,
                loop_var=loop_var,
                is_reduction=axis.is_reduction,
                hint_id=hint_id,
            )
        )
    return hints


def _host_dim_walk(
    op: ComputedBuffer, dep: MemoryDep, host_dim: int
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """``(stride, extent)`` of ``dep`` along the loop var at output ``host_dim``
    of ``op``, or ``None`` where that is not a single affine loop var."""
    coords = op_out_coords(op)
    if host_dim >= len(coords) or len(coords[host_dim].free_symbols) != 1:
        return None
    var = next(iter(coords[host_dim].free_symbols))
    stride = sympy.diff(dep.index, var)
    extent = dict(zip(dep.var_names, dep.size)).get(var)
    if stride == 0 or stride.free_symbols or extent is None:
        return None
    return stride, extent


def tile_aligned_host_dims(
    op: ComputedBuffer, producer: ComputedBuffer
) -> frozenset[int] | None:
    """The output host dims of ``producer`` that ``op`` walks, in every read of
    it, along the same buffer dim at the same extent as ``producer`` writes
    them; ``None`` where no spec is read as written (a read or the write that
    is not a :class:`MemoryDep`).

    ``TileAxis.host_dim`` is positional, so a shared spec is not a shared
    logical dim: across ``[A, S, D] -> [S, A, D]`` host dim 0 is ``A`` on one
    side and ``S`` on the other, and the reader would index the writer's
    per-tile scratch as though it held a tile of ``S``. Anything not provably
    aligned counts as misaligned.
    """
    name = producer.get_name()
    write = next(
        (
            w
            for w in producer.get_read_writes().writes
            if isinstance(w, MemoryDep) and w.name == name
        ),
        None,
    )
    if write is None:
        return None
    dims = set(range(len(op_out_coords(producer))))
    for read in op.get_read_writes().reads:
        if read.name != name:
            continue
        if not isinstance(read, MemoryDep):
            return None
        dims = {
            d
            for d in dims
            if (walk := _host_dim_walk(producer, write, d)) is not None
            and walk == _host_dim_walk(op, read, d)
        }
    return frozenset(dims)


def _reads_tiles_as_written(
    op: Operation,
    by_buf: Mapping[str, ComputedBuffer],
    spec: TileSpec,
) -> bool:
    """Whether ``op`` reads each op of ``by_buf`` it reads with every tiled
    output axis of ``spec`` walked as that op writes it."""
    if not isinstance(op, ComputedBuffer):
        return True
    for name in {read.name for read in op.get_read_writes().reads}:
        producer = by_buf.get(name)
        if producer is not None and not spec.read_as_written(
            tile_aligned_host_dims(op, producer)
        ):
            return False
    return True


def derive_tiling_groups(
    graph: GraphLowering,
    choices: Mapping[str, TileSpec],
) -> list[tuple[list[Operation], TileSpec]]:
    """Group consecutive ops that share the same non-empty :class:`TileSpec`.

    Mirrors ``hints_to_coarse_tile_groups``' consecutive-run shape with the hint
    key replaced by the chosen ``TileSpec``: a run breaks whenever an op is
    untiled (absent from ``choices`` or mapped to the empty spec) or its spec
    differs from the run's. Contiguity is a hard requirement, not an
    optimization -- ``validate_coarse_tile_groups`` and ``_apply_plan`` both rely
    on each group occupying one contiguous stretch of the operation list.

    Two non-adjacent runs carrying the same spec are therefore **two groups**,
    each minting its own hint ids and group id. They are not the same group and
    not an error: ``TileSpec`` equality is structural, so unrelated regions
    anywhere in the graph collide on a small alphabet (~6 counts per axis over
    at most two dims), and refusing them would refuse ordinary graphs.

    **Precondition on the caller, which this signature cannot check.** Ops meant
    to tile together have to be contiguous in ``graph.operations``. A chooser
    walking producer/consumer reachability is not walking contiguity: an op it
    could not tile -- a menu-backed one, or one already carrying ``dim_hints``
    -- sitting in the middle of a region leaves the second half reading the
    first half's *full* extent while the chooser priced both at the per-tile
    footprint. That is a mispricing rather than an illegal graph, and a
    name->spec map carries no region identity to detect it with, so it belongs
    to whoever builds ``choices``. ``_validate_contiguous`` remains the backstop
    for the illegal case.

    A run also breaks at an op that reads an op of its *stretch* -- the ops
    since the spec last changed, not only its group -- along a different
    logical dim than the spec tiles it by (:func:`tile_aligned_host_dims`):
    one group would hand it the wrong slice, two make it a cross-group read of
    the full buffer. Taking the stretch over-breaks only where that op is
    already in an earlier group, and makes where a group starts depend on the
    specs between an op and what it reads alone, which the SA co-optimizer
    re-derives per move (``SaCoOptimizingSolver._run_bounds``).

    ``choices`` is keyed by operation name (``op.get_operation_name()``).
    """
    groups: list[tuple[list[Operation], TileSpec]] = []
    current_ops: list[Operation] = []
    current_spec: TileSpec | None = None
    stretch_by_buf: dict[str, ComputedBuffer] = {}
    for op in graph.operations:
        spec = choices.get(op.get_operation_name())
        if spec is not None and spec.is_untiled:
            spec = None
        if spec != current_spec:
            stretch_by_buf = {}
        if (
            spec is not None
            and spec == current_spec
            and _reads_tiles_as_written(op, stretch_by_buf, spec)
        ):
            current_ops.append(op)
        else:
            if current_ops:
                assert current_spec is not None
                groups.append((current_ops, current_spec))
            current_ops = [op] if spec is not None else []
            current_spec = spec
        if spec is not None and isinstance(op, ComputedBuffer):
            stretch_by_buf[op.get_name()] = op
    if current_ops:
        assert current_spec is not None
        groups.append((current_ops, current_spec))
    return groups


def _derive_hint_id_base(graph: GraphLowering) -> int:
    """``max(hint_id present in the graph, default=-1) + 1``.

    Derived, never a reserved constant: whatever hint ids a pre-stickification
    hint-driven group already minted, this pass mints strictly above them, so
    ``validate_coarse_tile_groups`` can never see a hint id in two groups.
    """
    ids = [h.hint_id for op in graph.operations for h in getattr(op, "dim_hints", [])]
    return max(ids, default=-1) + 1


def _derive_group_idx_offset(graph: GraphLowering) -> int:
    """``max(loop_group_id[0] present, default=-1) + 1`` -- the same derivation
    ``_maybe_coarse_tile_span_overflow`` uses to avoid a ``loop_group_id``
    collision with a hint-driven group stamped pre-stickification."""
    used = [
        op.loop_info.loop_group_id[0]
        for op in graph.operations
        if getattr(op, "loop_info", None) is not None
    ]
    return max(used, default=-1) + 1


class CoarseTilingPass(ScratchpadOptimizationPass):
    """Apply a declared coarse tiling to a graph, inside the scratchpad pass.

    The tiling is an *input* (``choices``: operation name -> TileSpec),
    not a search. Consecutive ops sharing a non-empty spec form one loop group;
    the pass mints hint ids and a group-id offset from bases derived off the
    graph, stamps each op's ``dim_hints``, validates group contiguity, then calls
    ``coarse_tile``. With empty (or all-untiled) ``choices`` it is a no-op and
    the op count is unchanged -- which is what keeps it inert until a solver
    hands it real choices.
    """

    def __init__(
        self,
        choices: Mapping[str, TileSpec],
        staged_reads: Optional[Mapping[tuple[str, str], Collection[str]]] = None,
    ) -> None:
        self._choices = dict(choices)
        # ``(source, sizing op)`` pairs a planner placed a staging copy for,
        # each mapped to the readers that copy may serve.
        # ``None`` runs no read copies at all: an unplaced copy lands in HBM,
        # where it costs a write and a read to save nothing.
        self._staged_reads = staged_reads
        # Copy name -> the pair it was staged for, once ``apply_pass`` ran.
        self.staged_copies: dict[str, tuple[str, str]] = {}

    def _stamped_groups(self, graph: GraphLowering) -> list[tuple]:
        """Derive the groups and stamp each member's ``dim_hints``.

        Shared by :meth:`apply_pass` and :meth:`plan_only`, which differ only in
        whether they go on to mutate the IR.
        """
        groups_specs = derive_tiling_groups(graph, self._choices)
        if not groups_specs:
            return []
        # Both bases are derived off the graph *before* this pass stamps any of
        # its own hints/groups, so pre-existing (hint-driven) ids are avoided
        # and the ids this pass mints increase monotonically.
        next_hint_id = _derive_hint_id_base(graph)
        groups: list[tuple] = []
        for group_ops, spec in groups_specs:
            hint_ids = list(range(next_hint_id, next_hint_id + len(spec.axes)))
            next_hint_id += len(spec.axes)
            levels = [
                (hint_id, sympy.Integer(axis.count))
                for hint_id, axis in zip(hint_ids, spec.axes)
            ]
            for op in group_ops:
                op.dim_hints = tile_spec_to_dim_hints(op, spec, hint_ids)
            groups.append((group_ops, levels))
        validate_coarse_tile_groups(groups)
        return groups

    def plan_only(self, graph: GraphLowering) -> None:
        """Raise whatever :meth:`apply_pass` would raise, leaving ``graph`` as
        it was found.

        For a caller that must not be left holding a half-transformed graph: the
        decisions are all made before any IR is rewritten
        (``plan_coarse_tile_groups`` is explicitly zero-mutation), but
        ``dim_hints`` are stamped along the way, so those are restored here. A
        caller that treats a refusal as fatal can then fail on an untouched
        graph -- which matters where something downstream (a fallback solver,
        say) is entitled to assume the graph was never touched.
        """
        saved = [(op, getattr(op, "dim_hints", None)) for op in graph.operations]
        try:
            groups = self._stamped_groups(graph)
            if groups:
                plan_coarse_tile_groups(graph.operations, groups)
        finally:
            for op, hints in saved:
                if hints is None:
                    if hasattr(op, "dim_hints"):
                        del op.dim_hints
                else:
                    op.dim_hints = hints

    def apply_pass(self, graph: GraphLowering) -> None:
        group_idx_offset = _derive_group_idx_offset(graph)
        groups = self._stamped_groups(graph)
        if not groups:
            return
        # This pass runs inside scratchpad/LX planning -- after stickification
        # (insert_restickify) and the post-stickify span-overflow WSR pass -- so
        # every op already carries a committed FixedTiledLayout, and the
        # post-stickify entry point is the right one.
        #
        # Read copies run here where the span-overflow caller leaves them off.
        # Its reason -- nothing minted this late can be in LX, so the copy is
        # HBM-to-HBM -- does not hold for a caller whose whole job is to decide
        # what is in LX: the copy is tile-sized with fresh contiguous strides,
        # which is the cheapest thing a tiled op could hold resident, against an
        # operand the loop otherwise re-reads from HBM every iteration.
        self.staged_copies = coarse_tile_post_stickify(
            graph,
            groups=groups,
            group_idx_offset=group_idx_offset,
            run_read_copies=self._staged_reads is not None,
            staged_reads=self._staged_reads,
        )
