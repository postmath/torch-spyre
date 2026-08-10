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

"""Cost-model ``OpFeatures`` for a *candidate* core division.

The vendored extractor (:mod:`torch_spyre._inductor.dump_cost_model`) reads each
op's **committed** ``op_it_space_splits``, so it yields features for the division
the compiler already chose. A co-optimizer needs features for every division in a
buffer's candidate menu, because the division is exactly what it is searching
over.

That turns out to need no re-derivation. ``CoreDivision`` stores
``(output_splits, reduction_splits)`` -- the coeff-keyed ``ItSpaceSplits`` pair
produced by :func:`pass_utils.splits_by_index_coeff` -- which is the same type,
in the same encoding, that ``op_it_space_splits`` holds and that
:func:`pass_utils.apply_splits_from_index_coeff` consumes. So a candidate is
evaluated by temporarily installing its pair on the op and re-running the
extractor: every division-dependent field (``cores``, ``reduction_cores``,
``matmul_rows_per_core`` / ``_cols_per_core``, ``tile_rows_per_core``) is then
recomputed by the vendored code rather than by a second, drifting copy of its
axis-decoding rules.

``is_lx`` is deliberately *not* resolved here. It is the other half of what the
co-optimizer searches over, it is a plain per-argument flag, and no other feature
depends on it -- so features are emitted once per (op, division) and residency is
applied at scoring time by :func:`with_residency`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from torch._inductor.ir import ComputedBuffer

from torch_spyre._inductor import config
from torch_spyre._inductor.constants import DEVICE_NAME
from torch_spyre._inductor.cost_model import OpFeatures
from torch_spyre._inductor.logging_utils import get_inductor_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

logger = get_inductor_logger("scratchpad.op_features")

# The attribute the vendored extractor reads the division from.
_SPLITS_ATTR = "op_it_space_splits"


def features_for_division(op, division: "CoreDivision") -> Optional[OpFeatures]:
    """``OpFeatures`` for ``op`` as if it were divided per ``division``.

    Returns ``None`` when the op cannot be featurized (the vendored extractor is
    best-effort and swallows its own failures, so a ``None`` here means the op
    itself was rejected, not that the division was bad).

    Temporarily swaps ``op_it_space_splits``. The swap is restored on every path
    including failure, so a caller that iterates a menu leaves the op exactly as
    it found it -- important because the same ``op`` objects stay live in the
    graph after capture.
    """
    from torch_spyre._inductor.dump_cost_model import extract_op_features

    had = hasattr(op, _SPLITS_ATTR)
    saved = getattr(op, _SPLITS_ATTR, None)
    try:
        setattr(op, _SPLITS_ATTR, (division.output_splits, division.reduction_splits))
        return extract_op_features(op)
    except Exception:  # noqa: BLE001 - featurization is best-effort by design
        logger.debug("could not featurize op for a candidate division", exc_info=True)
        return None
    finally:
        if had:
            setattr(op, _SPLITS_ATTR, saved)
        else:  # never had one: do not leave an attribute the op did not carry
            try:
                delattr(op, _SPLITS_ATTR)
            except AttributeError:
                pass


def features_for_menu(op, divisions) -> list[Optional[OpFeatures]]:
    """``features_for_division`` over a buffer's whole candidate menu, index for
    index with ``divisions`` so a menu index selects its features directly."""
    return [features_for_division(op, cd) for cd in divisions]


def _is_fusable_operation(op) -> bool:
    """The IR-operation analogue of ``fusion._is_fusable_node``.

    A ``ComputedBuffer`` on the Spyre device is what becomes a fusable
    ``SchedulerNode`` later; anything else -- an extern kernel, a fallback, a CPU
    op -- forces a bundle boundary. This is an *estimate*: whether a given
    operation ends up fused or extern is a scheduling decision that has not been
    made yet at this point in the pipeline.
    """
    if not isinstance(op, ComputedBuffer):
        return False
    device = op.get_device()
    return device is not None and device.type == DEVICE_NAME


def estimate_bundles(operations: Sequence) -> list[list]:
    """Estimate the SuperDSC bundles (fused kernels) ``operations`` will become.

    The cost model scores one bundle at a time and bundle membership changes the
    result -- external inputs are deduplicated across a bundle, the pointwise
    arity derate counts its ops, the underfill derate takes its worst tile, and
    the turnaround term uses its totals -- so the co-optimizer needs the grouping
    to score anything faithfully.

    It cannot ask for the real one. This runs in the last *pre-scheduling* pass,
    where ``V.graph.scheduler`` is still ``None``; fusion is decided two stages
    later by :func:`fusion.spyre_fuse_nodes`. What makes an estimate viable is
    that the real rule is order-preserving and structural, so it is reproduced
    here by sharing :func:`fusion.group_contiguous_fusable` and supplying the
    IR-level predicate.

    Returns groups of the input operations, in order, so ``[op.get_name() ...]``
    per group gives the buffer names in each bundle.

    Fusion can be off entirely (``config.bundle_symbolic_args``), in which case
    the real pass leaves every node alone and this returns one bundle per
    operation to match.

    Accuracy, measured against the real grouping on a softmax graph (estimate at
    allocator time vs. :func:`fusion.spyre_fuse_nodes` output at fusion time,
    compared by written buffer name):

    ==============  ==================  ==========================
    ..              boundary bundle     fused bundle
    ==============  ==================  ==========================
    estimated       ``buf6``            ``buf0`` .. ``buf5``
    actual          ``buf6`` (extern)   ``buf7``, ``buf0`` .. ``buf5``
    ==============  ==================  ==========================

    The bundle count, the run structure and the boundary placement all came out
    right -- the extern kernel was correctly identified as a boundary. What the
    estimate missed is ``buf7``, a ``SchedulerNode`` that does not exist in
    ``graph.operations`` at this point because it is created later in scheduling.
    So expect the shape to be right and the membership to under-count by any
    nodes that scheduling introduces.
    """
    from torch_spyre._inductor.fusion import group_contiguous_fusable

    if not config.bundle_symbolic_args:
        return [[op] for op in operations]
    return group_contiguous_fusable(list(operations), _is_fusable_operation)


def with_residency(features: OpFeatures, lx_names: set[str]) -> OpFeatures:
    """``features`` with each argument's ``is_lx`` set from ``lx_names``.

    The cost model charges an LX-resident argument no HBM traffic
    (``elems * loop_factor * (1 - is_lx)``), so this is what turns a placement
    decision into a cost. Returns a new object; the input is left alone so one
    extracted menu can be scored against many candidate placements.
    """
    return dataclasses.replace(
        features,
        args=[
            dataclasses.replace(a, is_lx=(a.name in lx_names)) for a in features.args
        ],
    )
