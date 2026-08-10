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

from collections.abc import Callable

from torch._inductor.scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from . import config
from .constants import DEVICE_NAME
from .scheduler import CountedLoopSchedulerNode


def _make_fused(
    nodes: list[SchedulerNode | CountedLoopSchedulerNode],
) -> BaseSchedulerNode | None:
    if len(nodes) > 1:
        return FusedSchedulerNode(nodes[0].scheduler, nodes)
    elif len(nodes) == 1:
        return nodes[0]
    return None


def _is_spyre_node(node: BaseSchedulerNode) -> bool:
    """True if the node computes on the Spyre device."""
    device = node.get_device()
    return device is not None and device.type == DEVICE_NAME


def _is_fusable_node(node: BaseSchedulerNode) -> bool:
    """True if ``node`` may join the run being fused: a Spyre compute node.

    Everything else -- fallback nodes, CPU nodes, extern kernels -- forces a
    bundle boundary.
    """
    return isinstance(node, (SchedulerNode, CountedLoopSchedulerNode)) and (
        _is_spyre_node(node)
    )


def group_contiguous_fusable(items: list, is_fusable: Callable) -> list[list]:
    """Split ``items`` into maximal contiguous runs of fusable entries, with each
    non-fusable entry its own single-element run.

    This is the *policy* behind :func:`spyre_fuse_nodes` -- one SuperDSC bundle
    per maximal run of fusable ops, order preserved -- factored out so that the
    co-optimizer, which must estimate the same grouping two stages earlier (from
    IR operations, before any scheduler exists), shares it rather than carrying a
    copy that can drift. Callers differ only in ``is_fusable``, which is the
    irreducible difference: one sees scheduler nodes, the other IR operations.

    See ``scratchpad/op_features.estimate_bundles`` for the early-stage caller.
    """
    groups: list[list] = []
    run: list = []
    for item in items:
        if is_fusable(item):
            run.append(item)
            continue
        if run:
            groups.append(run)
            run = []
        groups.append([item])  # a boundary is its own bundle
    if run:
        groups.append(run)
    return groups


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.
    """
    if len(nodes) == 0:
        return nodes
    if not config.bundle_symbolic_args:
        # Without symbolic args, tensor addresses are baked-in constants from
        # SEGMENT_OFFSETS, which has a fixed number of slots.  Fusing ops could
        # exceed that slot count, so disable fusion when symbolic args are off.
        return nodes

    # One bundle per maximal contiguous run of fusable nodes; a boundary node
    # comes back as a single-element run, and ``_make_fused`` returns it
    # unchanged, so this preserves the previous behaviour exactly.
    fused_nodes: list[BaseSchedulerNode] = []
    for group in group_contiguous_fusable(nodes, _is_fusable_node):
        if fused := _make_fused(group):
            fused_nodes.append(fused)
    return fused_nodes
