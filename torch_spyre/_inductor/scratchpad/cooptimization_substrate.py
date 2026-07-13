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

"""Co-optimization substrate adapter: the §7 coupling surface, in one place.

The joint work-division + LX-layout SA engine is built as a third engine on a
colleague's (unlanded, still-moving) joint core-division + LX-placement
substrate -- a sibling to that substrate's CP-SAT and DFS engines.  This module
is the *one* seam where that dependency lives, per the implementation plan's
substrate-churn-isolation rule (Plan §8.1 / Phase 0): the SA engine and its
tests code against the structural :class:`typing.Protocol` types declared here,
never against the real substrate's concrete classes.  Two implementations
satisfy the surface:

* the **fake** substrate (``tests/inductor/fake_cooptimization_substrate.py``),
  which replays captured real dumps -- used through Phases 0-5; and
* the **real** substrate's ``CoreDivision`` / ``CoreDivisionBuffer`` /
  ``CoOptimizingSolver`` (``plan_solver.py`` on the colleague's branch), bound in
  at Phase 6 (the "real adapter impl").

Because the data surface is expressed as *structural* protocols, the real
dataclasses satisfy it directly -- no wrapping/conversion is needed, only the
Phase-6 binding described at :data:`_REAL_SUBSTRATE_SEAM`.

The exact members mirrored here are the Plan §7 coupling surface; when the
colleague's branch churns, re-verify only against this file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class CoreDivisionProtocol(Protocol):
    """One permissible core-division of a buffer's producing op (Plan §7 / §2.1).

    Mirrors the read surface the SA engine uses from the substrate's
    ``CoreDivision``: the stride/coeff-keyed split encoding plus the derived
    per-core-slicing queries.
    """

    # Stride/coeff-keyed split encoding (the shape of ``op.op_it_space_splits``).
    output_splits: dict[int, int]
    reduction_splits: dict[int, int]

    @property
    def cores_used(self) -> int:
        """Total cores this division consumes (product of every split).

        Not in the minimal §7 list, but the SA engine reads it to reason about
        the ``product of splits <= SENCORES`` budget (Plan §2.1); both the fake
        and real substrate expose it.
        """
        ...

    @property
    def is_clean(self) -> bool:
        """True when no reduction axis is split (no per-core partial sums), so
        the output is fully sliced across cores."""
        ...

    @property
    def output_partition(self) -> int:
        """How many cores the output buffer is sliced across."""
        ...

    def signature_key(self) -> Optional[tuple]:
        """Per-core slicing signature, or ``None`` for a reduction-split division.

        Part of the §7 coupling surface, but the *current* substrate does not
        consult it (nor ``is_clean``): compatibility is decided by per-core-view
        comparison in ``cd_parent_matches``, which a coeff-keyed signature like
        this one would conflate across reductions/reshapes. So this is **not** the
        K-split eviction gate (contrary to an earlier reading of Plan §7.4) -- a
        reduction-split producer is evicted because ``cd_parent_matches`` excludes
        its partial-reduction write on the producer side, so its division never
        appears as a ``parent_idx``; a reduction split may still appear as a
        *consumer* index (a K-split reading a clean parent via the PSUM ring).
        """
        ...


@runtime_checkable
class CoreDivisionBufferProtocol(Protocol):
    """A buffer carrying joint core-division metadata (Plan §7 coupling surface).

    Combines the lifetime/liveness fields the layout packer needs with the
    core-division menu, the ``cd_parent_matches`` slicing gate, the residency
    pin, and the two solver-written outputs (``chosen_division``, ``address``).
    """

    # --- identity & liveness (the substrate's LifetimeBoundBuffer base) ------ #
    name: str
    size: int
    # Sorted op indices at which the buffer is accessed; non-empty.
    uses: list[int]
    # True for graph inputs (all accesses reads); False for computed buffers.
    first_use_is_read: bool
    in_place_parents: list[str]

    # --- joint core-division metadata ---------------------------------------- #
    # Pre-enumerated, deduped candidate menu; seed/legacy division at index 0.
    core_divisions: Sequence[CoreDivisionProtocol]
    # Producer buffer names; defines the producer->consumer edges for matching.
    parents: list[str]
    # parent_name -> [(parent_div_idx, this_div_idx), ...] pairs inducing the
    # same per-core slicing.  The sole slicing-match predicate; an absent/empty
    # entry means no compatible division across that edge (Plan §3.1 / §7.2).
    cd_parent_matches: dict[str, list[tuple[int, int]]]
    # HBM traffic a *resident* buffer still incurs (graph-boundary clones).
    boundary_cost: int
    # HBM traffic a *spilled* buffer incurs beyond consumer re-reads (the
    # producer's own HBM write).
    spill_write_cost: int
    # Why the buffer may not be resident, or ``None`` if it may (Plan §7.4).
    residency_reason: Optional[str]
    # Residency pin flag; False iff pinned out of LX (kept in sync with
    # ``residency_reason`` by the substrate).
    placement: bool

    # --- solver-written outputs ---------------------------------------------- #
    # Chosen menu index (a division *state* is one index per buffer, Plan §7.2)
    # and LX address (``None`` when spilled).  The SA engine writes both.
    chosen_division: Optional[int]
    address: Optional[int]

    @property
    def residency_allowed(self) -> bool:
        """True iff the buffer may be made resident (the ``placement`` flag).

        This is the fixed, division-invariant pin gate; the compatibility and
        packer gates are layered on top of it (Plan §7.4, three-gate residency).
        """
        ...

    @property
    def start_time(self) -> int:
        """First access (``uses[0]``)."""
        ...

    @property
    def end_time(self) -> int:
        """One past the last access (``uses[-1] + 1``)."""
        ...


class CoOptimizingSolver(ABC):
    """Abstract joint core-division + LX-placement solver (Plan §7 / §7.3).

    The base class the SA engine (``SaCoOptimizingSolver``, Phase 3) subclasses,
    a sibling to the substrate's CP-SAT and DFS engines.  It mirrors the real
    substrate's ``CoOptimizingSolver`` ABC so that, at Phase 6, the SA engine can
    be re-parented onto the real base with no interface change (see
    :data:`_REAL_SUBSTRATE_SEAM`).

    A concrete solver consumes :class:`CoreDivisionBufferProtocol` buffers (each
    carrying a candidate ``core_divisions`` menu and the ``cd_parent_matches``
    slicing gate), jointly chooses a division and an LX placement per buffer, and
    returns the same buffers with ``chosen_division`` / ``address`` written and
    ``spill_reasons`` populated.
    """

    def __init__(self, size: int, alignment: int = 128) -> None:
        """Args:
        size: total scratchpad capacity in bytes.
        alignment: byte alignment for every placement (128 = one Spyre stick).
        """
        self.limit = size
        self.alignment = alignment
        # Per-buffer drop cause for the most recent solve ({name: reason},
        # spilled buffers only); read back by the allocator's debug log.
        self.spill_reasons: dict[str, str] = {}

    @abstractmethod
    def plan_layout_and_core_divs(
        self,
        buffers: Sequence[CoreDivisionBufferProtocol],
        log_lx_usage: bool = False,
    ) -> list[CoreDivisionBufferProtocol]:
        """Choose a core division and LX placement for each buffer.

        Returns the same buffers with ``address`` (``None`` when spilled) and
        ``chosen_division`` set, and ``self.spill_reasons`` populated.
        """
        ...

    @staticmethod
    def _spill_cost(buffer: CoreDivisionBufferProtocol, num_children: int) -> int:
        """HBM traffic if ``buffer`` is spilled: one re-read per consumer times
        its size, plus the producer's own HBM write (``spill_write_cost``).

        Shared with the substrate's other engines so their objectives agree.
        Units follow ``buffer.size``.  (Note: this is the substrate's
        consumer-*count* traffic proxy; the SA engine's richer shared scorer adds
        the node term and cohort multiplicity on top -- Plan §7.1.)
        """
        return num_children * buffer.size + buffer.spill_write_cost


# Phase-6 binding point.  While the colleague's branch is unlanded (Phases 0-5)
# the SA engine subclasses the local :class:`CoOptimizingSolver` above and runs
# against the fake substrate.  At Phase 6 the real adapter impl re-parents the SA
# engine onto the substrate's own ``CoOptimizingSolver`` (so ``select_allocator``
# recognizes it) -- the structural protocols above already accept the real
# ``CoreDivision`` / ``CoreDivisionBuffer`` unchanged, so no data conversion is
# needed.  Keep this the single edit site for that swap.
_REAL_SUBSTRATE_SEAM = "torch_spyre._inductor.scratchpad.plan_solver:CoOptimizingSolver"
