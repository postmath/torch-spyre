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

"""Joint work-division + LX-layout simulated-annealing engine (Plan Phases 3-5).

``SaCoOptimizingSolver`` is the third co-optimization engine (a sibling to the
substrate's CP-SAT and DFS solvers, Plan §7.3). It *composes* the incremental
:class:`PermutationBasedLayoutSolver` packer (not subclasses it) and drives a
single annealing loop over the joint state ``(pi, W)``:

* ``pi`` -- the layout permutation, held in the packer.
* ``W`` -- the work division, one ``chosen_division`` menu index per buffer.

The move set is **reorder, atomic division flip, and region-recolor**, each
structural move run as a compound move+burst and judged as a unit by the shared
scorer's Metropolis test. Region-recolor (Plan §4.3 / §7.2) picks a non-trivial
(split) anchor tiling and floods the ``cd_parent_matches`` relation bidirectionally
to a coordinated, mutually-compatible menu-index assignment over the reachable
region -- the region *is* the flood's reach, boundaries emerge for free, and a
join with no compatible index keeps the tie-break pick and accepts the internal
seam. (Region-recolor is Tier 0 -- plain uniform recolor; the §4.3 Tier 1/2
escalations remain gated on run evidence.)

Schedule / proposal mix (Plan §5). The default ``"reheating"`` schedule is the
multi-move self-calibrating reheating carrier of :mod:`cooptimization_schedule`:
one shared reheating clock, an independent acceptance band + move-scale EMA per
move type (region-recolor coldest), and a cycle-phase proposal mix that weights
each move by its neighborhood size times a hotness that lets structural moves
dominate hot phases and layout reorders dominate cold ones (§5.4). Per-move
acceptance traces and within-group ``|dE|`` CVs are recorded for the §5.3
bucketing decision. ``schedule="crude"`` selects the Phase-3/4 single geometric
cool + fixed weights, retained as the A/B baseline the reheating schedule beats.
Best-seen over ``(pi, W)`` from the §8.2 seed (every op at index 0, ``pi`` from
FirstFit) makes every returned state no worse than the baseline regardless of how
crude the moves/schedule are (§8.1).

Objective. The shared scorer (:mod:`cooptimization_scorer`) is authoritative.
On the (op-metadata-free) fake substrate this reduces to the substrate's own
HBM-traffic model -- per buffer, ``boundary_cost`` when resident else
``num_children * size + spill_write_cost`` when it misses LX -- with a zero node
term and unit cohort multiplicity (Plan §7.1). Residency, per-core sizing, and
the ``cd_parent_matches`` compatibility gate follow the substrate exactly (mirrors
``DfsLayoutSolver._evaluate``). The richer per-edge multiplicity and the matmul
node term are wired in when real ops are available (Phase 6); the scorer already
supports them.

Determinism (Plan §7.5): a seeded ``Random`` over index-ordered domains and the
integer fixed-point score make a run bit-for-bit reproducible.
"""

from __future__ import annotations

import heapq
import math
import random as rnd
import statistics
from collections.abc import Sequence
from typing import Optional, Union

from torch_spyre._inductor.scratchpad.cooling_schedules import (
    SelfCalibratingReheatingSchedule,
)
from torch_spyre._inductor.scratchpad.cooptimization_substrate import (
    CoOptimizingSolver,
    CoreDivisionBufferProtocol,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import SolverToPermutation
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    _NativePackerAdapter,
    make_permutation_packer,
)
from torch_spyre._inductor.scratchpad import cooptimization_scorer as scorer

# The packer is either the pure-Python solver or the native C++ adapter; both
# expose the same permutation-packer interface the co-optimizer drives, so
# ``make_permutation_packer`` may return either. Use ``.quality()`` (not the
# Python-only ``total_quality`` attribute) so both work.
Packer = Union[PermutationBasedLayoutSolver, _NativePackerAdapter]

# Cause recorded for a buffer the SA engine left out of LX (mirrors the
# substrate's shared drop cause).
_SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"

# Per-move-type acceptance bands (accept_hi, accept_lo) for the reheating
# schedule (Plan §5.2): reorders warmest, region-recolor the coldest floor so it
# freezes earliest. Guessed defaults pending benchmarks -- best-seen bounds any
# bad choice, and the per-move acceptance traces validate the resulting rates.
_DEFAULT_MOVE_BANDS = {
    "reorder": (0.6, 0.02),
    "flip": (0.3, 0.005),
    "recolor": (0.1, 0.001),
}


class SaCoOptimizingSolver(CoOptimizingSolver):
    """SA joint core-division + LX-placement engine (see module docstring).

    Args:
        size: scratchpad capacity in bytes.
        alignment: placement alignment (128 = one Spyre stick).
        seed: RNG seed; fixes the (deterministic) search trajectory.
        steps_per_buffer: annealing steps scale linearly with the buffer count
            (crude bounded budget; Plan Appendix H tunes this later).
        min_steps: floor on the step budget for tiny graphs.
        schedule: ``"reheating"`` (Plan §5 -- the multi-move self-calibrating
            reheating schedule + cycle-phase proposal mix) or ``"crude"`` (the
            Phase-3/4 single geometric cool + fixed proposal weights, retained as
            the A/B baseline the reheating schedule must beat).
        cycles / horizons_per_cycle: reheating-schedule knobs (Plan §5.1).
        weight_floor: the ``w_floor`` in the cycle-phase proposal mix (Plan §5.4),
            so no applicable move type is ever fully starved.
        move_bands: per-move-type ``(accept_hi, accept_lo)`` acceptance bands for
            the reheating schedule; defaults to :data:`_DEFAULT_MOVE_BANDS`.
        reorder_weight / flip_weight / recolor_weight: fixed proposal weights for
            the ``"crude"`` schedule only.
        burst_fraction: layout-burst length as a fraction of the buffer count
            (Plan §4.4; the burst warms ``pi`` to the new footprints before the
            compound move is judged).
    """

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        *,
        seed: int = 0,
        steps_per_buffer: int = 40,
        min_steps: int = 200,
        schedule: str = "reheating",
        cycles: int = 4,
        horizons_per_cycle: float = 2.0,
        weight_floor: float = 0.05,
        move_bands: Optional[dict[str, tuple[float, float]]] = None,
        reorder_weight: float = 0.5,
        flip_weight: float = 0.3,
        recolor_weight: float = 0.2,
        burst_fraction: float = 0.5,
        nested: bool = False,
        inner_curve: str = "convex",
        inner_annealed: bool = False,
        inner_len_base: float = 0.25,
        inner_len_max: float = 3.0,
        early_abandon: bool = True,
        polish_frac: float = 0.2,
        abandon_k: float = 30.0,
    ) -> None:
        super().__init__(size, alignment)
        if schedule not in ("reheating", "crude"):
            raise ValueError("schedule must be 'reheating' or 'crude'")
        if inner_curve not in ("constant", "linear", "convex", "adaptive"):
            raise ValueError("inner_curve must be constant|linear|convex|adaptive")
        self._seed = seed
        self._steps_per_buffer = steps_per_buffer
        self._min_steps = min_steps
        self._schedule = schedule
        self._cycles = cycles
        self._horizons_per_cycle = horizons_per_cycle
        self._weight_floor = weight_floor
        # Merge (not replace) so a partial override keeps the defaults for the
        # move types it omits -- otherwise _choose_move_reheating can select a
        # move type whose band is missing and crash in the reheating schedule.
        self._move_bands = {**_DEFAULT_MOVE_BANDS, **(move_bands or {})}
        self._reorder_weight = reorder_weight
        self._flip_weight = flip_weight
        self._recolor_weight = recolor_weight
        self._burst_fraction = burst_fraction
        # Nested two-timescale mode (experimental): the outer loop anneals over
        # structure (flip/recolor) and each proposal runs an inner layout loop
        # whose length grows over the run; see :meth:`_anneal_nested`.
        self._nested = nested
        self._inner_curve = inner_curve
        self._inner_annealed = inner_annealed
        self._inner_len_base = inner_len_base
        self._inner_len_max = inner_len_max
        self._early_abandon = early_abandon
        self._polish_frac = polish_frac
        self._abandon_k = abandon_k
        # Best-seen over the anneal (set in _anneal, read in _step); declared here
        # so their type is known across methods.
        self._best_score: int
        self._best_snap: tuple[Packer, list[int]]

    # -- public interface ----------------------------------------------------

    def plan_layout_and_core_divs(
        self,
        buffers: Sequence[CoreDivisionBufferProtocol],
        log_lx_usage: bool = False,
    ) -> list[CoreDivisionBufferProtocol]:
        """Anneal the joint ``(pi, W)`` state and write ``chosen_division`` /
        ``address`` back to each buffer; populate ``spill_reasons``. Returns the
        same buffers (the one-shot interface satisfied by an internal solve)."""
        self.spill_reasons = {}
        # Tier-0 move instrumentation (Plan §4.3 / §8.3): per-type proposal and
        # accept counts, recolor improvement count, the flooded region sizes, and
        # the anchor-tiling ``output_partition`` of every proposed recolor and of
        # the accepted subset. The last two answer the open "should the anchor
        # tiling be weighted by output_partition?" question empirically -- if
        # aggressive (high-partition) anchors are proposed but rarely accepted
        # once the real node term is on (Phase 6), that is the measurement that
        # justifies escalating to a §4.3 Tier-1 biased proposal. Populated only by
        # the main annealing loop (not the calibration probes).
        self.moves_proposed = {"reorder": 0, "flip": 0, "recolor": 0, "none": 0}
        self.moves_accepted = {"reorder": 0, "flip": 0, "recolor": 0, "none": 0}
        self.recolor_improved = 0
        self.recolor_region_sizes: list[int] = []
        self.recolor_anchor_partitions: list[int] = []
        self.recolor_accepted_partitions: list[int] = []
        self._last_recolor_region_size = 0
        self._last_recolor_anchor_partition = 0
        # Within-group score-delta stats per move type (online n / sum / sum-of-
        # squares over nonzero |dE|), for the §5.3 within-group-CV instrumentation
        # that gates whether a move type needs size-bucketed sub-groups. Read via
        # :meth:`move_scale_cv`.
        self._ms_n = {"reorder": 0, "flip": 0, "recolor": 0, "none": 0}
        self._ms_sum = {"reorder": 0.0, "flip": 0.0, "recolor": 0.0, "none": 0.0}
        self._ms_sqsum = {"reorder": 0.0, "flip": 0.0, "recolor": 0.0, "none": 0.0}
        n = len(buffers)
        if n == 0:
            return list(buffers)

        self._bufs = buffers
        self._rng = rnd.Random(self._seed)
        self._precompute_topology()

        # Seed: every op at the committed division (index 0); pi from FirstFit.
        self.chosen = [0] * n
        self.packer = self._build_seed_packer()

        self._anneal()
        self._write_back()
        return list(buffers)

    # -- static topology (division-invariant) --------------------------------

    def _precompute_topology(self) -> None:
        """Precompute the division-invariant graph structure used every step:
        the name->index map, each buffer's parent indices, and -- keyed by parent
        index -- its children with the ``(parent_div, child_div)`` pairs that keep
        that edge tiling-compatible (mirrors ``DfsLayoutSolver``'s ``children_of``).
        ``num_children`` is the consumer count the spill cost is scaled by.
        """
        bufs = self._bufs
        self._name_to_idx = {b.name: i for i, b in enumerate(bufs)}
        n = len(bufs)
        self._parents_idx: list[set[int]] = [set() for _ in range(n)]
        # parent_idx -> list of (child_idx, frozenset of compatible (p_idx, c_idx))
        self._children: list[list[tuple[int, frozenset]]] = [[] for _ in range(n)]
        for c_idx, c in enumerate(bufs):
            for p_name in c.parents:
                # Every parent resolves: the substrate builds ``parents`` by
                # intersecting an op's reads with the solver's buffer set (reads of
                # graph inputs / constants / externs are dropped), so
                # ``parents ⊆ buffer set`` by construction. Assert it rather than
                # silently skipping -- a miss would signal a coupling-surface change
                # in the (in-flux) substrate branch (Plan §8.1), not stale input.
                p_idx = self._name_to_idx.get(p_name)
                assert p_idx is not None, (
                    f"parent {p_name!r} of {c.name!r} is not in the solver's "
                    f"buffer set (unexpected substrate shape)"
                )
                self._parents_idx[c_idx].add(p_idx)
                pairs = frozenset(
                    (int(a), int(b)) for a, b in c.cd_parent_matches.get(p_name, [])
                )
                self._children[p_idx].append((c_idx, pairs))
        self._num_children = [len(self._children[i]) for i in range(n)]

        # Region-recolor support (Plan §7.2). ``_edge_pairs[(p, c)]`` is the
        # compatible ``(p_div, c_div)`` set on the edge p->c; ``_children_idx``
        # lists each op's children by index (deterministic flood order).
        self._children_idx = [sorted(c for c, _ in self._children[i]) for i in range(n)]
        self._edge_pairs: dict[tuple[int, int], frozenset] = {
            (i, c): pairs for i in range(n) for c, pairs in self._children[i]
        }
        # Non-trivial (split) menu indices per op -- the only legal recolor
        # anchors, so recolor stays a coordinated *splitting* move and leaves
        # undividing to atomic flips (Plan §7.2). Anchor candidates are the ops
        # that have at least one.
        self._nontrivial_menu = [
            sorted(
                j for j, cd in enumerate(b.core_divisions) if cd.output_partition > 1
            )
            for b in bufs
        ]
        self._anchor_candidates = [i for i in range(n) if self._nontrivial_menu[i]]

    # -- division-dependent derivations --------------------------------------

    def _per_core_size(self, idx: int, div_idx: int) -> int:
        """Per-core footprint of buffer ``idx`` under menu index ``div_idx``:
        ``ceil(total_size / output_partition)`` (Plan §2.2)."""
        part = self._bufs[idx].core_divisions[div_idx].output_partition
        return math.ceil(self._bufs[idx].size / part)

    def _eligible(self, idx: int) -> bool:
        """Whether buffer ``idx`` may be LX-resident under the current ``W``
        (the three division-dependent gates of Plan §7.4, mirroring
        ``DfsLayoutSolver._evaluate``): the fixed residency pin, a per-core
        footprint that fits at all, and a division carrying a compatible
        ``cd_parent_matches`` pair on *every* child edge -- the sole compatibility
        gate; a division missing a pair on any edge is gated out. Those pairs are
        per-core-view / core-count based (not ``is_clean`` based): a reduction
        split can appear as a *consumer* index (a K-split reading a clean parent
        via the PSUM ring), but a reduction-split *producer* writes a partial sum
        the substrate never lets a child read from LX, so it never appears as a
        parent index -- such a buffer is always gated out here."""
        b = self._bufs[idx]
        if not b.residency_allowed:
            return False
        if self._per_core_size(idx, self.chosen[idx]) > self.limit:
            return False
        ci = self.chosen[idx]
        return all(
            (ci, self.chosen[c_idx]) in pairs for c_idx, pairs in self._children[idx]
        )

    # -- seed ----------------------------------------------------------------

    def _lifetime_buffers(self, sizes: list[int]) -> list[LifetimeBoundBuffer]:
        """Plain lifetime buffers (name/size/lifetime/in-place) the packer and
        FirstFit consume; ``sizes`` are the current per-core footprints. In-place
        parents are kept in full -- the packer treats an ineligible (absent)
        parent transparently, so it never in-places onto one."""
        out = []
        for i, b in enumerate(self._bufs):
            out.append(
                LifetimeBoundBuffer(
                    name=b.name,
                    size=sizes[i],
                    uses=list(b.uses),
                    first_use_is_read=b.first_use_is_read,
                    in_place_parents=[
                        p for p in b.in_place_parents if p in self._name_to_idx
                    ],
                )
            )
        return out

    def _build_seed_packer(self) -> Packer:
        """Build the packer for the seed state: per-core sizes at index 0, a
        FirstFit-derived ``pi`` (Plan §8.2), and the seed eligibility mask."""
        n = len(self._bufs)
        sizes = [self._per_core_size(i, 0) for i in range(n)]
        eligible = [self._eligible(i) for i in range(n)]

        # pi from a FirstFit pass over the eligible buffers (ineligible ones are
        # marked unplaceable so they sort last, matching their HBM residency).
        ff_bufs = self._lifetime_buffers(sizes)
        for i in range(n):
            if not eligible[i]:
                ff_bufs[i].placement = False
        pi = SolverToPermutation(
            FirstFitLayoutSolver(self.limit, self.alignment)
        ).permutation(ff_bufs)

        return make_permutation_packer(
            self._lifetime_buffers(sizes),
            pi,
            self.limit,
            self.alignment,
            eligible=eligible,
        )

    # -- scoring (shared scorer; lower is better) ----------------------------

    def _score(self) -> int:
        """The shared objective for the current state, in integer fixed-point
        time units. A buffer with a packer address is LX-resident (its address is
        ``None`` iff ineligible or spilled), so traffic is ``boundary_cost`` when
        resident else ``num_children * size + spill_write_cost`` -- the substrate
        HBM-traffic model (Plan §7.1). The node term is zero on the fake (no op
        metadata); the scorer's fixed-point conversion keeps this deterministic.
        """
        traffic = 0
        for i, b in enumerate(self._bufs):
            if self.packer.addresses[i] is not None:
                traffic += b.boundary_cost
            else:
                traffic += self._spill_cost(b, self._num_children[i])
        memory_fixed = scorer.to_fixed_us(traffic / scorer.hbm_bytes_per_us())
        node_fixed = 0  # no op-kind metadata in the fake substrate (Phase 6)
        return memory_fixed + node_fixed

    # -- moves ---------------------------------------------------------------

    def _flippable(self) -> list[int]:
        """Buffer indices whose division menu offers an alternative (>1 entry)."""
        return [
            i for i in range(len(self._bufs)) if len(self._bufs[i].core_divisions) > 1
        ]

    def _atomic_flip(self, idx: int, new_div: int) -> None:
        """Change buffer ``idx``'s division to ``new_div`` and ripple: resize its
        per-core footprint, then refresh eligibility for ``idx`` and its parents --
        the only buffers whose LX-feasibility a single flip can change (a buffer's
        eligibility depends on its own division and its children's, so flipping
        ``idx`` touches ``idx`` and every op that has ``idx`` as a child)."""
        self.chosen[idx] = new_div
        self.packer.resize(idx, self._per_core_size(idx, new_div))
        for x in sorted({idx} | self._parents_idx[idx]):
            self.packer.set_eligible(x, self._eligible(x))

    def _flood_region(self, anchor: int, tiling: int) -> dict[int, int]:
        """Flood the ``cd_parent_matches`` relation from ``(anchor, tiling)`` to a
        menu-index assignment over the reachable region (Plan §7.2).

        Bidirectional: from an assigned op ``u`` (index ``iu``), a child ``c`` joins
        at the smallest ``ic`` with ``(iu, ic)`` compatible, and a parent ``p`` at
        the smallest ``ip`` with ``(ip, iu)`` compatible. The reachable set *is* the
        region; boundaries emerge for free (no compatible index across an edge).
        First-assignment-wins with a min-index frontier and sorted candidates makes
        this deterministic and independent of ``cd_parent_matches`` list order
        (Plan §7.5); a join reached with no compatible index simply is not
        extended -- its edge becomes an accepted internal seam, never a failure.
        """
        assignment = {anchor: tiling}
        heap = [anchor]
        while heap:
            u = heapq.heappop(heap)
            iu = assignment[u]
            for c in self._children_idx[u]:  # down: u -> c
                if c in assignment:
                    continue
                cands = sorted(ic for ip, ic in self._edge_pairs[(u, c)] if ip == iu)
                if cands:
                    assignment[c] = cands[0]
                    heapq.heappush(heap, c)
            for p in sorted(self._parents_idx[u]):  # up: p -> u
                if p in assignment:
                    continue
                cands = sorted(ip for ip, ic in self._edge_pairs[(p, u)] if ic == iu)
                if cands:
                    assignment[p] = cands[0]
                    heapq.heappush(heap, p)
        return assignment

    def _apply_recolor(self, assignment: dict[int, int]) -> None:
        """Commit a flooded region coloring: set every region op's division, resize
        its footprint, and refresh eligibility for the region plus the parents of
        region ops (the same ripple as a flip, unioned over the region)."""
        for op, div in assignment.items():
            self.chosen[op] = div
        affected = set(assignment)
        for op in sorted(assignment):
            self.packer.resize(op, self._per_core_size(op, self.chosen[op]))
            affected |= self._parents_idx[op]
        for x in sorted(affected):
            self.packer.set_eligible(x, self._eligible(x))

    def _recolor(self) -> None:
        """One region-recolor move: a size-proportional anchor (uniform op, so a
        region is hit ∝ its op-count), a random non-trivial anchor tiling, flood,
        then recolor + burst."""
        anchor = self._rng.choice(self._anchor_candidates)
        tiling = self._rng.choice(self._nontrivial_menu[anchor])
        assignment = self._flood_region(anchor, tiling)
        self._last_recolor_region_size = len(assignment)
        self._last_recolor_anchor_partition = (
            self._bufs[anchor].core_divisions[tiling].output_partition
        )
        self._apply_recolor(assignment)
        self._burst()

    def _burst(self) -> None:
        """A short cold layout burst: greedily accept O(1) adjacent swaps that do
        not lower the packer's layout quality, letting ``pi`` adapt to the new
        footprints before the compound move is judged (Plan §4.4). ``swap`` is its
        own inverse, so a non-improving step is reverted in place."""
        n = len(self._bufs)
        if n < 2:
            return
        burst_len = max(1, int(self._burst_fraction * n))
        for _ in range(burst_len):
            i = self._rng.randrange(n - 1)
            if self.packer.swap(i) < 0:
                self.packer.swap(i)  # revert (self-inverse)

    # -- annealing loop ------------------------------------------------------

    def _snapshot(self):
        return (self.packer.copy(), list(self.chosen))

    def _restore(self, snap) -> None:
        self.packer, self.chosen = snap[0], snap[1]

    def _calibrate_temperature(self) -> float:
        """A crude scale estimate: the *median* absolute score delta over a sample
        of random (fixed-weight) moves from the current state. Serves as the crude
        schedule's ``T0`` and the reheating schedule's pre-snap seed center. The
        median (not mean) is robust to region-recolor's large deltas. Falls back to
        1.0 when nothing moved. Restores state; consumes RNG deterministically."""
        base = self._score()
        deltas: list[int] = []
        for _ in range(min(64, 4 * len(self._bufs) + 8)):
            snap = self._snapshot()
            self._execute_move(self._choose_move_crude())
            d = abs(self._score() - base)
            if d > 0:
                deltas.append(d)
            self._restore(snap)
        return float(statistics.median(deltas)) if deltas else 1.0

    # -- move selection & execution -----------------------------------------

    def _applicable_moves(self) -> list[str]:
        """Move types available this step, in fixed (deterministic) order: reorder
        needs >=2 buffers, flip a multi-entry menu, recolor a non-trivial anchor."""
        moves = []
        if len(self._bufs) >= 2:
            moves.append("reorder")
        if self._flippable_ops:
            moves.append("flip")
        if self._anchor_candidates:
            moves.append("recolor")
        return moves

    def _weighted_choice(self, choices: list[tuple[str, float]]) -> str:
        """Deterministic weighted pick over ``choices`` (in fixed order)."""
        total = sum(w for _, w in choices)
        r = self._rng.random() * total
        acc = 0.0
        for name, weight in choices:
            acc += weight
            if r < acc:
                return name
        return choices[-1][0]

    def _choose_move_crude(self) -> str:
        """Fixed-weight move choice (the crude schedule and the calibrator)."""
        applicable = self._applicable_moves()
        if not applicable:
            return "none"
        w = {
            "reorder": self._reorder_weight,
            "flip": self._flip_weight,
            "recolor": self._recolor_weight,
        }
        return self._weighted_choice([(m, w[m]) for m in applicable])

    @staticmethod
    def _hotness(name: str, phi: float) -> float:
        """Structural moves are hot early (cycle phase near 0), layout reorders
        late (near 1) -- Plan §5.4."""
        return phi if name == "reorder" else 1.0 - phi

    def _choose_move_reheating(self, phi: float) -> str:
        """Cycle-phase proposal mix (Plan §5.4): weight each applicable move by its
        neighborhood size times ``max(w_floor, hotness(m, phi))``, so structural
        moves dominate hot phases and layout reorders dominate cold ones."""
        applicable = self._applicable_moves()
        if not applicable:
            return "none"
        choices = [
            (m, self._neighborhoods[m] * max(self._weight_floor, self._hotness(m, phi)))
            for m in applicable
        ]
        return self._weighted_choice(choices)

    def _execute_move(self, name: str) -> None:
        """Apply move ``name`` in place; structural moves carry their own burst."""
        n = len(self._bufs)
        if name == "reorder":
            self.packer.rotate(self._rng.randrange(n), self._rng.randrange(n))
        elif name == "flip":
            idx = self._rng.choice(self._flippable_ops)
            menu = len(self._bufs[idx].core_divisions)
            offset = self._rng.randrange(1, menu)  # a different index, wrap-around
            self._atomic_flip(idx, (self.chosen[idx] + offset) % menu)
            self._burst()
        elif name == "recolor":
            self._recolor()
        # "none": no applicable move; no-op.

    # -- instrumentation -----------------------------------------------------

    def _record_move_scale(self, name: str, scale: float) -> None:
        """Fold a nonzero ``|dE|`` into ``name``'s within-group stats (Plan §5.3)."""
        if scale > 0.0:
            self._ms_n[name] += 1
            self._ms_sum[name] += scale
            self._ms_sqsum[name] += scale * scale

    def move_scale_cv(self) -> dict[str, float]:
        """Coefficient of variation (std / mean) of ``|dE|`` within each move type
        -- the §5.3 signal for whether a type should be split into size buckets.
        ``0.0`` for a type with < 2 samples or a zero mean."""
        out: dict[str, float] = {}
        for m, n in self._ms_n.items():
            if n < 2:
                out[m] = 0.0
                continue
            mean = self._ms_sum[m] / n
            var = max(0.0, self._ms_sqsum[m] / n - mean * mean)
            out[m] = (math.sqrt(var) / mean) if mean > 0.0 else 0.0
        return out

    def _record_recolor(self, name: str, accepted: bool) -> None:
        if name != "recolor":
            return
        self.recolor_region_sizes.append(self._last_recolor_region_size)
        self.recolor_anchor_partitions.append(self._last_recolor_anchor_partition)
        if accepted:
            self.recolor_accepted_partitions.append(self._last_recolor_anchor_partition)

    # -- annealing loop ------------------------------------------------------

    def _step(self, name: str, temperature: float, cur: int) -> tuple[int, bool, float]:
        """Execute one judged move: propose ``name``, apply the Metropolis test
        against ``temperature``, and update best-seen + instrumentation. Returns
        ``(new_cur, accepted, |dE|)``."""
        snap = self._snapshot()
        self._execute_move(name)
        self.moves_proposed[name] += 1
        new = self._score()
        delta = new - cur
        scale = float(abs(delta))
        # `or` short-circuits, so the RNG is drawn only when delta > 0.
        accepted = delta <= 0 or self._rng.random() < math.exp(-delta / temperature)
        self._record_move_scale(name, scale)
        if accepted:
            self.moves_accepted[name] += 1
            cur = new
            if cur < self._best_score:
                self._best_score = cur
                self._best_snap = self._snapshot()
                if name == "recolor":
                    self.recolor_improved += 1
        else:
            self._restore(snap)
        self._record_recolor(name, accepted)
        return cur, accepted, scale

    def _anneal(self) -> None:
        n = len(self._bufs)
        steps = max(self._min_steps, self._steps_per_buffer * n)
        self._flippable_ops = self._flippable()
        # Static proposal-mix neighborhoods (Plan §5.4): reorder ~ n reinsertion
        # points; flip ~ the available local labels; recolor ~ the anchor tilings
        # (n_regions x n_colors, with the anchor's non-trivial menu size as
        # n_colors, §7.2).
        self._neighborhoods = {
            "reorder": n,
            "flip": sum(
                len(self._bufs[i].core_divisions) - 1 for i in self._flippable_ops
            ),
            "recolor": sum(
                len(self._nontrivial_menu[a]) for a in self._anchor_candidates
            ),
        }

        cur = self._score()
        # Baseline = the seed state's score; best-seen never rises above it, the
        # >=-baseline guarantee (Plan §8.1).
        self.baseline_score = cur
        self._best_score = cur
        self._best_snap = self._snapshot()

        if self._applicable_moves():
            if self._nested:
                self._anneal_nested(steps, cur)
            elif self._schedule == "crude":
                self._anneal_crude(steps, cur)
            else:
                self._anneal_reheating(steps, cur)

        self.best_score = self._best_score
        self._restore(self._best_snap)

    def _anneal_crude(self, steps: int, cur: int) -> None:
        """Phase-3/4 baseline: one geometric cool + fixed proposal weights. Kept as
        the A/B the reheating schedule must beat."""
        t0 = self._calibrate_temperature()
        t_end = max(t0 / 1000.0, 1e-9)
        for step in range(steps):
            frac = step / (steps - 1) if steps > 1 else 1.0
            temperature = t0 * (t_end / t0) ** frac
            cur, _, _ = self._step(self._choose_move_crude(), temperature, cur)

    def _anneal_reheating(self, steps: int, cur: int) -> None:
        """Plan §5: the multi-move self-calibrating reheating schedule with the
        cycle-phase proposal mix. The schedule self-calibrates each move type's
        band from its streamed ``|dE|``, seeded (pre-snap) from a crude median."""
        schedule = SelfCalibratingReheatingSchedule(
            bands=self._move_bands,
            total_steps=steps,
            cycles=self._cycles,
            horizons_per_cycle=self._horizons_per_cycle,
            seed_center=self._calibrate_temperature(),
        )
        schedule.reset()
        while not schedule.finished:
            name = self._choose_move_reheating(schedule.cycle_phase())
            cur, accepted, scale = self._step(name, schedule.temperature(name), cur)
            schedule.update(accepted, scale, name)

    # -- nested two-timescale loop (experimental) ----------------------------

    def _choose_structural_move(self) -> str:
        """Outer-loop move: flip or recolor by weight (no standalone reorder --
        layout lives in the inner loop). ``none`` if no structural move exists."""
        moves = []
        if self._flippable_ops:
            moves.append("flip")
        if self._anchor_candidates:
            moves.append("recolor")
        if not moves:
            return "none"
        w = {"flip": self._flip_weight, "recolor": self._recolor_weight}
        return self._weighted_choice([(m, w[m]) for m in moves])

    def _apply_structural(self, name: str) -> None:
        """Apply a structural move's division change + ripple, WITHOUT the burst
        (the inner layout loop replaces it)."""
        if name == "flip":
            idx = self._rng.choice(self._flippable_ops)
            menu = len(self._bufs[idx].core_divisions)
            offset = self._rng.randrange(1, menu)
            self._atomic_flip(idx, (self.chosen[idx] + offset) % menu)
        elif name == "recolor":
            anchor = self._rng.choice(self._anchor_candidates)
            tiling = self._rng.choice(self._nontrivial_menu[anchor])
            assignment = self._flood_region(anchor, tiling)
            self._last_recolor_region_size = len(assignment)
            self._last_recolor_anchor_partition = (
                self._bufs[anchor].core_divisions[tiling].output_partition
            )
            self._apply_recolor(assignment)

    def _inner_len(self, progress: float, acc: int, prop: int, n: int) -> int:
        """Inner-loop length (in swap steps) for the current outer progress. Grows
        from ``inner_len_base * n`` to ``inner_len_max * n`` along ``inner_curve``:
        constant/linear/convex in outer progress, or adaptive in the structural
        *reject* rate (invest more in layout as structure stops moving)."""
        base, mx = self._inner_len_base, self._inner_len_max
        if self._inner_curve == "constant":
            f = base
        elif self._inner_curve == "linear":
            f = base + (mx - base) * progress
        elif self._inner_curve == "convex":
            f = base + (mx - base) * progress * progress
        else:  # adaptive
            reject = 1.0 - (acc / prop) if prop > 0 else 0.0
            f = base + (mx - base) * reject
        return max(1, int(f * n))

    def _calibrate_inner_qtemp(self) -> float:
        """Median |quality delta| over a sample of layout reinsertions -- the
        annealed inner loop's (constant) temperature, in packer-quality units.
        Restores state; consumes RNG deterministically."""
        n = len(self._bufs)
        if n < 2:
            return 1.0
        deltas = []
        for _ in range(min(64, 4 * n + 8)):
            i = self._rng.randrange(n)
            j = self._rng.randrange(n)
            dq = self.packer.rotate(i, j)
            self.packer.rotate(j, i)  # revert (pop-i-insert-j inverse)
            if dq != 0.0:
                deltas.append(abs(dq))
        return float(statistics.median(deltas)) if deltas else 1.0

    def _inner_layout_loop(self, steps: int, qtemp: float) -> int:
        """Warm-started inner layout anneal: ``steps`` single-buffer reinsertions
        (``rotate`` -- an arbitrary-position move, far faster-mixing than adjacent
        swaps) on the current packer. Greedy-cold (keep only non-worsening) or
        annealed (Metropolis on the quality delta at ``qtemp``, tracking + restoring
        the best-quality layout). Returns the number of steps taken."""
        n = len(self._bufs)
        steps = int(steps)
        if n < 2 or steps < 1:
            return 0
        best_q = self.packer.quality()
        best_packer: Optional[Packer] = None
        for _ in range(steps):
            i = self._rng.randrange(n)
            j = self._rng.randrange(n)
            dq = self.packer.rotate(i, j)  # quality delta (higher is better)
            if self._inner_annealed:
                keep = dq >= 0 or (
                    qtemp > 0 and self._rng.random() < math.exp(dq / qtemp)
                )
                if not keep:
                    self.packer.rotate(j, i)  # revert
                elif self.packer.quality() > best_q:
                    best_q = self.packer.quality()
                    best_packer = self.packer.copy()
            elif dq < 0:
                self.packer.rotate(j, i)  # greedy revert
        if best_packer is not None:
            self.packer = best_packer
        return steps

    def _anneal_nested(self, budget: int, cur: int) -> None:
        """Outer SA over structure; each proposal runs an inner layout loop, is
        judged on the full score (end + early-abandon), and a final pure-layout
        polish refines the best structure. Layout warm-starts across structural
        moves (pi persists, Plan §2.2)."""
        n = len(self._bufs)
        qtemp = self._calibrate_inner_qtemp()
        polish = int(self._polish_frac * budget)
        outer_budget = max(0, budget - polish)

        # No structural move available -> spend the whole budget polishing layout.
        if self._choose_structural_move() == "none":
            self._inner_layout_loop(budget, qtemp)
            best = self._score()
            if best < self._best_score:
                self._best_score = best
                self._best_snap = self._snapshot()
            return

        t0 = self._calibrate_temperature()
        t_end = max(t0 / 1000.0, 1e-9)
        spent = prop = acc = 0
        while spent < outer_budget:
            name = self._choose_structural_move()
            if name == "none":
                break
            progress = spent / outer_budget if outer_budget else 1.0
            temperature = t0 * (t_end / t0) ** progress
            snap = self._snapshot()
            self._apply_structural(name)
            target = self._inner_len(progress, acc, prop, n)
            # End + early-abandon: run a burn-in, peek the score, and skip the
            # inner tail if the move is hopelessly worse at this temperature.
            burn = max(1, target // 4) if self._early_abandon else target
            used = self._inner_layout_loop(burn, qtemp)
            if self._early_abandon and target > burn:
                if self._score() - cur <= self._abandon_k * temperature:
                    used += self._inner_layout_loop(target - burn, qtemp)
            new = self._score()
            delta = new - cur
            self.moves_proposed[name] += 1
            self._record_move_scale(name, float(abs(delta)))
            prop += 1
            accepted = delta <= 0 or self._rng.random() < math.exp(-delta / temperature)
            if accepted:
                self.moves_accepted[name] += 1
                acc += 1
                cur = new
                if cur < self._best_score:
                    self._best_score = cur
                    self._best_snap = self._snapshot()
                    if name == "recolor":
                        self.recolor_improved += 1
            else:
                self._restore(snap)
            self._record_recolor(name, accepted)
            spent += 1 + used

        # Final polish: a long pure-layout anneal on the best structure found.
        if polish > 0:
            self._restore(self._best_snap)
            self._inner_layout_loop(polish, qtemp)
            polished = self._score()
            if polished < self._best_score:
                self._best_score = polished
                self._best_snap = self._snapshot()

    # -- write-back ----------------------------------------------------------

    def _write_back(self) -> None:
        """Commit the best state to the buffers and record spill causes."""
        for i, b in enumerate(self._bufs):
            addr = self.packer.addresses[i]
            b.chosen_division = self.chosen[i]
            b.address = addr
            if addr is None:
                self.spill_reasons[b.name] = b.residency_reason or _SOLVER_CHOSE_SPILL
