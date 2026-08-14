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

"""Joint work-division + LX-layout simulated-annealing engine.

``SaCoOptimizingSolver`` is a third co-optimization engine alongside the
substrate's CP-SAT and DFS solvers. It anneals the joint state ``(pi, W)``:

* ``pi`` -- the layout permutation, held in a *composed* (not subclassed)
  :class:`PermutationBasedLayoutSolver` packer, because this loop mixes move
  types and scores a richer objective than the packer's own ``quality()``.
* ``W`` -- the work division, one ``chosen_division`` menu index per buffer.

Moves are reorder, atomic division flip, and region-recolor; each structural
move runs as a compound move+burst judged as a unit by one Metropolis test.
Region-recolor floods the ``cd_parent_matches`` relation bidirectionally from a
non-trivial (split) anchor tiling, so the region *is* the flood's reach and
boundaries emerge for free; an edge with no compatible index becomes an accepted
internal seam. The uniform flood is a deliberately unbalanced proposal -- it
ratchets toward homogeneous regions -- and the recolor instrumentation exists to
detect whether that loses heterogeneous optima; balanced alternatives
(beta-biased local proposals, block-Gibbs over tree-like regions) stay unbuilt
until it does.

Best-seen over ``(pi, W)`` from the seed state (every op at index 0, ``pi`` from
FirstFit) keeps every returned state no worse than that baseline, which is what
lets each piece of the search be added and validated on its own.

Objectives, schedules, and the per-knob tuning evidence:
``docs/source/compiler/sa_co_optimization.md`` and
``docs/source/compiler/benchmarks/``.

Determinism: a seeded ``Random`` over index-ordered domains and the integer
fixed-point score make a run bit-for-bit reproducible.
"""

from __future__ import annotations

import copy
import heapq
import math
import random as rnd
import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from torch_spyre._inductor.scratchpad.cooling_schedules import (
    SelfCalibratingReheatingSchedule,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import SolverToPermutation
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    ceil_div,
)
from torch_spyre._C import NativePermutationLayoutSolver
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    make_permutation_packer,
)
from torch_spyre._inductor.scratchpad import cooptimization_scorer as scorer
from torch_spyre._inductor.logging_utils import get_inductor_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective

logger = get_inductor_logger("scratchpad.sa_cooptimizer")


def _bundle_objective(
    buffers: Sequence["CoreDivisionBuffer"],
) -> Optional["BundleCostObjective"]:
    """Build a :class:`BundleCostObjective` from the live Inductor graph.

    Per-division ``OpFeatures`` and the fused-bundle grouping are read off
    ``V.graph``, which is ambient while the allocator runs -- this solver *is* an
    Inductor pass. Only the buffer order comes from the arguments.

    Returns ``None`` when there is no live graph, the normal case for a run
    driven from a serialized capture; the caller then falls back to the
    memory-only objective and logs it.
    """
    from torch._inductor.virtualized import V

    from torch_spyre._inductor.fusion import estimate_bundles
    from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective
    from torch_spyre._inductor.scratchpad.op_features import features_for_menu

    # Unset, ``V.graph`` is a ``NullHandler`` rather than ``None``, so detect the
    # live graph by what this needs from it rather than by identity.
    graph: Any = getattr(V, "graph", None)
    if graph is None:
        return None
    if not hasattr(graph, "operations") or not hasattr(graph, "get_buffer"):
        return None

    features: dict[str, list] = {}
    for buf in buffers:
        try:
            op = graph.get_buffer(buf.name)
        except Exception:  # noqa: BLE001 - not every solver buffer is a graph buffer
            continue
        features[buf.name] = features_for_menu(op, buf.core_divisions)
    bundles = [
        [op.get_name() for op in group] for group in estimate_bundles(graph.operations)
    ]
    return BundleCostObjective([b.name for b in buffers], features, bundles)


# ``make_permutation_packer`` returns either the pure-Python or the native C++
# packer. Use ``.quality()`` (not the Python-only ``total_quality`` attribute) so
# both work.
Packer = Union[PermutationBasedLayoutSolver, NativePermutationLayoutSolver]

# Cause recorded for a buffer the SA engine left out of LX.
_SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"

# Per-move-type acceptance bands (accept_hi, accept_lo) for the reheating
# schedule: reorders warmest, region-recolor the coldest floor so it freezes
# earliest. Guessed, not swept.
_DEFAULT_MOVE_BANDS = {
    "reorder": (0.6, 0.02),
    "flip": (0.3, 0.005),
    "recolor": (0.1, 0.001),
}


class SaCoOptimizingSolver(CoreDivisionLayoutSolver):
    """SA joint core-division + LX-placement engine.

    **Argument dependencies.** Most arguments are gated by another's value, and a
    gated one is silently ignored rather than rejected. Diagrams:
    ``docs/source/compiler/sa_co_optimization.md``. Live sets::

        always          seed, steps_per_buffer, min_steps, max_steps,
                        cost_objective, trace_every, nested
        nested=False    schedule; burst_fraction, burst_fractions, burst_move;
          (default)     reorder_move (+ sweep_biased_i, sweep_cleanup unless
                        reorder_move="random")
          schedule=       "crude"      reorder_weight, flip_weight, recolor_weight
                          "reheating"  cycles, horizons_per_cycle, weight_floor,
                                       move_bands, reorder_neighborhood_scale
        nested=True     flip_weight, recolor_weight, inner_len_base, inner_curve
                        (+ inner_len_max unless inner_curve="constant"),
                        inner_annealed, early_abandon (+ abandon_k if set),
                        polish_frac -- and ``schedule`` and every burst/reorder
                        knob above are ignored

    ``node_term`` is separate: it enters only the memory-only objective, so it is
    dead unless ``cost_objective`` resolves to ``None`` (see :meth:`_score`). Flip
    knobs need a buffer with a multi-entry menu and recolor knobs a non-trivial
    anchor, so both go dead on a graph offering neither.

    A gated argument is still not a bit-for-bit no-op: the T0 calibration probe
    (:meth:`_calibrate_temperature`) draws crude-weighted moves and runs bursts in
    every mode, so changing a dormant knob shifts T0 and the RNG stream. An A/B on
    one measures noise, not nothing.

    Benchmark files named below live in ``docs/source/compiler/benchmarks/``.

    Args:
        buffers: the buffers to plan, in the allocator's order. Declared as
            ``Sequence[LifetimeBoundBuffer]`` so the class itself satisfies
            ``CoreDivisionSolverFactory`` (``Callable`` parameters are
            contravariant, so a narrower annotation would not), but every buffer
            passed must be a :class:`CoreDivisionBuffer` -- the engine reads the
            ``core_divisions`` menu and the ``cd_parent_matches`` relation off
            each one.

            **Mutated in place, and their order is an index.** The returned list
            is these same objects with ``chosen_division`` and ``address``
            written back, so a caller needing the input preserved must copy first
            (the benchmarks ``deepcopy``). Position ``i`` is the index used by
            ``chosen``, by the packer's permutation, and by the cost objective.
            Solvers are single-use: construct a fresh one per buffer set.
        size: scratchpad capacity in bytes.
        alignment: placement alignment (128 = one Spyre stick).
        seed: RNG seed; fixes the (deterministic) search trajectory.
        steps_per_buffer: annealing steps scale linearly with the buffer count
            (a crude bounded budget).
        min_steps: floor on the step budget for tiny graphs.
        max_steps: ceiling on the *total* step budget, so a large graph cannot
            run away. Sits above the layout-only annealer's clamp
            (``SelfCalibratingReheatingSchedule.max_steps``, 5_000) since this
            engine searches divisions too. Bounds *steps*, not wall-clock.
        schedule: ``"crude"`` (default -- one geometric cool with fixed proposal
            weights) or ``"reheating"`` (the multi-move self-calibrating
            reheating schedule + cycle-phase proposal mix). Ignored entirely when
            ``nested=True``, which brings its own outer schedule.

            Crude is the default on the score/CPU frontier, not on score: at
            matched wall-clock the two tie on score and crude's steps cost
            0.59-0.76x, because it proposes ~50% reorders where reheating
            proposes ~46% recolors and a recolor dirties far more bundles under
            the cost objective (``coopt_schedule_default.md``).

            Reheating still converges in fewer steps -- median 40 vs 82 to come
            within 1% of the best any arm reaches (``coopt_convergence.md``) --
            and is worth up to 23% at budgets below convergence. So do not drop
            it, and do not lower ``steps_per_buffer`` without re-checking this:
            most of the headroom between ``spb`` 20 and 40 is what makes the
            schedule choice free.
        cycles / horizons_per_cycle: reheating-schedule knobs, so dormant under
            the default ``schedule``. The cycle count
            changes the result on 3 of 11 corpus graphs and the default is at
            most 0.35% off the best count anywhere; below convergence the spread
            is noise, not a ranking (``coopt_cycle_sweep.md``).
        weight_floor: the ``w_floor`` in the cycle-phase proposal mix, so no
            applicable move type is ever fully starved.
        move_bands: per-move-type ``(accept_hi, accept_lo)`` acceptance bands for
            the reheating schedule; defaults to :data:`_DEFAULT_MOVE_BANDS`.
        reorder_weight / flip_weight / recolor_weight: fixed proposal weights for
            the ``"crude"`` schedule, i.e. the live path. Nested mode reads the
            flip/recolor pair for its outer move choice and ignores
            ``reorder_weight``.

            The defaults land on the score/CPU frontier; reorder-only is the one
            clear mistake, +0.523% for 0.86x the CPU
            (``coopt_move_weights.md``). Before retuning: the whole spread is
            ~0.1% of score against a cost model unvalidated on hardware, and
            these are not independent of ``schedule`` -- giving crude the
            reheating mix reproduces reheating's behaviour, so the schedule knob
            is largely a mix knob.
        burst_fraction: layout-burst length as a fraction of the buffer count,
            applied to both structural moves; the burst warms ``pi`` to the new
            footprints before the compound move is judged. ``0.0`` disables it.
            Nested mode never uses it -- its inner layout loop replaces it.

            Measured **inert** on this corpus: no length differs significantly
            from any other, under either primitive, from zero up to a burst
            costing 30% of the budget, and structural acceptance moves by only
            ~1.5% across every arm (``coopt_burst.md``). The default is the
            cheap end of that flat surface, not a measured winner; expect a
            corpus that discriminates to move it.
        burst_move: the burst's layout primitive, so dead at
            ``burst_fraction=0`` -- ``"rotate"`` (default, an arbitrary
            reinsertion, what ``reorder`` uses) or ``"swap"`` (the O(1) adjacent
            exchange the burst originally used). Rotate was adopted to
            test whether the swap burst's inertness was the primitive's fault; it
            was not. Retained as the control arm.
        burst_fractions: per-move override, ``{"flip": f, "recolor": f}``, merged
            over ``burst_fraction`` so a partial override keeps the shared value.

            Exists because a flip moves one op where a recolor moves a whole
            region, and the warm layout stops transferring at large division
            changes (``warm_start_transfer.md``) -- so recolor plausibly wants a
            different burst. Measured, the asymmetry is not resolvable (best
            split arm beats the best shared value by 0.004%).
        node_term: add the cross-core reduction (PSUM-ring) cost of the chosen
            divisions to the objective. Off by default, and a **no-op unless
            ``cost_objective`` resolves to ``None``** -- it is a term of the
            memory-only objective, which the cost model replaces outright.

            Only the reduction half of the node term is computable from a buffer;
            the matmul half needs the producing op's b/m/n/k extents (see
            :meth:`_precompute_node_costs`). That makes this one-sided -- it
            charges for splitting a reduction axis with no modelled reward -- so
            the search duly drives every reduction split to zero. It is an
            instrument, not a better objective; landing the matmul half is what
            would make it a default.

            A bool rather than a scale: the intended pairing, an external cost
            model's ``macs / cores`` matmul term, distinguishes neither
            ``reduction_cores`` nor a K-split from an output split at fixed core
            count, so any positive coefficient yields the same decisions and a
            magnitude is not identifiable. Stays at the hardware-fitted
            ``_PSUM_PER_CORE_ELEM_US``.
        cost_objective: which objective the search minimizes.

            * ``"bundle"`` (default) -- build a :class:`BundleCostObjective` here
              off the live Inductor graph, so this form needs nothing from the
              caller. That is what lets it be the default: the allocator's
              ``CoreDivisionSolverFactory`` passes only
              ``(buffers, size, alignment)``, so an instance could never arrive
              that way. With no live graph it logs and falls back to memory-only.
            * ``None`` -- the memory-only spill-traffic objective, explicitly.
            * an instance -- used as given; how the benchmarks drive captures.

            The bundle plans are 18-57% cheaper on 9 of 11 corpus graphs and
            worse on none (``coopt_cost_objective.md``), but that is the model
            grading its own plans -- **no device time has been measured** -- and
            the memory objective's ``best <= baseline`` guarantee on *traffic*
            does not carry over, since this objective trades residency away for
            divisions. Pass ``None`` for the old behaviour.
        reorder_move: how a ``reorder`` proposal picks its reinsertion position.
            Single-loop modes only -- nested mode proposes no standalone reorder.

            * ``"sweep_quality"`` (default) -- the layout-only annealer's
              best-first sweep: lift one buffer, probe every reinsertion
              position, try them in descending packer ``quality()`` order.
              Ranking by the ``quality()`` proxy is deliberate: it breaks ties
              among the many score-identical positions (reorder acceptance runs
              at 96-100%) and steers ``pi`` toward states a later structural move
              can exploit.
            * ``"sweep_score"`` -- the same sweep ranked by the true objective at
              every position. Exact, but buys nothing over the proxy at up to
              1.4x per step.
            * ``"random"`` -- a single random ``(i, j)`` rotation, the A/B
              baseline.

            Under the cost objective the sweep is indistinguishable from
            ``random`` at every budget, and costs nothing either
            (``coopt_reorder_move.md``); its earlier ~3% win was under the
            memory-only objective, whose step-function shape gave it far more
            ties to break. A knob with no measured consequence in either
            direction -- prefer ``"random"`` if simplicity is worth anything.
        reorder_neighborhood_scale: multiplier on ``reorder``'s neighborhood size
            in the reheating schedule's cycle-phase proposal mix, so dormant under
            the default ``schedule``. Leave it at 1.

            Reheating weights each move by its neighborhood, and ``reorder``'s
            (``n``) is dwarfed by the flip/recolor menus, so it spends only 7-12%
            of proposals on reorder against ``crude``'s ~50%. Raising this does
            not recover the difference: promising arms collapsed to noise on
            held-out seeds (``coopt_band_retune_scale.md``, ``..._validate.md``).
        sweep_biased_i: bias the sweep's choice of which buffer to lift toward
            ones that are not fully allocated (the layout-only annealer's
            weighting). On by default: turning it off erases the sweep's
            advantage. Dead at ``reorder_move="random"``, which runs no sweep.
        sweep_cleanup: run a placement-neutral tidy pass after an accepted sweep
            rotation, sorting adjacent non-overlapping buffers into address
            order. Off by default: O(n) swaps per accepted move with quadratic
            worst-case backtracking, and no measured benefit. Dead at
            ``reorder_move="random"``.
        nested: run the two-timescale loop (:meth:`_anneal_nested`) instead of
            the single loop. The outer anneal proposes only *structural* moves
            (flip/recolor); each is followed by an inner layout loop that
            re-adapts ``pi`` to the new footprints, and the whole proposal is
            judged once on the full objective. ``pi`` warm-starts across
            structural moves rather than being rebuilt.

            The economics: one full rescore per *structural* move instead of one
            per step, with the inner loop driven on the packer's incremental
            ``quality()`` -- a large saving under the cost objective.

            Off by default: at the default ``steps_per_buffer`` it is behind on 4
            of 11 graphs and ahead on none, because each outer move spends an
            entire inner loop and a small budget buys few structural evaluations
            (``coopt_nested_ab.md``); it needs a median 196 steps to come within
            1% of the best any arm reaches, against the incumbent's 40
            (``coopt_convergence.md``). It ties or beats the incumbent at 4-256x
            that budget, so raise ``steps_per_buffer`` with it or leave it off.
        inner_curve: how the inner layout loop's length grows over the outer run,
            from ``inner_len_base * n`` to ``inner_len_max * n`` steps:
            ``"constant"`` (never grows), ``"linear"`` / ``"convex"`` in outer
            progress, or ``"adaptive"`` in the structural *reject* rate -- invest
            more in layout as structure stops moving. Nested mode only.

            ``"constant"`` is the default on measurement (mean +0.02% against
            the incumbent where ``convex`` gives +0.23% and degrades badly at the
            shipping budget). Effect sizes over 5 seeds, not CI-tested.
        inner_annealed: run the inner layout loop as a Metropolis anneal on the
            packer-quality delta (at a calibrated constant ``qtemp``) rather than
            greedy-cold; nested mode only, and the sole consumer of ``qtemp``.
            Off, and should stay off: worse than greedy on every
            graph that discriminates, because a small early inner budget makes it
            wander instead of converge.
        inner_len_base / inner_len_max: the inner loop's length envelope, as
            multiples of the buffer count ``n``; nested mode only.
            ``inner_curve`` interpolates between them, and ``"constant"`` pins the
            length at ``inner_len_base * n``, ignoring ``inner_len_max``.
        early_abandon: run only a quarter of the inner loop, peek at the full
            score, and skip the remainder when the proposal is already worse than
            ``abandon_k * temperature``. Saves the tail of hopeless proposals.
            Nested mode only.
        abandon_k: the multiple of the current temperature above which
            ``early_abandon`` gives up on a proposal. Larger = more patient. Dead
            unless ``early_abandon``.
        trace_every: sample the best-seen score every N steps into ``trace``, a
            list of ``(steps_taken, best_score)``. ``0`` (default) disables it.

            Exists because endpoint comparisons go blind once every arm
            converges, which on this corpus happens below the default budget --
            the ``schedule`` choice reads as a dead tie there and is worth 23% at
            ``spb=2``. The sampling touches neither the RNG nor any search state,
            so a traced solve follows the identical trajectory (asserted in the
            tests). ``steps_taken`` counts a nested outer move as ``1 + inner``,
            so schedules that spend their budget differently share an x-axis.
        polish_frac: fraction of the nested budget held back for a final
            pure-layout anneal on the *best* structure found. Exists because a
            rejected proposal's inner-loop layout work is discarded with it, so
            the winning structure can be left under-refined.

            Defaults to ``0.0``: 8 of 11 graphs are polish-insensitive and on
            ``flash_attention`` more polish steadily hurts, since it steals
            budget from the outer loop and freezes structure too early
            (``coopt_polish_sweep.md``). Measured under the memory-only
            objective, so re-check it if nested is ever turned on.
    """

    def __init__(
        self,
        buffers: Sequence[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
        *,
        seed: int = 0,
        steps_per_buffer: int = 40,
        min_steps: int = 200,
        max_steps: int = 15_000,
        schedule: str = "crude",
        cycles: int = 4,
        horizons_per_cycle: float = 2.0,
        weight_floor: float = 0.05,
        move_bands: Optional[dict[str, tuple[float, float]]] = None,
        reorder_weight: float = 0.5,
        flip_weight: float = 0.3,
        recolor_weight: float = 0.2,
        burst_fraction: float = 0.1,
        burst_fractions: Optional[dict[str, float]] = None,
        burst_move: str = "rotate",
        node_term: bool = False,
        cost_objective: Union[str, "BundleCostObjective", None] = "bundle",
        reorder_move: str = "sweep_quality",
        reorder_neighborhood_scale: float = 1.0,
        sweep_biased_i: bool = True,
        sweep_cleanup: bool = False,
        nested: bool = False,
        inner_curve: str = "constant",
        inner_annealed: bool = False,
        inner_len_base: float = 0.25,
        inner_len_max: float = 3.0,
        early_abandon: bool = True,
        polish_frac: float = 0.0,
        abandon_k: float = 30.0,
        trace_every: int = 0,
    ) -> None:
        super().__init__(buffers, size, alignment)
        # Narrowed from the contravariant parameter type (see the ``buffers``
        # arg). Same objects as the base's ``self.buffers``, so write-back
        # through either name is visible in both.
        self._bufs: Sequence[CoreDivisionBuffer] = cast(
            "list[CoreDivisionBuffer]", list(buffers)
        )
        if schedule not in ("reheating", "crude"):
            raise ValueError("schedule must be 'reheating' or 'crude'")
        if inner_curve not in ("constant", "linear", "convex", "adaptive"):
            raise ValueError("inner_curve must be constant|linear|convex|adaptive")
        if reorder_move not in ("random", "sweep_quality", "sweep_score"):
            raise ValueError("reorder_move must be random|sweep_quality|sweep_score")
        self._seed = seed
        self._steps_per_buffer = steps_per_buffer
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._schedule = schedule
        self._cycles = cycles
        self._horizons_per_cycle = horizons_per_cycle
        self._weight_floor = weight_floor
        # Merge, not replace: a partial override that dropped a band would let
        # _choose_move_reheating pick a move type the schedule has no band for.
        self._move_bands = {**_DEFAULT_MOVE_BANDS, **(move_bands or {})}
        self._reorder_weight = reorder_weight
        self._flip_weight = flip_weight
        self._recolor_weight = recolor_weight
        # Per-move burst lengths, merged over the shared value the way
        # ``move_bands`` is.
        self._burst_fractions = {
            "flip": burst_fraction,
            "recolor": burst_fraction,
            **(burst_fractions or {}),
        }
        unknown = set(self._burst_fractions) - {"flip", "recolor"}
        if unknown:
            raise ValueError(f"burst_fractions keys must be flip/recolor: {unknown}")
        if burst_move not in ("swap", "rotate"):
            raise ValueError("burst_move must be 'swap' or 'rotate'")
        self._burst_move = burst_move
        self._node_term = node_term
        if isinstance(cost_objective, str):
            if cost_objective != "bundle":
                raise ValueError(f"unknown cost_objective {cost_objective!r}")
            cost_objective = _bundle_objective(self._bufs)
            if cost_objective is None:
                logger.info(
                    "cost_objective='bundle' requested with no live Inductor "
                    "graph; falling back to the memory-only objective"
                )
        self._cost_objective = cost_objective
        # Set in _precompute_node_costs, which _precompute_topology calls.
        self._node_costs: Optional[list[list[int]]] = None
        self._reorder_move = reorder_move
        if reorder_neighborhood_scale <= 0.0:
            raise ValueError("reorder_neighborhood_scale must be > 0")
        self._reorder_neighborhood_scale = reorder_neighborhood_scale
        self._sweep_biased_i = sweep_biased_i
        self._sweep_cleanup = sweep_cleanup
        # Sweep-reorder cost instrumentation: positions probed and candidates
        # score-evaluated, so a benchmark can price a step without guessing at
        # the acceptance depth.
        self.sweep_probes = 0
        self.sweep_evals = 0
        self.sweep_steps = 0
        self._nested = nested
        self._inner_curve = inner_curve
        self._inner_annealed = inner_annealed
        self._inner_len_base = inner_len_base
        self._inner_len_max = inner_len_max
        # ``trace`` is [(steps_taken, best_score_so_far)], sampled every
        # ``trace_every`` steps.
        self._trace_every = max(0, int(trace_every))
        self._steps_taken = 0
        self._last_trace = 0
        self.trace: list[tuple[int, int]] = []
        self._early_abandon = early_abandon
        self._polish_frac = polish_frac
        self._abandon_k = abandon_k
        # Best-seen over the anneal (set in _anneal, read in _step); declared for
        # the types.
        self._best_score: int
        self._best_snap: tuple[Packer, list[int]]

    # -- public interface ----------------------------------------------------

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        """Not supported: this engine is joint-only.

        :class:`MemoryPlanSolver` declares this abstract, but placement-only
        annealing belongs to the standalone layout-only annealer; this engine
        adds a joint path rather than replacing that one, and
        ``CoOptimizingAllocator`` only ever calls
        :meth:`plan_layout_and_core_divisions`.
        """
        raise NotImplementedError(
            "SaCoOptimizingSolver is a joint core-division + placement engine; "
            "use plan_layout_and_core_divisions, or "
            "SimulatedAnnealingLayoutSolver for placement-only annealing."
        )

    def plan_layout_and_core_divisions(self) -> list[CoreDivisionBuffer]:
        """Anneal the joint ``(pi, W)`` state and write ``chosen_division`` /
        ``address`` back to each buffer; populate ``spill_reasons``. Returns the
        solver's own buffers. Single-use: construct a fresh solver per set."""
        self.spill_reasons = {}
        # Move instrumentation, populated by the main loop only (not the
        # calibration probes). The anchor ``output_partition`` traces answer the
        # open "weight anchor tilings by output_partition?" question: aggressive
        # anchors proposed but rarely accepted would justify a balanced
        # (beta-biased) recolor proposal.
        self.moves_proposed = {"reorder": 0, "flip": 0, "recolor": 0, "none": 0}
        self.moves_accepted = {"reorder": 0, "flip": 0, "recolor": 0, "none": 0}
        self.recolor_improved = 0
        self.recolor_region_sizes: list[int] = []
        self.recolor_anchor_partitions: list[int] = []
        self.recolor_accepted_partitions: list[int] = []
        self._last_recolor_region_size = 0
        self._last_recolor_anchor_partition = 0
        # Online n / sum / sum-of-squares over nonzero |dE| per move type, for
        # :meth:`move_scale_cv`.
        self._ms_n = {"reorder": 0, "flip": 0, "recolor": 0, "none": 0}
        self._ms_sum = {"reorder": 0.0, "flip": 0.0, "recolor": 0.0, "none": 0.0}
        self._ms_sqsum = {"reorder": 0.0, "flip": 0.0, "recolor": 0.0, "none": 0.0}
        # Sweep-reorder cost counters, per solve.
        self.sweep_probes = 0
        self.sweep_evals = 0
        self.sweep_steps = 0
        self._steps_taken = 0
        self._last_trace = 0
        self.trace = []
        n = len(self._bufs)
        if n == 0:
            return list(self._bufs)

        self._rng = rnd.Random(self._seed)
        self._precompute_topology()

        # Seed: every op at the committed division (index 0); pi from FirstFit.
        self.chosen = [0] * n
        self.packer = self._build_seed_packer()

        self._anneal()
        self._write_back()
        return list(self._bufs)

    # -- static topology (division-invariant) --------------------------------

    def _assert_unsized_buffers_are_pinned(self) -> None:
        """Assert every unsized buffer carries a ``residency_reason``.

        An unsized buffer carries the ``-1`` ``mem_usage`` sentinel
        ``mem_usage_by_buf`` (``utils.py``) emits when it cannot size a buffer.
        :meth:`_per_core_size` clamps that to ``0``, which passes
        :meth:`_eligible`'s capacity gate -- so such a buffer reaching the search
        would be placed occupying no space and the buffer above it would land on
        the same address. A wrong layout, not a crash.

        What prevents it is a coupling across three files: ``mem_usage_by_buf``
        emits ``-1`` on exactly the conditions ``_op_output_good_for_lx_reuse``
        (``allocator.py``) refuses, so the allocator gives every such buffer an
        "op not allowed" ``residency_reason`` and the pin gate rejects it first.
        Nothing in the search re-derives that, so assert it rather than depend on
        the three staying in lockstep -- one pass per solve against an
        O(steps x n) search.
        """
        for b in self._bufs:
            assert b.size >= 0 or b.residency_reason is not None, (
                f"buffer {b.name} is unsized (size={b.size}) but carries no "
                "residency_reason, so nothing gates it out of LX residency; its "
                "per-core footprint would clamp to 0 and the buffer placed above "
                "it would land on the same address"
            )

    def _precompute_topology(self) -> None:
        """Precompute the division-invariant graph structure used every step:
        the name->index map, each buffer's parent indices, and -- keyed by parent
        index -- its children with the ``(parent_div, child_div)`` pairs that keep
        that edge tiling-compatible.

        No consumer *count* is derived here: :meth:`_spill_cost` scales by
        reads-served instead. ``_children`` remains available for the cohort
        multiplicity when op metadata is wired in.
        """
        self._assert_unsized_buffers_are_pinned()
        bufs = self._bufs
        self._name_to_idx = {b.name: i for i, b in enumerate(bufs)}
        n = len(bufs)
        self._parents_idx: list[set[int]] = [set() for _ in range(n)]
        # parent_idx -> list of (child_idx, frozenset of compatible (p_idx, c_idx))
        self._children: list[list[tuple[int, frozenset]]] = [[] for _ in range(n)]
        foreign_parents = 0
        for c_idx, c in enumerate(bufs):
            for p_name in c.parents:
                # A parent outside the solver's buffer set is skipped, not
                # asserted: ``_build_cd_bound_buffers`` assigns
                # ``parents=info["op_inputs"]`` unfiltered, so graph inputs,
                # constants and extern outputs appear here. The edge only exists
                # to gate a child's division against reading the parent's per-core
                # slice *from LX*, and a buffer the solver does not own is never
                # LX-resident, so there is nothing to gate. Clone-eligible graph
                # inputs *are* solver buffers and still resolve normally.
                p_idx = self._name_to_idx.get(p_name)
                if p_idx is None:
                    foreign_parents += 1
                    continue
                self._parents_idx[c_idx].add(p_idx)
                pairs = frozenset(
                    (int(a), int(b)) for a, b in c.cd_parent_matches.get(p_name, [])
                )
                self._children[p_idx].append((c_idx, pairs))
        if foreign_parents:
            logger.debug(
                "dropped %d parent edge(s) naming buffers outside the solver's "
                "set (graph inputs / constants / externs)",
                foreign_parents,
            )

        # Region-recolor support. ``_edge_pairs[(p, c)]`` is the compatible
        # ``(p_div, c_div)`` set on the edge p->c; ``_children_idx`` lists each
        # op's children by index (deterministic flood order).
        self._children_idx = [sorted(c for c, _ in self._children[i]) for i in range(n)]
        self._edge_pairs: dict[tuple[int, int], frozenset] = {
            (i, c): pairs for i in range(n) for c, pairs in self._children[i]
        }
        # Non-trivial (split) menu indices per op -- the only legal recolor
        # anchors, so recolor stays a coordinated *splitting* move and leaves
        # undividing to atomic flips.
        self._nontrivial_menu = [
            sorted(
                j for j, cd in enumerate(b.core_divisions) if cd.output_partition > 1
            )
            for b in bufs
        ]
        self._anchor_candidates = [i for i in range(n) if self._nontrivial_menu[i]]
        self._precompute_spill_costs()

    def _precompute_node_costs(self) -> None:
        """Table of ``node_costs[buffer][division]`` -- the cross-core reduction
        (PSUM-ring) cost of choosing that division, in fixed-point time units.
        ``None`` when the node term is off.

        Only the *reduction* half is expressible here: it needs the reduction
        split and the per-core output size, both of which a buffer carries. The
        matmul half needs the producing op's b/m/n/k iteration-space *extents*,
        which ``CoreDivision`` does not record (it keys splits by index
        coefficient, and the K extent is contracted away). See the ``node_term``
        arg for what that asymmetry means for the search.

        ``shared_weight`` is not recoverable from a buffer either; it only selects
        between two PSUM coefficients, so this takes ``ReductionNode``'s default.

        Precomputed because the table is division-invariant, keeping
        :meth:`_score` to a pair of list walks.
        """
        if not self._node_term:
            self._node_costs = None
            return
        elem = scorer.dtype_bytes()
        table: list[list[int]] = []
        for idx, buf in enumerate(self._bufs):
            row = []
            for div_idx in range(len(buf.core_divisions)):
                cd = buf.core_divisions[div_idx]
                split = math.prod(cd.reduction_splits.values())
                if split <= 1:
                    row.append(0)
                    continue
                row.append(
                    scorer.node_cost_fixed(
                        scorer.ReductionNode(
                            reduction_split=split,
                            output_elems_per_core=(
                                self._per_core_size(idx, div_idx) // elem
                            ),
                        )
                    )
                )
            table.append(row)
        self._node_costs = table

    def _precompute_spill_costs(self) -> None:
        """Cache each buffer's spill cost and the HBM bandwidth constant for
        :meth:`_score`.

        Both are loop-invariant: ``spill_cost`` reads only ``boundary`` /
        ``read_count`` / ``first_use_is_read`` / ``size``, none of which a move
        touches (a division flip changes the *per-core* footprint the packer sees,
        never the total size), and the bandwidth is a hardware constant.
        """
        self._spill_costs = [self._spill_cost(b) for b in self._bufs]
        self._hbm_bytes_per_us = scorer.hbm_bytes_per_us()
        self._precompute_node_costs()

    # -- division-dependent derivations --------------------------------------

    def _per_core_size(self, idx: int, div_idx: int) -> int:
        """Per-core footprint of buffer ``idx`` under menu index ``div_idx``:
        ``ceil(total_size / output_partition)``.

        Uses the substrate's :func:`ceil_div` rather than ``math.ceil`` on a float
        quotient, so this rounds identically to every other footprint-division
        site with no float intermediate to disagree about.

        Clamped non-negative so the packer never sees a negative size from the
        ``mem_usage`` ``-1`` sentinel, mirroring the other packer-feeding sites in
        ``allocator.py``. The clamp alone would make an unsized buffer look
        *placeable* at zero footprint; what rules that out is asserted in
        :meth:`_assert_unsized_buffers_are_pinned`."""
        part = self._bufs[idx].core_divisions[div_idx].output_partition
        return max(0, ceil_div(self._bufs[idx].size, part))

    def _eligible(self, idx: int) -> bool:
        """Whether buffer ``idx`` may be LX-resident under the current ``W``
        (the three division-dependent gates, mirroring
        ``DfsLayoutSolver._evaluate``): the fixed residency pin, a per-core
        footprint that fits at all, and a division carrying a compatible
        ``cd_parent_matches`` pair on *every* child edge.

        Those pairs are per-core-view / core-count based, not ``is_clean`` based: a
        reduction split can appear as a *consumer* index (a K-split reading a clean
        parent via the PSUM ring), but never as a parent index, since a
        reduction-split producer writes a partial sum no child may read from LX --
        so such a producer is always gated out here."""
        b = self._bufs[idx]
        # Not ``MemoryPlanSolver.excluded()``: that folds in a ``min_footprint >
        # limit`` test, which is division-dependent and is the next gate down.
        if b.residency_reason is not None:
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
        parent transparently, so it never in-places onto one.

        ``residency_reason`` is carried so ``MemoryPlanSolver.excluded()`` sees the
        fixed pins during the FirstFit seed pass; the packer ignores it, taking an
        explicit ``eligible`` mask instead.
        """
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
                    residency_reason=b.residency_reason,
                )
            )
        return out

    def _build_seed_packer(self) -> Packer:
        """Build the packer for the seed state: per-core sizes at index 0, a
        FirstFit-derived ``pi``, and the seed eligibility mask."""
        n = len(self._bufs)
        sizes = [self._per_core_size(i, 0) for i in range(n)]
        eligible = [self._eligible(i) for i in range(n)]

        # pi from a FirstFit pass over the per-core sizes. ``_lifetime_buffers``
        # carries ``residency_reason``, so ``FirstFitLayoutSolver.excluded()``
        # leaves the fixed pins unplaced and ``SolverToPermutation`` sorts them
        # after every placed buffer, where they stop displacing eligible buffers
        # to higher addresses. They keep a slot, so pi stays a permutation of all
        # n indices and the packer's ``eligible`` mask lines up index-for-index.
        #
        # Transient, division-dependent ineligibility is deliberately *not*
        # expressed here: it must keep its slot so it can re-enter coherently.
        ff_bufs = self._lifetime_buffers(sizes)
        # Deep-copied so FirstFit lays out its own objects, never the ones the
        # solver mutates; SolverToPermutation reads addresses back by name.
        pi = SolverToPermutation(
            FirstFitLayoutSolver(copy.deepcopy(ff_bufs), self.limit, self.alignment)
        ).permutation(ff_bufs)

        return make_permutation_packer(
            self._lifetime_buffers(sizes),
            pi,
            self.limit,
            self.alignment,
            eligible=eligible,
        )

    # -- scoring (lower is better) -------------------------------------------

    @staticmethod
    def _spill_cost(buffer: CoreDivisionBuffer) -> int:
        """Differential HBM traffic a spill adds over residency, in bytes.

        Duplicates :meth:`_LifetimeBufferWithCpVars.spill_cost` in
        ``ilp_solver_ortools.py`` so the two engines score the same quantity;
        lifting the formula off that CP-SAT-private wrapper into
        ``plan_solver.py`` is a follow-up.

        The reads residency would have served from LX, plus the producer's write,
        which residency turns into a free LX write -- a graph input has no producer
        write to save and a graph output's write-out is unavoidable either way, so
        both cancel, exactly ``boundary != Intermediate``.

        The ``first_use_is_read`` discount drops an input's first read, the
        clone-in that pinning cannot avoid; a computed buffer's first use is the
        producing write, which ``read_count`` already excludes.

        ``size`` is clamped non-negative for the ``mem_usage`` ``-1`` sentinel (see
        :meth:`_per_core_size`). Such buffers are pinned out of LX.
        """
        is_intermediate = buffer.boundary == BufferType.Intermediate
        reads_served = buffer.read_count - (1 if buffer.first_use_is_read else 0)
        return (reads_served + (1 if is_intermediate else 0)) * max(0, buffer.size)

    def _score(self) -> int:
        """The shared objective for the current state, in integer fixed-point
        time units. A buffer with a packer address is LX-resident (its address is
        ``None`` iff ineligible or spilled).

        Hot path: reads ``packer.addresses`` **once** and sums the costs
        precomputed in :meth:`_precompute_spill_costs`. The native packer
        materializes a fresh list per ``addresses`` access, so a per-buffer read
        inside the loop was quadratic; hoisting it is 8-31x faster on the captures.

        The memory-only objective is *differential* -- ``spill_cost`` is the
        traffic a spill adds **over** residency -- so a resident buffer contributes
        zero and only spilled ones are summed. Same shape as the CP-SAT engine's
        ``spill_cost() * (1 - in_buffer)``, so the two are comparable on one
        yardstick.
        """
        if self._cost_objective is not None:
            # The cost model prices compute as well as traffic, so it replaces the
            # memory-only objective rather than adding to it.
            addresses = self.packer.addresses
            resident = frozenset(
                b.name
                for b, address in zip(self._bufs, addresses)
                if address is not None
            )
            return self._cost_objective.score(self.chosen, resident)

        traffic = sum(
            cost
            for cost, address in zip(self._spill_costs, self.packer.addresses)
            if address is None
        )
        memory_fixed = scorer.to_fixed_us(traffic / self._hbm_bytes_per_us)
        # Reduction (PSUM) node cost of the current division vector; 0 unless the
        # node term is enabled. The matmul half stays unmodelled -- see
        # :meth:`_precompute_node_costs`.
        node_fixed = (
            0
            if self._node_costs is None
            else sum(row[d] for row, d in zip(self._node_costs, self.chosen))
        )
        return memory_fixed + node_fixed

    # -- moves ---------------------------------------------------------------

    def _flippable(self) -> list[int]:
        """Buffer indices whose division menu offers an alternative (>1 entry)."""
        return [
            i for i in range(len(self._bufs)) if len(self._bufs[i].core_divisions) > 1
        ]

    def _atomic_flip(self, idx: int, new_div: int) -> None:
        """Change buffer ``idx``'s division to ``new_div`` and ripple: resize its
        per-core footprint, then refresh eligibility for ``idx`` and its parents.
        Those are the only buffers a flip can change, since eligibility depends on
        an op's own division and its children's."""
        self.chosen[idx] = new_div
        self.packer.resize(idx, self._per_core_size(idx, new_div))
        for x in sorted({idx} | self._parents_idx[idx]):
            self.packer.set_eligible(x, self._eligible(x))

    def _flood_region(self, anchor: int, tiling: int) -> dict[int, int]:
        """Flood the ``cd_parent_matches`` relation from ``(anchor, tiling)`` to a
        menu-index assignment over the reachable region.

        Bidirectional: from an assigned op ``u`` (index ``iu``), a child ``c`` joins
        at the smallest ``ic`` with ``(iu, ic)`` compatible, and a parent ``p`` at
        the smallest ``ip`` with ``(ip, iu)`` compatible. The reachable set *is* the
        region; an edge with no compatible index is simply not extended across --
        an accepted internal seam, never a failure.

        First-assignment-wins with a min-index frontier and sorted candidates makes
        this independent of ``cd_parent_matches`` list order.
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
        """One region-recolor move: a uniform anchor op (so a region is hit
        ∝ its op-count), a random non-trivial anchor tiling, flood, recolor,
        burst."""
        anchor = self._rng.choice(self._anchor_candidates)
        tiling = self._rng.choice(self._nontrivial_menu[anchor])
        assignment = self._flood_region(anchor, tiling)
        self._last_recolor_region_size = len(assignment)
        self._last_recolor_anchor_partition = (
            self._bufs[anchor].core_divisions[tiling].output_partition
        )
        self._apply_recolor(assignment)
        self._burst("recolor")

    def _burst(self, move: str) -> None:
        """A short cold layout burst: greedily accept layout steps that do not
        lower the packer's quality, letting ``pi`` adapt to the new footprints
        before the compound move is judged.

        ``move`` selects the length (see ``burst_fractions``), ``burst_move`` the
        primitive. Rejected steps are reverted rather than snapshotted: ``swap`` is
        its own inverse and ``rotate(j, i)`` undoes ``rotate(i, j)``.
        """
        n = len(self._bufs)
        if n < 2:
            return
        fraction = self._burst_fractions[move]
        # Zero means *no* burst. The floor of 1 is there so a small positive
        # fraction still does something on a small graph.
        burst_len = 0 if fraction <= 0.0 else max(1, int(fraction * n))
        if self._burst_move == "swap":
            for _ in range(burst_len):
                i = self._rng.randrange(n - 1)
                if self.packer.swap(i) < 0:
                    self.packer.swap(i)  # revert (self-inverse)
        else:
            for _ in range(burst_len):
                i = self._rng.randrange(n)
                j = self._rng.randrange(n)
                if self.packer.rotate(i, j) < 0:
                    self.packer.rotate(j, i)  # revert

    # -- annealing loop ------------------------------------------------------

    def _snapshot(self) -> tuple[Packer, list[int]]:
        """An independent copy of the joint state ``(pi, W)``: the packer's
        dynamic layout (``copy`` shares only plan-lifetime structures) plus the
        division vector."""
        return (self.packer.copy(), list(self.chosen))

    def _invalidate_cost_objective(self) -> None:
        """Tell the cost objective its diff baseline is stale.

        Restoring a snapshot rewinds ``(pi, W)`` behind the objective's back. Its
        cached bundle *values* stay valid (they are keyed on state), but the
        baseline it diffs against no longer describes the live state."""
        if self._cost_objective is not None:
            self._cost_objective.invalidate()

    def _adopt(self, snap: tuple[Packer, list[int]]) -> None:
        """Install ``snap`` as the live state by *taking ownership* of it -- no
        copy, so the engine goes on mutating those objects and the caller must
        treat ``snap`` as dead from here on.

        The rejection path's restore, where ``snap`` was taken at the top of the
        iteration and dies with it. Zero-copy because a step already pays one O(n)
        packer copy for its snapshot. A snapshot that must *survive* the subsequent
        mutation (``_best_snap``) needs :meth:`_restore_copy`.
        """
        self.packer, self.chosen = snap[0], snap[1]
        self._invalidate_cost_objective()

    def _restore_copy(self, snap: tuple[Packer, list[int]]) -> None:  # noqa: D401
        """Install a *copy* of ``snap`` as the live state, leaving ``snap``
        itself untouched and reusable.

        The variant every *retained* snapshot needs, i.e. ``_best_snap``: the nested
        polish restores it and keeps mutating the live packer, so adopting it would
        let those ``rotate`` / ``resize`` calls rewrite the recorded best layout.
        Since a polish that fails to improve does not refresh ``_best_snap``, the
        engine would publish ``_best_score`` beside a state it no longer describes.
        """
        self.packer, self.chosen = snap[0].copy(), list(snap[1])
        self._invalidate_cost_objective()

    def _calibrate_temperature(self) -> float:
        """A crude scale estimate: the *median* absolute score delta over a sample
        of random (fixed-weight) moves -- the crude schedule's ``T0`` and the
        reheating schedule's pre-snap seed center. Median, not mean, to survive
        region-recolor's large deltas; 1.0 when nothing moved. Restores state;
        consumes RNG deterministically."""
        base = self._score()
        deltas: list[int] = []
        for _ in range(min(64, 4 * len(self._bufs) + 8)):
            snap = self._snapshot()
            self._execute_move(self._choose_move_crude())
            d = abs(self._score() - base)
            if d > 0:
                deltas.append(d)
            self._adopt(snap)  # snap dies here; a fresh one is taken next probe
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
        return self._rng.choices(applicable, weights=[w[m] for m in applicable])[0]

    @staticmethod
    def _hotness(name: str, phi: float) -> float:
        """Structural moves are hot early (cycle phase near 0), layout reorders
        late (near 1)."""
        return phi if name == "reorder" else 1.0 - phi

    def _choose_move_reheating(self, phi: float) -> str:
        """Cycle-phase proposal mix: weight each applicable move by its
        neighborhood size times ``max(w_floor, hotness(m, phi))``, so structural
        moves dominate hot phases and layout reorders dominate cold ones."""
        applicable = self._applicable_moves()
        if not applicable:
            return "none"
        weights = [
            self._neighborhoods[m] * max(self._weight_floor, self._hotness(m, phi))
            for m in applicable
        ]
        return self._rng.choices(applicable, weights=weights)[0]

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
            self._burst("flip")
        elif name == "recolor":
            self._recolor()
        # "none": no applicable move; no-op.

    # -- instrumentation -----------------------------------------------------

    def _record_move_scale(self, name: str, scale: float) -> None:
        """Fold a nonzero ``|dE|`` into ``name``'s within-group stats."""
        if scale > 0.0:
            self._ms_n[name] += 1
            self._ms_sum[name] += scale
            self._ms_sqsum[name] += scale * scale

    def move_scale_cv(self) -> dict[str, float]:
        """Coefficient of variation (std / mean) of ``|dE|`` within each move type
        -- the signal for whether a type should be split into size buckets.
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

    def _choose_reinsertion_source(self, allocated: list[bool]) -> int:
        """Pick the permutation *position* to lift out for a sweep reorder.

        With ``sweep_biased_i`` this is the layout-only annealer's bias (weight ``n``
        for a fully-allocated buffer, ``n_allocated + 1`` otherwise), which
        oversamples the buffers that miss LX -- the ones the objective prices.
        Unbiased is the A/B control separating "sweep over j" from "bias over i".
        """
        n = len(allocated)
        if not self._sweep_biased_i:
            return self._rng.randrange(n)
        n_allocated = sum(1 for a in allocated if a)
        return self._rng.choices(
            range(n), weights=[n if a else n_allocated + 1 for a in allocated]
        )[0]

    def _sweep_upper_bound(self, i: int, allocated: list[bool]) -> int:
        """Highest reinsertion position worth probing for the buffer at position
        ``i`` -- the layout-only annealer's monotonicity bound.

        A buffer's address is non-decreasing in its position, so one that is *not*
        legally allocated can only be made to fit by moving earlier: past the last
        legally-allocated position, nothing it reaches changes the outcome. An
        allocated buffer has no such bound and sweeps to the end.
        """
        n = len(allocated)
        if allocated[i]:
            return n - 1
        last = max((pos for pos, a in enumerate(allocated) if a), default=0)
        return min(n - 1, last + 1)

    def _step_reorder_sweep(
        self, temperature: float, cur: int
    ) -> tuple[int, bool, float]:
        """One best-first reinsertion reorder, the layout-only annealer's move
        (:meth:`SimulatedAnnealingLayoutSolver.annealing_step_rotate`) ported to
        the joint objective. Same contract as :meth:`_step`.

        Lift the buffer at position ``i`` out, probe every reinsertion position by
        rotating it to 0 and bubbling it forward one adjacent swap at a time, then
        try the positions **best-first**, accepting the first that clears the
        Metropolis test. Where a random ``(i, j)`` rotation samples the size-``n``
        neighborhood blindly once, this sees all of it and spends its acceptance
        budget on the good end.

        Ranking is ``sweep_quality`` (the packer's ``quality()``, O(1) per position
        so the sweep is O(n), paying a real ``_score()`` only for the candidates it
        tries) or ``sweep_score`` (exact, one rescore per probe). Quality is a
        *proxy*: it weights a resident buffer by uses x size where the objective
        prices a spilled one by reads-served x size.

        The probe walks the live packer and restores from the step's own snapshot
        rather than sweeping a copy: placement is a pure function of the
        permutation, so rotate-to-``j`` lands in the same state whichever
        intermediate positions the walk passed through.
        """
        packer = self.packer
        perm = packer.permutation
        n = len(self._bufs)
        allocated = [packer.is_fully_allocated(perm[k]) for k in range(n)]
        i = self._choose_reinsertion_source(allocated)
        upper = self._sweep_upper_bound(i, allocated)
        exact = self._reorder_move == "sweep_score"

        snap = self._snapshot()
        self.moves_proposed["reorder"] += 1

        # keys[p] ranks position p (higher is better); None = not a candidate.
        # scores[p] caches the true objective, which only the exact sweep has free.
        keys: list[Optional[float]] = [None] * n
        scores: list[Optional[int]] = [None] * n

        def probe(p: int) -> None:
            if exact:
                scores[p] = self._score()
                keys[p] = -float(scores[p])  # type: ignore[arg-type]
            else:
                keys[p] = packer.quality()

        if i != 0:
            packer.rotate(i, 0)
            probe(0)
        for p in range(1, upper + 1):
            packer.swap(p - 1)  # bubble the lifted buffer from p-1 to p
            if p != i:
                probe(p)
        pos = max(upper, 0)  # where the lifted buffer now sits

        order = sorted(
            (p for p, k in enumerate(keys) if k is not None),
            key=lambda p: -keys[p],  # type: ignore[operator]
        )
        self.sweep_steps += 1
        self.sweep_probes += len(order)

        deltas: list[float] = []
        for j in order:
            if scores[j] is None:
                packer.rotate(pos, j)
                pos = j
                scores[j] = self._score()
                self.sweep_evals += 1
            candidate = scores[j]
            assert candidate is not None
            delta = candidate - cur
            if delta != 0:
                deltas.append(float(abs(delta)))
            if delta <= 0 or self._rng.random() < math.exp(-delta / temperature):
                if pos != j:
                    packer.rotate(pos, j)
                self.moves_accepted["reorder"] += 1
                cur = candidate
                if self._sweep_cleanup:
                    cur = self._reorder_cleanup()
                if cur < self._best_score:
                    self._best_score = cur
                    self._best_snap = self._snapshot()
                scale = sum(deltas) / len(deltas) if deltas else 0.0
                self._record_move_scale("reorder", scale)
                return cur, True, scale

        self._adopt(snap)  # nothing accepted; this step's snapshot dies here
        scale = sum(deltas) / len(deltas) if deltas else 0.0
        self._record_move_scale("reorder", scale)
        return cur, False, scale

    def _reorder_cleanup(self) -> int:
        """The layout-only annealer's post-rotation tidy pass
        (:meth:`SimulatedAnnealingLayoutSolver.annealing_step_swap`), generalized
        to sweep the whole permutation.

        Swap any adjacent pair whose buffers do not overlap in time and sit in
        decreasing address order. Placement-neutral *now* -- neither buffer can
        affect the other's address -- but it leaves the permutation in a form from
        which a later rotation can reach states the unsorted order could not.
        Returns the objective re-read rather than assumed, so this stays honest if
        a swap ever does move something.
        """
        packer = self.packer
        perm = packer.permutation
        n = len(perm)
        top_or_inf = packer.top_or_inf
        k = 0
        while k < n - 1:
            a, b = perm[k], perm[k + 1]
            if (not packer.overlaps(a, b)) and top_or_inf(a) > top_or_inf(b):
                packer.swap(k)
                k = max(0, k - 1)  # the new pair below may now be out of order
            else:
                k += 1
        return self._score()

    def _tick_trace(self, steps: int = 1) -> None:
        """Advance the step counter and sample the convergence trace.

        Touches neither the RNG nor any search state: a trace that perturbed the
        trajectory would measure a different search than it claims to describe, and
        the determinism guarantee would hide that rather than surface it.

        ``steps`` is the work the caller just consumed -- 1 for a judged move,
        ``1 + inner`` for a nested outer move.
        """
        self._steps_taken += steps
        if not self._trace_every:
            return
        if self._steps_taken - self._last_trace >= self._trace_every:
            self._last_trace = self._steps_taken
            self.trace.append((self._steps_taken, self._best_score))

    def _step(self, name: str, temperature: float, cur: int) -> tuple[int, bool, float]:
        """Execute one judged move: propose ``name``, apply the Metropolis test
        against ``temperature``, and update best-seen + instrumentation. Returns
        ``(new_cur, accepted, |dE|)``."""
        if name == "reorder" and self._reorder_move != "random":
            out = self._step_reorder_sweep(temperature, cur)
            self._tick_trace()
            return out
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
            self._adopt(snap)  # this step's snapshot is dead either way
        self._record_recolor(name, accepted)
        self._tick_trace()
        return cur, accepted, scale

    def _anneal(self) -> None:
        n = len(self._bufs)
        # clamp(steps_per_buffer * n, min_steps, max_steps), the shape the
        # layout-only annealer's schedule uses. The ceiling binds only well past
        # the validated corpus (375 buffers at the default spb, vs. n <= 79 in the
        # captures), so it is insurance rather than a tuned bound.
        steps = min(self._max_steps, max(self._min_steps, self._steps_per_buffer * n))
        if self._steps_per_buffer * n > self._max_steps:
            logger.debug(
                "SA co-optimizer step budget clamped to max_steps=%d for %d "
                "buffers (steps_per_buffer=%d would ask for %d); layout quality "
                "is traded for bounded compile time.",
                self._max_steps,
                n,
                self._steps_per_buffer,
                self._steps_per_buffer * n,
            )
        self._flippable_ops = self._flippable()
        # Static proposal-mix neighborhoods: reorder ~ n reinsertion points, flip ~
        # the available local labels, recolor ~ the anchor tilings.
        self._neighborhoods = {
            "reorder": n * self._reorder_neighborhood_scale,
            "flip": sum(
                len(self._bufs[i].core_divisions) - 1 for i in self._flippable_ops
            ),
            "recolor": sum(
                len(self._nontrivial_menu[a]) for a in self._anchor_candidates
            ),
        }

        cur = self._score()
        # The seed state's score; best-seen never rises above it.
        self.baseline_score = cur
        self._best_score = cur
        self._best_snap = self._snapshot()
        if self._trace_every:
            self.trace.append((0, cur))

        if self._applicable_moves():
            if self._nested:
                self._anneal_nested(steps, cur)
            elif self._schedule == "crude":
                self._anneal_crude(steps, cur)
            else:
                self._anneal_reheating(steps, cur)

        self.best_score = self._best_score
        if self._trace_every:
            # Always land the endpoint, so a curve's last point is the reported
            # score rather than the last multiple of trace_every before it.
            self.trace.append((self._steps_taken, self._best_score))
        # Copy, not adopt: ``_best_snap`` stays the record of the published
        # ``best_score``, so the live state _write_back walks must not alias it.
        self._restore_copy(self._best_snap)

    def _anneal_crude(self, steps: int, cur: int) -> None:
        """The default schedule: one geometric cool + fixed proposal weights."""
        t0 = self._calibrate_temperature()
        t_end = max(t0 / 1000.0, 1e-9)
        for step in range(steps):
            frac = step / (steps - 1) if steps > 1 else 1.0
            temperature = t0 * (t_end / t0) ** frac
            cur, _, _ = self._step(self._choose_move_crude(), temperature, cur)

    def _anneal_reheating(self, steps: int, cur: int) -> None:
        """The multi-move self-calibrating reheating schedule with the cycle-phase
        proposal mix: one shared reheating clock, and per move type an acceptance
        band and a move-scale EMA calibrated from its streamed ``|dE|`` (seeded
        pre-snap from a crude median)."""
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
        return self._rng.choices(moves, weights=[w[m] for m in moves])[0]

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
        (``rotate``, far faster-mixing than adjacent swaps) on the current packer,
        either greedy-cold (keep only non-worsening) or annealed (Metropolis on the
        quality delta at ``qtemp``, restoring the best-quality layout at the end).
        Either way the layout left behind is never worse than the one handed in.
        Returns the number of steps taken."""
        n = len(self._bufs)
        steps = int(steps)
        if n < 2 or steps < 1:
            return 0
        best_q = self.packer.quality()
        # The entry layout is itself a candidate for "best". Only the annealed
        # variant needs this: without it a run of accepted worsening steps leaves
        # ``best_packer`` unset and returns the degraded random-walk endpoint.
        # Greedy-cold never worsens, so it pays no copy.
        best_packer: Optional[Packer] = (
            self.packer.copy() if self._inner_annealed else None
        )
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
            self.packer = best_packer  # a local copy; nothing else holds it
        return steps

    def _anneal_nested(self, budget: int, cur: int) -> None:
        """Outer SA over structure; each proposal runs an inner layout loop, is
        judged on the full score (end + early-abandon), and a final pure-layout
        polish refines the best structure. Layout warm-starts across structural
        moves (pi persists)."""
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
            # Burn in, peek the score, and skip the inner tail if the move is
            # hopelessly worse at this temperature.
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
                self._adopt(snap)  # outer snapshot is per-iteration, dead here
            self._record_recolor(name, accepted)
            self._tick_trace(1 + used)
            spent += 1 + used

        # Final polish: a long pure-layout anneal on the best structure found.
        if polish > 0:
            # Copy: the polish mutates the live packer and only refreshes
            # ``_best_snap`` if it improves, so aliasing would corrupt the
            # best-seen layout on a failed polish (see :meth:`_restore_copy`).
            self._restore_copy(self._best_snap)
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
