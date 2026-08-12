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
Today it reduces to the substrate's own HBM-traffic model: the *differential*
``spill_cost`` (``read_count`` re-reads plus the producer write, the latter only
for an ``Intermediate`` buffer) summed over the buffers that miss LX, with a zero
node term and unit cohort multiplicity (Plan §7.1). Because the cost is
differential, a resident buffer contributes nothing -- the same shape as the
CP-SAT engine's objective, so the two are comparable on one yardstick. Residency,
per-core sizing, and the ``cd_parent_matches`` compatibility gate follow the
substrate exactly. The richer per-edge multiplicity and the matmul node term are
wired in when real op metadata is available (Phase 6); the scorer already
supports them.

Determinism (Plan §7.5): a seeded ``Random`` over index-ordered domains and the
integer fixed-point score make a run bit-for-bit reproducible.
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

    The objective needs three things the solver is not handed: per-division
    ``OpFeatures``, the fused-bundle grouping, and the buffer order. Only the
    last comes from the arguments; the other two are read off ``V.graph``, which
    is set while the allocator runs -- this solver *is* an Inductor pass, so the
    graph is ambient rather than absent.

    Returns ``None`` when there is no live graph. That is the normal case for
    every test and benchmark that drives the solver from a serialized capture,
    where no amount of plumbing could produce features. The caller falls back to
    the memory-only objective and says so in the log, rather than raising (which
    would make the string unusable off-hardware) or degrading silently (which
    would let a run believe it used the cost model when it did not).
    """
    from torch._inductor.virtualized import V

    from torch_spyre._inductor.fusion import estimate_bundles
    from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective
    from torch_spyre._inductor.scratchpad.op_features import features_for_menu

    # Unset, ``V.graph`` is a ``NullHandler`` rather than ``None`` or a raise, so
    # detect the live graph by the two things this needs from it instead of by
    # identity -- that also covers any future stand-in handler.
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


# The packer is either the pure-Python solver or the native C++ solver; both
# expose the same permutation-packer interface the co-optimizer drives, so
# ``make_permutation_packer`` may return either. Use ``.quality()`` (not the
# Python-only ``total_quality`` attribute) so both work.
Packer = Union[PermutationBasedLayoutSolver, NativePermutationLayoutSolver]

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


class SaCoOptimizingSolver(CoreDivisionLayoutSolver):
    """SA joint core-division + LX-placement engine (see module docstring).

    Args:
        buffers: the buffers to plan, in the allocator's order. Declared as
            ``Sequence[LifetimeBoundBuffer]`` so the class itself satisfies
            ``CoreDivisionSolverFactory`` (``Callable`` parameters are
            contravariant, so a narrower annotation would not), but every buffer
            actually passed must be a :class:`CoreDivisionBuffer` -- the joint
            engine reads the ``core_divisions`` menu and the
            ``cd_parent_matches`` compatibility relation off each one.

            **Mutated in place, and their order is an index.** The returned list
            is these same objects with ``chosen_division`` and ``address``
            written back, so a caller that needs the input preserved must copy
            first (the benchmarks all ``deepcopy``). Position ``i`` here is the
            index used by ``chosen``, by the packer's permutation, and by the
            cost objective, so the three stay aligned only as long as this order
            does. Solvers are single-use: construct a fresh one per buffer set.
        size: scratchpad capacity in bytes.
        alignment: placement alignment (128 = one Spyre stick).
        seed: RNG seed; fixes the (deterministic) search trajectory.
        steps_per_buffer: annealing steps scale linearly with the buffer count
            (crude bounded budget; Plan Appendix H tunes this later).
        min_steps: floor on the step budget for tiny graphs.
        max_steps: ceiling on the *total* step budget, so a large graph cannot
            run away. Mirrors the layout-only annealer's schedule-level clamp
            (``SelfCalibratingReheatingSchedule.max_steps``, 5_000) but sits
            higher: this engine searches divisions as well as layout, so it
            wants a larger budget at the same buffer count. Note this bounds
            *steps*, not wall-clock -- per-step cost also grows with ``n``.
        schedule: ``"crude"`` (default -- a single geometric cool with fixed
            proposal weights) or ``"reheating"`` (Plan §5 -- the multi-move
            self-calibrating reheating schedule + cycle-phase proposal mix).

            Crude is the default on the score/CPU frontier, not on score.

            The two are statistically tied on score, and crude is cheaper, so
            reheating sits off the frontier. At the shipping budget the tie is
            flat:
            re-measured under the cost objective
            (``docs/source/compiler/benchmarks/coopt_schedule_default.md``, 20 fresh seeds) every
            capacity x move cell lands within noise of zero, 25 of 33 exactly
            tied, and the capacity sweep finds no crossover anywhere from 0.80
            down to 0.10 of the footprint.

            That tie is an artifact of measuring after both have arrived. On this
            corpus 8 of 11 graphs reach their final score by
            ``steps_per_buffer`` 20, against a default of 40, so a sweep at or
            above the default compares two converged searches. Measured *before*
            convergence, crude is worse by +23.6% at ``spb=2``, +4.7% at 5, +2.8%
            at 10, +0.7% at 20, and +0.16% at 40 -- a clean decay to the tie. The
            schedule is doing real work; the corpus is just too easy at the
            budget we ship to show it.

            Traced directly (``docs/source/compiler/benchmarks/coopt_convergence.md``) the same fact
            reads off the curve: median 40 steps for reheating to come within 1%
            of the best score any arm reaches, against 82 for crude, and crude is
            slower on every graph that converges at all -- 616 steps vs 224 on
            ``flash_attention``, 205 vs 41 on ``block_x4``.

            **Those are steps, and a step is no longer a fixed price.** The two
            schedules propose different move *types* -- crude ~50% reorder,
            reheating ~5% reorder against ~46% recolor -- and under the cost
            objective a recolor rewrites a region's divisions and dirties many
            bundles where a reorder only moves residency. Measured, reheating
            costs ~1.7x per step (``flash_big`` 1.87s vs 1.10s at ``spb=40``).
            The sweeps in this series assume the opposite ("schedule choice does
            not change per-step work"), which held under the memory-only
            objective and does not hold now.

            Re-run with per-arm wall-clock calibration, the step-count advantage
            does not survive being priced: at matched CPU the two are still a tie,
            now with a slight tilt the *other* way (mean -0.02% to -0.05% in
            crude's favour, all four capacity x move cells' CIs spanning zero,
            25-26 of 33 cells exactly tied). Crude wins 6-7 cells to reheating's
            1, and the residual mismatch runs against crude -- it received ~13%
            less CPU than its calibration intended -- so if anything the tilt is
            understated.

            Pooled over the matched-CPU run, crude scores -0.008% against
            reheating for 0.73x the CPU, and 9 of 11 graphs tie exactly. Crude
            converges in more *steps* -- a median 82 against 40 to come within 1%
            of the best any arm reaches -- but its steps cost 0.59-0.76x, because
            it proposes ~50% reorders where reheating proposes ~46% recolors and a
            recolor dirties far more bundles under the cost objective. The two
            effects cancel on score and do not cancel on time.

            **What this promotes.** The reheating-only knobs (``cycles``,
            ``move_bands``, ``horizons_per_cycle``, ``weight_floor``,
            ``reorder_neighborhood_scale``) are now dormant, and the crude-only
            ones -- ``reorder_weight``, ``flip_weight``, ``recolor_weight`` --
            are now on the live path. Those three have never been swept: they are
            the guessed weights this schedule shipped with as an A/B baseline,
            and nothing in the benchmark suite has ever varied them. That is the
            largest unmeasured surface in this engine.

            Two consequences. Do not read the tie as licence to drop reheating:
            on a harder graph, or at a tighter budget, the gap is large. And do
            not lower ``steps_per_buffer`` to save time without re-checking this
            -- most of the headroom between 20 and 40 is exactly what makes the
            schedule choice free.
        cycles / horizons_per_cycle: reheating-schedule knobs (Plan §5.1).
            ``cycles`` was measured as "mildly suboptimal at 4, worth ~2% on the
            graphs that matter" under the memory-only objective. Re-run under the
            cost objective (``docs/source/compiler/benchmarks/coopt_cycle_sweep.md``), the cycle count
            changes the result on 3 of 11 graphs and the default is at most 0.35%
            off the best count anywhere.

            Unlike ``schedule``, that is not a convergence artifact: checked
            before the search converges (``spb`` 2-20, where the schedule choice
            is still worth up to 23%), the spread across counts 1..16 is at most
            3.4% and the best count is inconsistent between budgets (8, 16, 4, 8,
            16) -- noise, not a ranking. Not worth changing.
        weight_floor: the ``w_floor`` in the cycle-phase proposal mix (Plan §5.4),
            so no applicable move type is ever fully starved.
        move_bands: per-move-type ``(accept_hi, accept_lo)`` acceptance bands for
            the reheating schedule; defaults to :data:`_DEFAULT_MOVE_BANDS`.
        reorder_weight / flip_weight / recolor_weight: fixed proposal weights for
            the ``"crude"`` schedule -- which, since crude became the default, is
            the live path.

            Guessed when crude existed only as the A/B baseline reheating had to
            beat, and now measured for the first time
            (``docs/source/compiler/benchmarks/coopt_move_weights.md``, matched wall-clock, 10 fresh
            seeds). **They survive.** Nothing swept is both significantly better
            on score and no more expensive; the defaults land on the score/CPU
            frontier, between the arms that buy score with CPU
            (``recolor-heavy`` -0.124% at 1.04x) and those that buy CPU with
            score (``reorder-heavy`` +0.079%, not significant, at 0.92x). Pushing
            all the way to reorder-only is the one clear mistake: +0.523%
            (significant) for 0.86x.

            Two things worth knowing before retuning them. The whole spread is
            ~0.1% of score against a cost model no one has checked on hardware.
            And they are not independent of ``schedule``: giving crude the
            reheating schedule's *observed* mix reproduces reheating's shape
            against crude -- a small score gain bought with CPU -- so the
            schedule knob is largely a mix knob, and moving these weights
            re-opens that decision.
        burst_fraction: layout-burst length as a fraction of the buffer count
            (Plan §4.4; the burst warms ``pi`` to the new footprints before the
            compound move is judged).
        node_term: add the cross-core reduction (PSUM-ring) cost of the chosen
            divisions to the objective. **Off by default, and deliberately so.**

            Today the objective is memory-only, so a division only matters if it
            changes what fits in LX -- which is why four of the eleven captures
            reach the objective's floor (all traffic from permanently-pinned
            buffers) and can never distinguish any move set or schedule.

            Enabling this prices the reduction half of the §6.3 node term, which
            is computable from a buffer alone. The matmul half is not (it needs
            the producing op's b/m/n/k extents and axis roles; see
            :meth:`_precompute_node_costs`), and that asymmetry is the hazard:
            the reduction term charges for splitting a reduction axis while the
            matmul term that would *reward* splitting stays unmodelled. So this
            is a one-sided objective, biased against reduction splits, and it is
            an instrument for measuring whether a node term restores the corpus's
            discriminating power -- not a better objective to solve against.
            Landing the matmul half is what would make it a default.

            Measured: enabling it does **not** restore discriminating power. The
            search drives every reduction split to zero -- final node cost 0 on
            all eleven captures, 0 ops keeping a split -- because a term that is
            pure cost with no modelled reward has its minimum at "never split".
            The same three graphs discriminate as before.

            Deliberately a bool rather than a scale. The intended pairing is an
            external cost model's matmul term (``macs / cores``), which supplies
            the missing reward. Against *that* scale this term acts only as a
            tie-breaker: that model reads neither ``reduction_cores`` nor, at
            fixed core count, anything separating a K-split from an output split
            (identical predictions for ``(m, n, k)`` of ``(8, 4, 1)``,
            ``(8, 1, 4)`` and ``(4, 2, 4)``). Any positive coefficient therefore
            yields the same decisions, so a magnitude is not identifiable from
            the available evidence and a float knob would be false precision. It
            stays at the hardware-fitted ``_PSUM_PER_CORE_ELEM_US``; add a scale
            only if a K-split calibration ever makes one identifiable.
        cost_objective: which objective the search minimizes -- the cost model's
            per-bundle prediction (:class:`BundleCostObjective`) or the
            memory-only spill traffic that preceded it.

            * ``"bundle"`` (default) -- build the cost objective here, off the
              live Inductor graph. The features and the fused-bundle grouping
              come from ``V.graph``, which is ambient while the allocator runs,
              so this form needs nothing from the caller. That is what lets it be
              the default at all: the allocator constructs solvers through a
              ``CoreDivisionSolverFactory`` that passes only
              ``(buffers, size, alignment)``, so an instance could never arrive
              that way. With no live graph -- any run driven from a serialized
              capture -- it logs and falls back to memory-only.
            * ``None`` -- the memory-only objective, explicitly.
            * an instance -- used as given; how the benchmarks drive captures.

            Default on the evidence in ``docs/source/compiler/benchmarks/coopt_cost_objective.md``:
            the cost objective's plans are 18-57% cheaper by the cost model's own
            reckoning, winning on 9 of 11 corpus graphs at every capacity and
            losing on none, with the gap widest where LX is roomy and the
            memory-only objective has no signal at all (it leaves the division
            vector untouched wherever the seed already fits).

            Two caveats travel with that default, deliberately recorded here
            rather than in a commit message. The comparison is the model grading
            its own plans -- **no device time has been measured** -- so this is a
            preliminary outcome, not a validated speedup. And the memory
            objective's ``best <= baseline`` guarantee on spill traffic does not
            carry over: this objective trades residency away for divisions (23 of
            26 buffers resident vs 25 on ``block_x2``), so a miscalibrated
            traffic term would regress traffic with nothing here to catch it.
            Pass ``None`` to get the old behaviour back.
        reorder_move: how a ``reorder`` proposal picks its reinsertion position.

            * ``"sweep_quality"`` (default) -- the layout-only annealer's
              best-first sweep: lift one buffer, probe every reinsertion
              position, and try them in descending packer ``quality()`` order.
            * ``"sweep_score"`` -- the same sweep ranked by the true objective at
              every position. Exact, but it buys nothing over the proxy (see
              below) while costing up to 1.4x per step.
            * ``"random"`` -- the original single random ``(i, j)`` rotation,
              retained as the A/B baseline.

            The sweep became the default on benchmark evidence that **no longer
            holds**. Under the memory-only objective, at matched wall-clock, it
            was better on every capture that discriminated and worse on none, by
            ~3% mean at capacity ``footprint//4`` (11 of 12 graph x budget cells,
            sign-test p = 0.003). Re-run under the cost objective
            (``docs/source/compiler/benchmarks/coopt_reorder_move.md``, 20 seeds), it is
            indistinguishable from ``random``: 0 of 33 cells significantly
            better, 0 significantly worse, mean +0.01%.

            The reversal is explicable rather than mysterious, and the paragraph
            below predicted it: the sweep's advantage came from ranking
            score-identical positions by a continuous proxy under an objective
            that was a coarse step function. The cost objective is not a step
            function, so there are far fewer ties to break.

            Nor is that tie a convergence artifact, which is the obvious
            objection given that most of this corpus has converged by the default
            budget: measured before convergence too, at ``spb`` 2 through 20, the
            two stay within -0.21% to +0.18% of each other. The sweep buys
            nothing at any budget, where the ``schedule`` choice at the same
            budgets is worth up to 23%.

            It stays the default because the re-run found no *cost* either -- a
            sweep step is ~1.0-1.1x a random one now that scoring dominates
            per-step time -- so this is a knob with no measured consequence in
            either direction, not a knob with a live justification. Prefer
            ``"random"`` if simplicity is worth anything to you.

            Ranking by ``quality()`` rather than the objective is deliberate. The
            objective counts only *spilled* buffers, so it is a coarse step
            function of ``pi`` that most rotations leave untouched (reorder
            acceptance runs at 96-100%); the continuous quality breaks ties among
            those score-identical positions and steers the layout toward states a
            later structural move can exploit.
        reorder_neighborhood_scale: multiplier on ``reorder``'s neighborhood size
            in the reheating schedule's cycle-phase proposal mix (§5.4). Exists
            for investigation, **not** as a tuning recommendation: leave it at 1.

            The mix weights each move by its neighborhood, and ``reorder``'s
            (``n``) is dwarfed by the flip/recolor menus, so reheating spends only
            7-12% of its proposals on reorder against ``crude``'s ~50%. That is
            why the (much stronger) sweep pays off far less under reheating. But
            raising this does *not* recover the difference: a sweep over
            1..32 x 2 cycle counts found arms at -2.5 to -2.8% whose advantage
            then collapsed to within noise on held-out seeds, while ``crude``'s
            ~3.3% lead replicated. See ``docs/source/compiler/benchmarks/coopt_band_retune_scale.md``
            and ``..._validate.md``.
        sweep_biased_i: bias the sweep's choice of which buffer to lift toward
            ones that are not fully allocated (the layout-only annealer's
            weighting). Carries a real part of the win -- turning it off erases
            the sweep's advantage -- so it is on by default.
        sweep_cleanup: run a placement-neutral tidy pass after an accepted sweep
            rotation, sorting adjacent non-overlapping buffers into address
            order. Off by default: it is O(n) swaps per accepted move with
            quadratic worst-case backtracking, and it showed no benefit.
        nested: run the two-timescale loop (:meth:`_anneal_nested`) instead of
            the single loop. The outer anneal proposes only *structural* moves
            (flip/recolor); each proposal is followed by an inner layout loop
            that re-adapts ``pi`` to the new footprints, and the whole proposal
            is then judged once on the full objective. Layout warm-starts across
            structural moves -- ``pi`` persists rather than being rebuilt.

            The economics: the single loop pays a full rescore every step, while
            this pays one per *structural* move and drives the inner loop on the
            packer's incremental ``quality()`` instead. Under the cost-model
            objective a full rescore is expensive, so that is a large saving.

            **Off by default, and the benchmark that looks like it argues
            otherwise does not.** At matched steps it ties or beats the incumbent
            on 9 of 11 graphs at a ~6x median wall-clock speedup -- but every
            level of that sweep is 4x to 256x the default ``steps_per_buffer``.
            At the default itself it is behind on 4 of 11 graphs and ahead on
            none (worst ``flash_big`` +6.5%, ``flash_attention`` +3.7%), because
            each outer move spends an entire inner loop, so a small budget buys
            few structural evaluations. It is still several times cheaper in
            solver time there, but that saving is fractions of a second per
            graph -- not a trade worth a few percent of plan quality. Turn this
            on together with a raised ``steps_per_buffer``, or not at all. See
            ``docs/source/compiler/benchmarks/coopt_nested_ab.md``.

            The convergence trace puts the same conclusion more starkly: at the
            default budget nested never comes within 1% of the best score any arm
            reaches on 5 of 11 graphs, and needs a median 196 steps where the
            incumbent needs 40 (``docs/source/compiler/benchmarks/coopt_convergence.md``).
        inner_curve: how the inner layout loop's length grows over the outer run,
            from ``inner_len_base * n`` to ``inner_len_max * n`` steps:
            ``"constant"`` (never grows), ``"linear"`` / ``"convex"`` in outer
            progress, or ``"adaptive"`` in the structural *reject* rate -- invest
            more in layout as structure stops moving. Nested mode only.

            ``"constant"`` is the default on measurement: mean +0.02% against
            the incumbent where ``convex`` gives +0.23%, and convex degrades
            badly at the shipping budget (+10.3% on ``flash_attention``, +15.8%
            on ``flash_big``). Effect sizes, not CI-tested -- the sweep behind
            them ran 5 seeds.
        inner_annealed: run the inner layout loop as a Metropolis anneal on the
            packer-quality delta (at a calibrated constant ``qtemp``) rather than
            greedy-cold. Off, and it should stay off: annealed is worse than
            greedy on every graph that discriminates, because with a small early
            inner budget it wanders instead of converging.
        inner_len_base / inner_len_max: the inner loop's length envelope, as
            multiples of the buffer count ``n``. ``inner_curve`` interpolates
            between them; ``"constant"`` pins the length at ``inner_len_base * n``
            and ignores ``inner_len_max``.
        early_abandon: run only a quarter of the inner loop, peek at the full
            score, and skip the remainder when the proposal is already worse than
            ``abandon_k * temperature``. Saves the tail of hopeless proposals.
        abandon_k: the multiple of the current temperature above which
            ``early_abandon`` gives up on a proposal. Larger = more patient.
        trace_every: sample the best-seen score every N steps into ``trace``, a
            list of ``(steps_taken, best_score)``. ``0`` (default) disables it.

            Exists because this series' benchmarks compare *endpoints*, and an
            endpoint goes blind the moment every arm converges -- which on the
            current corpus happens by ``steps_per_buffer`` 20, below the default
            of 40. That blindness is not hypothetical: the ``schedule`` choice
            reads as a dead tie at the shipping budget and is worth 23% at
            ``spb=2``, because reheating converges faster rather than better. A
            trace shows that directly instead of requiring someone to suspect it
            and go looking.

            The sampling touches no RNG and no search state, so a traced solve
            follows the identical trajectory to an untraced one -- asserted in
            the tests rather than assumed, since the engine's determinism
            guarantee would otherwise mask a perturbation as just another valid
            search. ``steps_taken`` counts a nested outer move as ``1 + inner``,
            so schedules that spend their budget differently share an x-axis.
        polish_frac: fraction of the nested budget held back for a final
            pure-layout anneal on the *best* structure found, after the outer
            loop ends. Exists because the outer loop only ever refines layout
            inside a proposal's inner loop, and a rejected proposal's layout work
            is discarded with it -- so without a polish the winning structure can
            be left under-refined.

            Defaults to ``0.0``: the hypothesis did not survive its sweep
            (``docs/source/compiler/benchmarks/coopt_polish_sweep.md``). 8 of 11 graphs were
            polish-insensitive and on ``flash_attention`` more polish steadily
            hurt -- 0.0 landed within +1.0% of the incumbent where 0.2 gave
            +12.3% -- because the polish steals budget from the outer structural
            loop and freezes structure on the best-so-far too early. Measured
            under the memory-only objective, so re-check it if nested is ever
            turned on.
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
        burst_fraction: float = 0.5,
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
        # Declared over ``LifetimeBoundBuffer`` so the class itself satisfies
        # ``CoreDivisionSolverFactory`` (``Callable`` parameters are
        # contravariant, so a narrower parameter would not), then narrowed here:
        # the joint engine needs the ``core_divisions`` menus every buffer it is
        # actually given carries. Same objects as the base's ``self.buffers``, so
        # write-back through either name is visible in both.
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
        # Merge (not replace) so a partial override keeps the defaults for the
        # move types it omits -- otherwise _choose_move_reheating can select a
        # move type whose band is missing and crash in the reheating schedule.
        self._move_bands = {**_DEFAULT_MOVE_BANDS, **(move_bands or {})}
        self._reorder_weight = reorder_weight
        self._flip_weight = flip_weight
        self._recolor_weight = recolor_weight
        self._burst_fraction = burst_fraction
        self._node_term = node_term
        # ``"bundle"`` is the self-sufficient form, and therefore the default:
        # the engine builds the cost objective itself off the live graph, so a
        # caller that cannot reach Inductor's IR (the allocator hands this class
        # nothing but buffers) still gets it. An instance is still accepted, and
        # is what the benchmarks pass when driving serialized captures.
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
        # Per-step sweep cost instrumentation (populated only by the sweep
        # reorder): positions probed and candidates score-evaluated, so a
        # benchmark can price a step without guessing at the acceptance depth.
        self.sweep_probes = 0
        self.sweep_evals = 0
        self.sweep_steps = 0
        # Nested two-timescale mode (experimental): the outer loop anneals over
        # structure (flip/recolor) and each proposal runs an inner layout loop
        # whose length grows over the run; see :meth:`_anneal_nested`.
        self._nested = nested
        self._inner_curve = inner_curve
        self._inner_annealed = inner_annealed
        self._inner_len_base = inner_len_base
        self._inner_len_max = inner_len_max
        # Convergence trace (off by default). ``trace`` is [(steps_taken,
        # best_score_so_far)], sampled every ``trace_every`` steps -- the
        # instrument for "which option gets there sooner", which an endpoint
        # comparison cannot answer once every option has converged.
        self._trace_every = max(0, int(trace_every))
        self._steps_taken = 0
        self._last_trace = 0
        self.trace: list[tuple[int, int]] = []
        self._early_abandon = early_abandon
        self._polish_frac = polish_frac
        self._abandon_k = abandon_k
        # Best-seen over the anneal (set in _anneal, read in _step); declared here
        # so their type is known across methods.
        self._best_score: int
        self._best_snap: tuple[Packer, list[int]]

    # -- public interface ----------------------------------------------------

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        """Not supported: this engine is joint-only.

        :class:`MemoryPlanSolver` declares this abstract, but the placement-only
        path deliberately belongs to the standalone layout-only annealer, which
        Plan §7.3 keeps as its own :class:`MemoryPlanSolver` ("we do not replace
        it"). ``CoOptimizingAllocator`` -- the only allocator this engine is
        injected into -- calls :meth:`plan_layout_and_core_divisions` exclusively,
        so this stays a loud stub rather than a second annealing path to maintain.
        """
        raise NotImplementedError(
            "SaCoOptimizingSolver is a joint core-division + placement engine; "
            "use plan_layout_and_core_divisions, or "
            "SimulatedAnnealingLayoutSolver for placement-only annealing."
        )

    def plan_layout_and_core_divisions(self) -> list[CoreDivisionBuffer]:
        """Anneal the joint ``(pi, W)`` state and write ``chosen_division`` /
        ``address`` back to each buffer; populate ``spill_reasons``. Returns the
        solver's own buffers (the one-shot interface satisfied by an internal
        solve) -- a solver is single-use, so construct a fresh one per set."""
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
        # Sweep-reorder cost counters, per solve (see :meth:`_step_reorder_sweep`).
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

    def _precompute_topology(self) -> None:
        """Precompute the division-invariant graph structure used every step:
        the name->index map, each buffer's parent indices, and -- keyed by parent
        index -- its children with the ``(parent_div, child_div)`` pairs that keep
        that edge tiling-compatible.

        The consumer *count* is no longer derived here: the landed spill cost
        scales by the buffer's reads-served count instead -- ``read_count``
        discounted by an input's unavoidable clone-in, see :meth:`spill_cost`.
        The two agree on every captured graph (see
        ``test_read_count_matches_consumer_count``), and ``_children`` remains
        available for the Phase-6 cohort multiplicity.
        """
        bufs = self._bufs
        self._name_to_idx = {b.name: i for i, b in enumerate(bufs)}
        n = len(bufs)
        self._parents_idx: list[set[int]] = [set() for _ in range(n)]
        # parent_idx -> list of (child_idx, frozenset of compatible (p_idx, c_idx))
        self._children: list[list[tuple[int, frozenset]]] = [[] for _ in range(n)]
        foreign_parents = 0
        for c_idx, c in enumerate(bufs):
            for p_name in c.parents:
                # A parent that is not a solver buffer is skipped, not asserted.
                # This used to assert, on the premise that the substrate built
                # ``parents`` by intersecting an op's reads with the solver's
                # buffer set. It does not: ``_build_cd_bound_buffers`` assigns
                # ``parents=info["op_inputs"]`` unfiltered, so graph inputs,
                # constants and extern outputs appear here and the assert fired on
                # 10 of the 11 corpus graphs -- i.e. on essentially every real
                # compile. (The sibling in-place path in the same builder already
                # guards this way, for the same reason.)
                #
                # Skipping is the correct semantics, not just the convenient one.
                # The edge exists to gate a child's division against reading the
                # parent's per-core slice *from LX*; a buffer the solver does not
                # own is never LX-resident, so there is nothing to gate. Note this
                # does not silently drop clone-eligible graph inputs -- those *are*
                # solver buffers (``rms_norm`` owns ``arg0_1``) and resolve here
                # normally.
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
        self._precompute_spill_costs()

    def _precompute_node_costs(self) -> None:
        """Table of ``node_costs[buffer][division]`` -- the cross-core reduction
        (PSUM-ring) cost of choosing that division, in fixed-point time units.
        ``None`` when the node term is off.

        Only the *reduction* half of the §6.3 node term is expressible here. It
        needs the reduction split (a product of ``reduction_splits`` values) and
        the per-core output size -- a split factor and an output-side quantity,
        both of which a buffer carries. The matmul half needs the producing op's
        b/m/n/k iteration-space *extents* and axis roles; ``CoreDivision`` records
        only how each axis is split, keyed by index coefficient, and the K extent
        is contracted away entirely so no output-side quantity can recover it. See
        the class docstring for what that asymmetry means for the search.

        ``shared_weight`` is not recoverable from a buffer either; it only selects
        between two PSUM coefficients, so this takes ``ReductionNode``'s default
        (the shared-weight coefficient) rather than inventing a per-op answer.

        The table is division-invariant, so :meth:`_score` stays a pair of list
        walks rather than recomputing products in the hot loop.
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

        Both are loop-invariant across the whole anneal: ``spill_cost`` reads only
        ``boundary`` / ``read_count`` / ``first_use_is_read`` / ``size``, none of
        which a move touches (a division flip changes the *per-core* footprint the
        packer sees, never the buffer's total size), and the bandwidth is a
        hardware constant behind a lazy module lookup.
        """
        self._spill_costs = [self._spill_cost(b) for b in self._bufs]
        self._hbm_bytes_per_us = scorer.hbm_bytes_per_us()
        self._precompute_node_costs()

    # -- division-dependent derivations --------------------------------------

    def _per_core_size(self, idx: int, div_idx: int) -> int:
        """Per-core footprint of buffer ``idx`` under menu index ``div_idx``:
        ``ceil(total_size / output_partition)`` (Plan §2.2).

        Uses the substrate's shared :func:`ceil_div` rather than ``math.ceil`` on
        a float quotient, so this rounds identically to every other
        footprint-division site (the CP-SAT engine and ``CoreDivisionBuffer``'s
        own per-core sizes) with no float intermediate to disagree about.

        Clamped to be non-negative: an unsized buffer carries the ``mem_usage``
        ``-1`` sentinel (a non-placeable "op not allowed" input in the captures),
        so its footprint is never actually used -- clamp it so a nonsense size can
        never look placeable, and so the packer never receives a negative size.
        Mirrors the ``max(0, ...)`` clamp at the other packer-feeding sites in
        ``allocator.py``."""
        part = self._bufs[idx].core_divisions[div_idx].output_partition
        return max(0, ceil_div(self._bufs[idx].size, part))

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
        # The fixed pin, read straight off the substrate field. Deliberately not
        # ``MemoryPlanSolver.excluded()``: that folds in a ``min_footprint >
        # limit`` capacity test, which is division-*dependent* and is the second
        # gate below -- routing through it would conflate the two gates.
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

        ``residency_reason`` is carried so that ``MemoryPlanSolver.excluded()``
        sees the fixed pins during the FirstFit seed pass (Plan §7.4); the packer
        ignores it, taking an explicit ``eligible`` mask instead.
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
        FirstFit-derived ``pi`` (Plan §8.2), and the seed eligibility mask."""
        n = len(self._bufs)
        sizes = [self._per_core_size(i, 0) for i in range(n)]
        eligible = [self._eligible(i) for i in range(n)]

        # pi from a FirstFit pass over the per-core sizes. ``_lifetime_buffers``
        # carries ``residency_reason``, so ``FirstFitLayoutSolver.excluded()``
        # leaves the fixed pins unplaced and ``SolverToPermutation`` sorts them
        # after every placed buffer (Plan §7.4: pi is ordered over the buffers
        # that can ever be resident). They keep their slot in pi -- it stays a
        # permutation of all n indices, so the packer's ``eligible`` mask still
        # lines up index-for-index -- they simply stop occupying prefix slots and
        # displacing eligible buffers to higher addresses.
        #
        # Transient, division-dependent ineligibility is deliberately *not*
        # expressed here: a buffer whose seed-division footprint exceeds capacity
        # is already tail-sorted by ``excluded()``'s ``min_footprint`` test (that
        # predates this and is unchanged), and eligibility that comes and goes
        # with ``W`` must keep its slot so it can re-enter coherently.
        ff_bufs = self._lifetime_buffers(sizes)
        # Deep-copied so FirstFit lays out its own objects: SolverToPermutation
        # only reads addresses back by name, and the buffers it is handed must
        # not be the ones the solver mutates.
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

    # -- scoring (shared scorer; lower is better) ----------------------------

    @staticmethod
    def _spill_cost(buffer: CoreDivisionBuffer) -> int:
        """Differential HBM traffic a spill adds over residency, in bytes.

        Duplicates :meth:`_LifetimeBufferWithCpVars.spill_cost` in
        ``ilp_solver_ortools.py`` so the two engines score the same quantity. It
        is duplicated rather than shared because the landed formula lives on a
        CP-SAT-private wrapper; lifting it into ``plan_solver.py`` and having both
        call it is a deliberate follow-up, not a prerequisite.

        The reads residency would have served from LX, plus the producer's own
        write, which residency turns into a free LX write. A graph input has no
        producer write to save and a graph output's write-out is unavoidable
        either way, so both cancel -- exactly ``boundary != Intermediate``.

        ``read_count`` counts the buffer's reads, not the savings: an input's
        first read is the clone-in that pinning cannot avoid, so it is discounted
        here. For a computed buffer the first use is the producing write and
        ``read_count`` already excludes it, hence the discount is keyed on
        ``first_use_is_read`` -- the same reasoning, and the same expression, as
        ``_LifetimeBufferWithCpVars.spill_cost``.

        ``size`` is clamped non-negative for the same reason as in
        :meth:`_per_core_size`: an unsized buffer carries the ``mem_usage`` ``-1``
        sentinel, and pricing it as negative traffic would let a mostly-resident
        state reach a *negative* total, which ``to_fixed_us`` rejects outright.
        Such buffers are pinned out of LX, so their traffic is the same in every
        state -- a constant the search cannot act on, and zero is the honest
        constant when the real size is unknown.
        """
        is_intermediate = buffer.boundary == BufferType.Intermediate
        reads_served = buffer.read_count - (1 if buffer.first_use_is_read else 0)
        return (reads_served + (1 if is_intermediate else 0)) * max(0, buffer.size)

    def _score(self) -> int:
        """The shared objective for the current state, in integer fixed-point
        time units. A buffer with a packer address is LX-resident (its address is
        ``None`` iff ineligible or spilled).

        Hot path: this runs at least once per annealing step, so it reads
        ``packer.addresses`` **once** and sums the per-buffer costs precomputed in
        :meth:`_precompute_spill_costs`. The native packer materializes a fresh
        list on every ``addresses`` access, so the obvious per-buffer read inside
        the loop was quadratic; hoisting it and pricing each spill from a
        precomputed vector is 8-31x faster across the captured graphs.

        The landed substrate objective is *differential* -- ``spill_cost`` is the
        traffic a spill adds **over** residency -- so a resident buffer
        contributes exactly zero and only the spilled ones are summed. This is
        the same shape as the CP-SAT engine's ``spill_cost() * (1 - in_buffer)``,
        which is what makes the two directly comparable on one yardstick
        (Plan §7.1).

        The node term is zero until real op metadata is available (Phase 6); the
        scorer's fixed-point conversion keeps this deterministic.
        """
        if self._cost_objective is not None:
            # The cost model prices compute as well as traffic, so it replaces
            # this objective rather than adding to it -- summing the two would
            # double-count the HBM term, which the model already carries.
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

    def _snapshot(self) -> tuple[Packer, list[int]]:
        """An independent copy of the joint state ``(pi, W)``: the packer's
        dynamic layout (``copy`` shares only plan-lifetime structures) plus the
        division vector."""
        return (self.packer.copy(), list(self.chosen))

    def _invalidate_cost_objective(self) -> None:
        """Tell the cost objective its diff baseline is stale.

        Restoring a snapshot rewinds ``(pi, W)`` behind the objective's back; its
        cached bundle *values* stay valid (they are keyed on state) but the
        baseline it diffs against no longer describes the live state."""
        if self._cost_objective is not None:
            self._cost_objective.invalidate()

    def _adopt(self, snap: tuple[Packer, list[int]]) -> None:
        """Install ``snap`` as the live state by *taking ownership* of it -- no
        copy, so the engine goes on mutating those very objects and the caller
        must treat ``snap`` as dead from here on.

        This is the rejection path's restore, where ``snap`` was taken at the top
        of the iteration and dies with it, so the transfer is free and safe.
        Zero-copy deliberately: a step already pays one O(n) packer copy for its
        snapshot, and copying again on every rejection would double the hot
        loop's copy traffic for nothing. A snapshot that must *survive* the
        subsequent mutation (``_best_snap``) needs :meth:`_restore_copy`.
        """
        self.packer, self.chosen = snap[0], snap[1]
        self._invalidate_cost_objective()

    def _restore_copy(self, snap: tuple[Packer, list[int]]) -> None:  # noqa: D401
        """Install a *copy* of ``snap`` as the live state, leaving ``snap``
        itself untouched and reusable.

        The variant every *retained* snapshot needs, i.e. ``_best_snap``: the
        nested polish restores it and then keeps mutating the live packer in
        place, and adopting it there would make those ``rotate`` / ``resize``
        calls rewrite the recorded best layout. A polish that fails to improve
        does not refresh ``_best_snap``, so the engine would end up publishing
        ``_best_score`` (the better number) alongside a state it no longer
        describes.
        """
        self.packer, self.chosen = snap[0].copy(), list(snap[1])
        self._invalidate_cost_objective()

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

    def _choose_reinsertion_source(self, allocated: list[bool]) -> int:
        """Pick the permutation *position* to lift out for a sweep reorder.

        With ``sweep_biased_i`` this is the layout-only annealer's bias (weight
        ``n`` for a fully-allocated buffer, ``n_allocated + 1`` for one that is
        not): the buffers that miss LX are the ones the objective actually prices,
        so they get sampled far above their share. Unbiased is the A/B control
        that separates "sweep over j" from "bias over i".
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

        A buffer's address is non-decreasing in its position, so one that is
        *not* legally allocated can only be made to fit by moving it earlier:
        past the last legally-allocated position nothing it can reach changes the
        outcome. A buffer that is allocated has no such bound and sweeps to the
        end.
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
        Metropolis test. A random ``(i, j)`` rotation gets one blind sample of a
        size-``n`` neighborhood; this sees the whole neighborhood and spends its
        acceptance budget on the good end of it.

        Two rankings, which is the point of the A/B:

        - ``sweep_quality`` ranks by the packer's own ``quality()`` -- O(1) per
          position, so the sweep is O(n) -- and pays a real ``_score()`` only for
          the candidates it actually tries, in rank order. Quality is a *proxy*:
          it weights a resident buffer by uses x size, where the objective prices
          a spilled one by reads-served x size.
        - ``sweep_score`` ranks by the true objective at every position, which is
          exact but costs a rescore per probe.

        The probe walks the live packer and restores from the step's own snapshot,
        rather than sweeping a second copy: the step already pays one O(n) packer
        copy, and placement is a pure function of the permutation, so the state
        reached by rotate-to-``j`` is the same whichever intermediate positions the
        walk passed through.
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
        # scores[p] caches the true objective there, which only the exact sweep
        # knows for free.
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
            # `or` short-circuits, so the RNG is drawn only when delta > 0.
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
        decreasing address order. That is placement-neutral *now* -- neither
        buffer can affect the other's address -- but it leaves the permutation in
        a form from which a later rotation can reach states the unsorted order
        could not. Returns the objective afterwards, read rather than assumed:
        the invariant says quality is preserved, and the score is a different
        functional of the same allocation set, so this stays honest if a swap ever
        does move something.
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

        Off unless ``trace_every`` is set, and a no-op beyond one integer
        compare when it is. Deliberately touches neither the RNG nor any search
        state: a trace that perturbs the trajectory measures a different search
        than the one it claims to describe, and this engine's determinism
        guarantee would hide that rather than surface it.

        ``steps`` is the work the caller just consumed -- 1 for a judged move,
        ``1 + inner`` for a nested outer move -- so the x-axis means the same
        thing across schedules that spend their budget differently.
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
        # clamp(steps_per_buffer * n, min_steps, max_steps) -- the same shape the
        # layout-only annealer's schedule uses, so neither engine's budget grows
        # without bound. The ceiling binds only well past the validated corpus
        # (15_000/40 = 375 buffers vs. n <= 79 in the captures), so it is
        # insurance rather than something the tuned range ever meets.
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
        # Static proposal-mix neighborhoods (Plan §5.4): reorder ~ n reinsertion
        # points; flip ~ the available local labels; recolor ~ the anchor tilings
        # (n_regions x n_colors, with the anchor's non-trivial menu size as
        # n_colors, §7.2).
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
        # Baseline = the seed state's score; best-seen never rises above it, the
        # >=-baseline guarantee (Plan §8.1).
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
            # The endpoint always lands in the trace, whatever the sampling
            # interval divides into, so a curve's last point is the reported
            # score rather than the last multiple of trace_every before it.
            self.trace.append((self._steps_taken, self._best_score))
        # Copy, not adopt: ``_best_snap`` stays the record of the published
        # ``best_score``, so the live state _write_back walks must not alias it.
        self._restore_copy(self._best_snap)

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
        annealed (Metropolis on the quality delta at ``qtemp``, tracking +
        restoring the best-quality layout). Either way the layout left behind is
        never worse than the one handed in. Returns the number of steps taken."""
        n = len(self._bufs)
        steps = int(steps)
        if n < 2 or steps < 1:
            return 0
        best_q = self.packer.quality()
        # The entry layout is itself a candidate for "best", so the loop can
        # never hand back something worse than it was given. Only the annealed
        # variant needs this: an all-accepted run of worsening Metropolis steps
        # otherwise leaves ``best_packer`` unset and returns the degraded
        # random-walk endpoint. Greedy-cold never worsens, so it pays no copy.
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
                self._adopt(snap)  # outer snapshot is per-iteration, dead here
            self._record_recolor(name, accepted)
            self._tick_trace(1 + used)
            spent += 1 + used

        # Final polish: a long pure-layout anneal on the best structure found.
        if polish > 0:
            # Copy: the polish mutates the live packer in place and only
            # refreshes ``_best_snap`` if it improves, so aliasing the retained
            # snapshot here would corrupt the best-seen layout on a failed polish.
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
