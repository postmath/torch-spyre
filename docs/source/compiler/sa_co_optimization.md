# Joint core-division + LX placement (the SA co-optimizer)

`SaCoOptimizingSolver` decides two things at once: how each buffer's work is divided across
cores, and where the resulting per-core buffers live in the LX scratchpad. The two are coupled —
a finer division shrinks a buffer's per-core footprint, which changes what fits in LX, which
changes whether the division was worth taking — so solving them separately leaves the
interaction on the table. This document covers what the engine optimizes, where its numbers come
from, and which of its settings rest on what evidence.

For the placement-only annealer (a *different* class, with its own default schedule) see
[Simulated Annealing Layout Planner](simulated_annealing_layout.md). For the surrounding
allocator and the other solvers, see
[Scratchpad Planning](scratchpad_planning.md).

## Where it sits

`config.layout_solver = "simulated_annealing"` with `co_optimizing_lx_planning` routes to
`CoOptimizingAllocator(layout_planning=SaCoOptimizingSolver)`. The allocator builds one
`CoreDivisionBuffer` per graph buffer — carrying the candidate division menu and the
`cd_parent_matches` compatibility relation — and hands the list to the solver, which mutates it
in place with a `chosen_division` and an `address`.

It runs as a **pre-scheduling pass**, which matters more than it sounds: `V.graph` is live but
`V.graph.scheduler` is still `None`, so fusion has not happened yet. Anything the engine wants to
know about the kernels its decisions will land in has to be *estimated* from the ordered
operation list (see [Bundles](#bundles-an-estimate-not-a-fact)).

One contract is worth stating because it was wrong for a while. The allocator sets
`parents = info["op_inputs"]` without intersecting the solver's buffer set, so an op's graph
inputs, constants and extern outputs appear there. The solver **skips** parents it does not own
rather than asserting on them — a buffer the solver does not own is never LX-resident, so the
edge has nothing to gate. Clone-eligible graph inputs are unaffected: those *are* solver buffers
and resolve normally. Before that fix, ten of the eleven corpus graphs failed an assertion here,
which meant the joint path could not run on a real compile at all.

## The objective

Two objectives exist. The engine's default is the cost model.

**Memory-only** (`cost_objective=None`) counts the HBM traffic a spill adds over residency,
converted once to fixed-point microseconds. It is a *differential* objective: a resident buffer
contributes exactly zero, so only spilled buffers are summed. Its weakness is that a core
division only matters through what it lets fit — so on a graph where everything fits, every
division scores the same and the search has nothing to optimize.

**`BundleCostObjective`** (`cost_objective="bundle"`, the default) sums the cost model's
per-fused-bundle predictions. It prices compute as well as traffic, so a division can pay for
itself, and it memoizes per bundle with dirty tracking so an incremental re-score stays
affordable (~5× the memory-only objective, not the 1–2 orders of magnitude a naive
implementation would cost).

The default is the **string** `"bundle"`, not an instance, and that is deliberate. The objective
needs three inputs — the buffers, per-division `OpFeatures`, and the bundle grouping — and only
the first is a solver argument. The allocator reaches the engine through a
`CoreDivisionSolverFactory` that passes `(buffers, size, alignment)` and nothing else, so an
instance could never arrive that way. Given the string, the engine builds the objective itself
from the ambient `V.graph`.

:::{warning}
With no live graph the string **falls back to the memory-only objective** and logs that it did.
That is the normal case for anything driving serialized captures — every test and benchmark in
this series. The failure mode to know about is the quiet one: a benchmark that constructs a
solver the obvious way gets memory-only, completes successfully, and reports numbers about the
engine that used to exist. This is why `docs/source/user_guide/examples/scratchpad/coopt_corpus.py` exists and why every
benchmark prints its objective before doing work.
:::

Measured against memory-only on the regenerated corpus
(`docs/source/compiler/benchmarks/coopt_cost_objective.md`): the cost objective's plans are 18–57% cheaper by the
cost model's own reckoning, winning on 9 of 11 graphs at every capacity and losing on none. The
gap is widest where LX is roomy, because that is exactly where memory-only has no signal — it
leaves the division vector untouched wherever the seed already fits, while the cost objective
moves nearly all of it and will trade residency away to do so.

:::{warning}
That comparison is the model grading plans chosen against itself. **No device time has been
measured.** It is a preliminary result, not a validated speedup. Switching also gives up the
memory-only objective's `best <= baseline` guarantee on spill traffic: this objective trades
residency for divisions (23 of 26 buffers resident vs 25 on `block_x2`), so a miscalibrated
traffic term would regress traffic with nothing in place to catch it.
:::

## The corpus

The benchmarks and tests run against `tests/inductor/cooptimization_captures_regen.json` plus
`tests/inductor/cooptimization_op_features.json`, both produced by
`docs/source/user_guide/examples/scratchpad/capture_op_features.py`. **Regenerating them requires a Spyre machine**: the feature
extractor reads live Inductor IR (committed layouts, `op_it_space_splits`), so it cannot run
against serialized captures.

Three artifacts come out of each compile — the solver's buffers, the estimated bundles, and the
per-division features — and they must come from the *same* compile. An earlier version captured
features alone, to be paired with the older `cooptimization_captures.json`. That pairing cannot
be made safe: every Inductor graph names its buffers `buf0..`, so names collide across unrelated
graphs without lining up, and the models had drifted (`softmax` captured at 1024×512 and
featurized at 192×256; two graphs disagreeing on buffer count outright). An objective built from
that pairing would have been wrong with nothing to signal it.

The older `cooptimization_captures.json` is still used by `test_sa_cooptimizer.py` for
shape-invariant guarantees, where no features are needed. It cannot be used for anything
involving the cost objective, and **scores are not comparable across the two corpora** — they
were captured from different pipeline revisions. A report regenerated against the new corpus is
a new measurement, not an update of an old one.

Fidelity of the regenerated corpus against the original, per `--verify`: `simple_attn`
reproduces exactly; `flash_attention`/`flash_big` are one buffer larger (graph inputs now enter
the solver's buffer set); `block_x2/x3/x4`, `softmax` and `rms_norm` are close; `mlp` and
`swiglu` are smaller because weight transposes no longer materialize as buffers; and `sdpa` is
an explicit decomposition, because `F.scaled_dot_product_attention` now lowers to a
`MultiOutputLayout` extern the allocator never sees.

## Bundles: an estimate, not a fact

The cost model scores one fused kernel at a time, and bundle membership changes the answer —
external inputs are deduplicated across a bundle, the pointwise arity derate counts its ops, the
underfill derate takes its worst tile. The co-optimizer cannot ask for the real grouping,
because fusion is decided two stages later. `fusion.estimate_bundles` reproduces the rule from
the operation list instead, sharing `group_contiguous_fusable` with the real pass so the two can
only diverge on the predicate.

:::{warning}
The estimate's accuracy has been checked against real fusion on **one** softmax graph, where the
bundle count, run structure and boundary placement were right and membership under-counted by a
node scheduling introduces later. On the current corpus it returns a *single* bundle for 9 of 11
graphs. If the real grouping splits differently, the search is optimizing a cost that is not the
cost that gets compiled — and this corpus would not show it. Validating the estimate across the
corpus is the most valuable open piece of work here.
:::

## What the defaults rest on

Every live setting has a measurement behind it; the per-knob docstrings in
`sa_cooptimizer.py` carry the numbers and caveats, and the reports carry the data.

| Setting | Default | Evidence |
|---|---|---|
| `cost_objective` | `"bundle"` | `coopt_cost_objective.md` — 18–57% cheaper plans, 9 of 11 graphs |
| `schedule` | `"crude"` | `coopt_schedule_default.md` — tied on score at 0.73× the CPU |
| `reorder_move` | `"sweep_quality"` | `coopt_reorder_move.md`, `coopt_reorder_focus.md` — a tie; kept because it is also free |
| `reorder/flip/recolor_weight` | 0.5 / 0.3 / 0.2 | `coopt_move_weights.md` — on the frontier; nothing dominates |
| `cycles` | 4 | `coopt_cycle_sweep.md` — ≤0.35% off the best count anywhere |
| `nested` | `False` | `coopt_nested_ab.md` — behind on 4 of 11 graphs at the default budget |
| `inner_curve` | `"constant"` | `coopt_nested_ab.md` — `convex` collapses at the shipping budget |
| `polish_frac` | `0.0` | `coopt_polish_sweep.md` — refuted its own hypothesis |
| `steps_per_buffer` | 40 | Convergence data below; raising it buys −0.06% for ~4× the CPU |

Two entries are dormant rather than active: `polish_frac` and `inner_curve` only take effect
when `nested=True`, and their evidence predates the cost objective. `cycles`, `move_bands`,
`horizons_per_cycle`, `weight_floor` and `reorder_neighborhood_scale` became dormant when the
default schedule changed — they shape the reheating schedule only.

## Three facts that govern how to measure this engine

These are properties of the current corpus and objective, not of the algorithms, and every
benchmark in the series depends on them.

**The search converges early.** 8 of 11 graphs reach their final score by `steps_per_buffer` 20,
against a default of 40. So a comparison run at or above the default is comparing searches that
have already arrived, and will report ties. That is not a knob being irrelevant; it is the
measurement being blind. The `schedule` choice reads as a dead tie at the shipping budget and is
worth 23% at `spb=2`.

**A step is not a fixed price.** Move types cost very different amounts to score under the cost
objective: a recolor rewrites a region's divisions and dirties many bundles, while a reorder only
moves residency. So anything that changes the *proposal mix* changes per-step cost — the crude
schedule costs 0.59–0.76× reheating per step purely because it proposes ~50% reorders against
reheating's ~46% recolors. Comparisons at equal *steps* are therefore not comparisons at equal
time, and the sweeps that assumed otherwise (true under the memory-only objective) had to be
re-run with per-arm wall-clock calibration.

**Two knobs are entangled.** Giving the crude schedule reheating's *observed* proposal mix
reproduces reheating's shape against crude — a small score gain bought with CPU. The schedule
knob is therefore largely a mix knob, and the proposal weights substantially subsume it.
Retuning the weights re-opens the schedule decision.

## Instruments

`trace_every=N` records `(steps_taken, best_score)` into `solver.trace`. Off by default and a
single integer compare when on; it touches neither the RNG nor any search state, so a traced
solve follows an identical trajectory to an untraced one — asserted in the tests rather than
assumed, because the engine's determinism guarantee would otherwise present a perturbed search
as just another valid one. This is what makes "which option gets there sooner" answerable at all.

`docs/source/user_guide/examples/scratchpad/coopt_corpus.py` is the single place that pairs the corpus with its features and
bundles, exposes `cost_objective_for()` so a memory-only arm reads as an explicit opt-out, and
reads `DEFAULT_SPB` / `MIN_STEPS` off the solver's signature so a sweep cannot drift from the
engine it is measuring.

## Reading the benchmark reports

All reports live in [docs/source/compiler/benchmarks/](benchmarks/index.md), with their raw
data in `data/` beside them and their plots in `_static/images/coopt/`. Two
conventions are worth knowing:

* **Seed hygiene.** Each wave uses a fresh seed range (0–4, 20–29, 30–49, 50–69, 70–89, 90–99),
  so a re-run is not reporting in-sample numbers. `FRESH_SEED_BASE` in `coopt_corpus.py` tracks
  where the next wave should start.
* **Generated conclusions.** Several reports had hand-written headline prose, and it went stale
  the first time the underlying sweep was re-run — still naming graphs at parity as regressions,
  and run lengths the grid no longer contained. Headlines are now computed from the results.
  A report whose prose is a constant should be treated with suspicion.

Reports predating the cost objective and the regenerated corpus (`coopt_band_retune_*.md`,
`coopt_polish_sweep.md`, `coopt_schedule_sweep.md`) describe a different objective on different
graphs. They are kept for their reasoning, not their numbers.

## Open work

1. **Validate against device time.** Every score in this series is the cost model's own
   prediction. Nothing has been timed on hardware.
2. **Validate `estimate_bundles` across the corpus**, not one graph. It currently returns one
   bundle for 9 of 11 graphs.
3. **A traffic guardrail**, if the cost objective's residency trades ever prove costly — the
   engine already computes the memory-only score and could log when a chosen plan regresses it.
4. **Nested mode at a raised budget.** It is off because it loses at `spb=40`, but it ties or
   beats the incumbent at 4–256× that budget while being several times cheaper per solve. If
   that is ever worth pursuing, `polish_frac`, `inner_curve`, `inner_len_base` and the budget
   want sweeping together rather than one at a time.

## Related documents

* [Scratchpad Planning](scratchpad_planning.md) — the allocator, the other solvers, and the
  co-optimization concept
* [Simulated Annealing Layout Planner](simulated_annealing_layout.md) — the placement-only
  annealer and its schedule
* [Work Division Planning](work_division_planning.md) — where the candidate core divisions come
  from
