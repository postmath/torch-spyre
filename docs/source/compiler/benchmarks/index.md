# Scratchpad and co-optimizer benchmarks

Measurements behind the scratchpad planner's and co-optimizer's design decisions. Each page is
generated from a sweep's raw results by the harness that produced it, so a page and its data
cannot drift apart: re-running `--report` rebuilds the prose and tables from `data/`.

The harnesses live in
[docs/source/user_guide/examples/scratchpad/](../../user_guide/examples/scratchpad/README.md),
which documents how to run them and what they need. Raw results are in `data/` next to these
pages — sweep JSON, plus console transcripts for the one experiment whose results were never
written up (`scratchpad_wst_*.out`). Plots are in `_static/images/coopt/`.

## Reading these

Three conventions matter for interpreting any page here:

* **Not all numbers are comparable.** The co-optimizer's objective changed from memory-only
  spill traffic to the cost model's per-bundle prediction, and the corpus was recaptured at the
  same time. Pages predating both (`coopt_band_retune_*`, `coopt_polish_sweep`,
  `coopt_schedule_sweep`) measure a different objective on different graphs. They are kept for
  their reasoning, not their figures. See
  [the co-optimizer document](../sa_co_optimization.md) for what changed.
* **Seed ranges are disjoint by wave** (0–4, 20–29, 30–49, 50–69, 70–89, 90–99), so a re-run is
  never reporting in-sample numbers.
* **Conclusions are computed, not written.** Several of these pages once carried hand-written
  headline prose, which went stale the first time the underlying sweep was re-run — naming
  graphs at parity as regressions, and run lengths the grid no longer contained. If a page's
  headline is a constant rather than generated, distrust it.

## The objective and the corpus

| Page | Question |
|---|---|
| [`coopt_cost_objective`](coopt_cost_objective.md) | Is the cost model a better objective than memory-only? |

## Schedule, moves and weights

| Page | Question |
|---|---|
| [`coopt_schedule_default`](coopt_schedule_default.md) | Which schedule should be the default, at matched wall-clock? |
| [`coopt_capacity_crossover`](coopt_capacity_crossover.md) | Does the schedule choice flip with LX capacity? |
| [`coopt_move_weights`](coopt_move_weights.md) | Are the crude schedule's proposal weights on the score/CPU frontier? |
| [`coopt_reorder_move`](coopt_reorder_move.md) | Best-first reinsertion sweep vs random rotation, at equal time |
| [`coopt_reorder_focus`](coopt_reorder_focus.md) | The same question at higher power, on the graphs that discriminate |
| [`coopt_cycle_sweep`](coopt_cycle_sweep.md) | How many reheating cycles? |
| [`coopt_schedule_sweep`](coopt_schedule_sweep.md) | Schedule vs run length (pre-cost-model) |
| [`coopt_band_retune_band`](coopt_band_retune_band.md) | Retuning reheating's acceptance band (pre-cost-model) |
| [`coopt_band_retune_scale`](coopt_band_retune_scale.md) | Retuning its reorder neighborhood scale (pre-cost-model) |
| [`coopt_band_retune_validate`](coopt_band_retune_validate.md) | Whether those retunings survive held-out seeds (pre-cost-model) |

## Search structure and convergence

| Page | Question |
|---|---|
| [`coopt_convergence`](coopt_convergence.md) | Which option reaches the best score soonest? |
| [`coopt_nested_ab`](coopt_nested_ab.md) | Is the nested two-timescale loop worth its complexity? |
| [`coopt_polish_sweep`](coopt_polish_sweep.md) | How much final layout polish? (pre-cost-model) |

## Packer and solver mechanics

These measure implementation cost rather than search quality, so the objective change does not
affect them.

| Page | Question |
|---|---|
| [`copy_vs_swap_results`](copy_vs_swap_results.md) | What do the reinsertion sweep's primitives cost? |
| [`capped_allocator_plan_results`](capped_allocator_plan_results.md) | Reference vs incremental permutation solver |
| [`simulated_annealing_solve_results`](simulated_annealing_solve_results.md) | Where does time go in one full anneal? |
| [`coopt_reorder_focus_cap4`](coopt_reorder_focus_cap4.md) | The reorder focus check at a tighter capacity |
| [`coopt_reorder_move_cap4`](coopt_reorder_move_cap4.md) | The reorder A/B at a tighter capacity |

```{toctree}
:hidden:
:maxdepth: 1

coopt_cost_objective
coopt_schedule_default
coopt_capacity_crossover
coopt_move_weights
coopt_reorder_move
coopt_reorder_focus
coopt_reorder_move_cap4
coopt_reorder_focus_cap4
coopt_cycle_sweep
coopt_schedule_sweep
coopt_band_retune_band
coopt_band_retune_scale
coopt_band_retune_validate
coopt_convergence
coopt_nested_ab
coopt_polish_sweep
copy_vs_swap_results
capped_allocator_plan_results
simulated_annealing_solve_results
```
