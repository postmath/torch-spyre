# Scratchpad examples and benchmark harnesses

Two different kinds of script live here, with very different prerequisites.

## Examples

`toy_layout.py`, `random_buffers.py` and `inplace_annealing.py` are self-contained: they build
their own synthetic buffers and need nothing but an installed `torch_spyre`. Run them directly.

## Benchmark harnesses

Everything named `profile_*.py`, plus `capture_op_features.py`, `coopt_corpus.py` and
`warm_start_transfer.py`, are the harnesses behind the measurements in
[docs/source/compiler/benchmarks/](../../../compiler/benchmarks/index.md). They are **not**
standalone examples, and will not run from an installed wheel. Each needs:

* **A repository checkout on `PYTHONPATH`.** They import test fixtures
  (`tests.inductor.cooptimization_capture_loader`) and write their reports back into the docs
  tree, so they must see the repo root:

  ```console
  $ PYTHONPATH=$(pwd) python3 docs/source/user_guide/examples/scratchpad/profile_coopt_convergence.py
  ```

  Run from the repository root. `profile_coopt_reorder_focus.py` imports its sibling, so invoke
  that one as a module instead:

  ```console
  $ PYTHONPATH=$(pwd) python3 -m docs.source.user_guide.examples.scratchpad.profile_coopt_reorder_focus
  ```

* **The captured corpus**, `tests/inductor/cooptimization_captures_regen.json` and
  `tests/inductor/cooptimization_op_features.json`. Both are committed, so a sweep needs no
  hardware. Regenerating them does: `capture_op_features.py` reads live Inductor IR and only
  runs on a Spyre machine.

* **CPU, not a Spyre card.** The sweeps solve serialized captures, so they are pure CPU work and
  parallelize across processes (`WORKERS` in each script). A full sweep is 1–20 minutes.

### Conventions worth knowing before adding one

* **Import the corpus, never re-derive it.** `coopt_corpus.py` is the one place that pairs the
  captures with their op features and estimated bundles. A solver built without an explicit
  objective silently falls back to the pre-cost-model one and reports numbers about an engine
  that no longer exists — which has happened. Use `cost_objective_for()` and call `announce()`
  so every run states its objective before doing work.
* **Compare at matched wall-clock when the arms cost different amounts per step.** Move types
  differ in scoring cost under the cost objective, so equal step counts are not equal time. The
  schedule, reorder and weight sweeps all calibrate per-arm `spb` grids for this reason; see
  `--calibrate` in `profile_coopt_schedule_default.py` for the pattern.
* **Generate conclusions from the results.** Write the headline prose from the data rather than
  by hand. Every hand-written headline in this series eventually contradicted its own tables.
* **Use a fresh seed range.** `FRESH_SEED_BASE` in `coopt_corpus.py` records where the next wave
  should start; 0–99 are spent.

### Where the outputs go

| Artifact | Location |
|---|---|
| Report page | `docs/source/compiler/benchmarks/*.md` |
| Raw results | `docs/source/compiler/benchmarks/data/*.json` |
| Plots | `docs/source/_static/images/coopt/*.png` |

Each harness takes `--report` to rebuild its page from existing raw results without re-running
the sweep, and most take `--smoke` for a fast subset that validates the plumbing.

:::{warning}
`--smoke` writes to the *same* results file and report page as a full run, so it overwrites real
data with a two-graph subset. Committed results are recoverable from version control, but
uncommitted ones are not — regenerate with a full run before committing anything a smoke run
touched.
:::
