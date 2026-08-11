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

"""A/B: nested two-timescale SA vs the current single-loop engine.

The nested engine (``SaCoOptimizingSolver(nested=True)``) anneals over structure
(flip/recolor) in the outer loop and runs an inner layout loop (rotate-based,
warm-started, length grows over the run) per structural proposal, judged end +
early-abandon, with a final pure-layout polish. This sweeps nested variants
(greedy/annealed inner x constant/linear/convex/adaptive length curve) against
crude and reheating, over run length, 5 seeds, all captured graphs.

Records both final best score AND wall-clock per solve, because the nested inner
loop skips the per-step full rescore -- so the honest "is it more efficient?"
axis is quality vs time, not quality vs step budget. Parallelized across
processes; results written incrementally.

**Objective.** Runs against :class:`BundleCostObjective`, the engine's default.
That is not automatic here: the default is the *string* ``"bundle"``, which
builds itself from the live Inductor graph, and a benchmark driving serialized
captures has no such graph -- so simply constructing the solver would silently
fall back to the memory-only objective and measure the old thing. The objective
is therefore built explicitly, from the same captured features + estimated
bundles the engine would have derived. ``--memory-only`` restores the previous
behaviour for comparison.

That also forces the corpus: ``cooptimization_captures.json`` carries no
features, so this now runs on ``cooptimization_captures_regen.json``, which was
captured alongside them. Scores are therefore not comparable to the numbers in
the committed ``coopt_nested_ab.md`` -- different objective *and* different
graphs. What carries over is the question: does the nested two-timescale loop
buy anything the single loop does not?

The objective changes the very thing this A/B is about. Per-step cost is now
dominated by the cost model rather than by the packer, and the nested engine's
whole premise is skipping per-step full rescores -- so the quality-vs-time
frontier is being re-measured under a per-step price roughly 5x the one the
nested variants were designed against.

**Step budget.** ``steps = clamp(steps_per_buffer * n, min_steps, max_steps)``
with the engine's default ``max_steps=15_000``, which this benchmark does not
override. So the top of the ``spb`` grid saturates: for ``n = 80`` every level at
or above 640 is the same 15_000-step solve. Levels that clamp to a step count
already covered are dropped rather than re-run (they would be bit-identical), and
the dropped set is logged.

Run from the repo root::

    python3 benchmarks/profile_coopt_nested_ab.py            # full A/B + report
    python3 benchmarks/profile_coopt_nested_ab.py --smoke    # tiny subset
    python3 benchmarks/profile_coopt_nested_ab.py --memory-only
    python3 benchmarks/profile_coopt_nested_ab.py --report   # report only
"""

from __future__ import annotations

import os

for _v in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

from tests.inductor.cooptimization_capture_loader import load_captures  # noqa: E402
from torch_spyre._inductor.scratchpad.sa_cooptimizer import (  # noqa: E402
    SaCoOptimizingSolver,
)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


_BENCH = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH)
_RESULTS = os.path.join(_BENCH, "results")
_FIXTURES = os.path.join(_REPO, "tests", "inductor")
CAPTURES = os.path.join(_FIXTURES, "cooptimization_captures_regen.json")
FEATURES = os.path.join(_FIXTURES, "cooptimization_op_features.json")

SWEEP_JSON = os.path.join(_RESULTS, "coopt_nested_ab.json")
REPORT_MD = os.path.join(_BENCH, "coopt_nested_ab.md")
DELTA_PNG = os.path.join(_RESULTS, "coopt_nested_delta.png")
FRONTIER_PNG = os.path.join(_RESULTS, "coopt_nested_frontier.png")

SEEDS = [0, 1, 2, 3, 4]
WORKERS = 48
# The engine's own defaults, read off its signature rather than copied, so a
# change there cannot silently desynchronize this sweep from the solver: the
# clamp bounds drive the level dedup, and DEFAULT_SPB is the operating point
# production actually runs at.
_SIG = inspect.signature(SaCoOptimizingSolver).parameters
_MIN_STEPS = _SIG["min_steps"].default
_MAX_STEPS = _SIG["max_steps"].default
DEFAULT_SPB = _SIG["steps_per_buffer"].default
# The default budget leads, because a config is chosen for where it will run.
# Without it this sweep started at 4x production and ranged to 256x, which is
# exactly the regime where the nested engine looks best -- and it reverses below
# that range. See the production-budget paragraph in the generated report.
SPB_GRID = [DEFAULT_SPB, 160, 640, 2560, 10240]
BIG_N = 70  # graphs at/above this size skip the most expensive spb
BIG_SPB_CAP = 2560

# config name -> solver kwargs. "reheat" is the incumbent baseline.
CONFIGS = {
    "crude": dict(schedule="crude"),
    "reheat": dict(schedule="reheating"),
    "nest-greedy-constant": dict(
        nested=True, inner_annealed=False, inner_curve="constant"
    ),
    "nest-greedy-linear": dict(nested=True, inner_annealed=False, inner_curve="linear"),
    "nest-greedy-convex": dict(nested=True, inner_annealed=False, inner_curve="convex"),
    "nest-greedy-adaptive": dict(
        nested=True, inner_annealed=False, inner_curve="adaptive"
    ),
    "nest-anneal-convex": dict(nested=True, inner_annealed=True, inner_curve="convex"),
    "nest-anneal-adaptive": dict(
        nested=True, inner_annealed=True, inner_curve="adaptive"
    ),
}
BASELINE = "reheat"

_GRAPHS: dict = {}


def _foot(bufs):
    return sum(math.ceil(b.size / b.core_divisions[0].output_partition) for b in bufs)


def _load_graphs():
    """``{graph: {buffers, features, bundles}}`` -- everything a solve needs.

    Features and bundles ride along because the cost objective cannot be built
    from buffers alone; see the module docstring.
    """
    from torch_spyre._inductor.cost_model import op_from_dict

    with open(FEATURES) as fh:
        raw = json.load(fh)["graphs"]
    out = {}
    for name, graphs in load_captures(CAPTURES).items():
        entry = raw[name]
        out[name] = {
            "buffers": graphs[0].buffers,
            "features": {
                n: [None if f is None else op_from_dict(f) for f in b["features"]]
                for n, b in entry["buffers"].items()
            },
            "bundles": entry["bundles"],
        }
    return out


def _objective(entry, buffers):
    """A fresh cost objective over ``buffers`` -- one per solve, never shared:
    it memoizes per bundle and tracks dirty state against the last scored
    ``(chosen, resident)``."""
    from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective

    return BundleCostObjective(
        [b.name for b in buffers], entry["features"], entry["bundles"]
    )


def _effective_steps(n, spb):
    """The engine's own clamp, replicated so the sweep can skip levels that
    resolve to a solve it has already run (see the module docstring)."""
    return min(_MAX_STEPS, max(_MIN_STEPS, spb * n))


def _init_worker():
    global _GRAPHS
    _GRAPHS = _load_graphs()
    import atexit

    atexit.register(os._exit, 0)


def _work(task):
    """task = (name, spb, config, seed, memory_only) -> (task, best, base, secs)."""
    import copy

    name, spb, config, seed, memory_only = task
    entry = _GRAPHS[name]
    bufs = copy.deepcopy(entry["buffers"])
    cap = max(1, _foot(bufs) // 2)
    # Built outside the timed region: the engine builds its own from the live
    # graph in production, so its construction is not part of what this A/B
    # compares. Scoring with it *is*, and that stays inside.
    objective = None if memory_only else _objective(entry, bufs)
    t0 = time.time()
    s = SaCoOptimizingSolver(
        bufs,
        cap,
        128,
        seed=seed,
        steps_per_buffer=spb,
        cost_objective=objective,
        **CONFIGS[config],
    )
    s.plan_layout_and_core_divisions()
    return task, s.best_score, s.baseline_score, time.time() - t0


def _levels_for(n, spb_grid):
    """The ``spb`` levels worth running for a graph of ``n`` buffers, plus the
    ones dropped as duplicates: ``(kept, dropped)``. Two levels that clamp to the
    same step count give bit-identical solves, so running both measures nothing
    and costs twice."""
    kept, dropped, seen = [], [], set()
    cap_spb = BIG_SPB_CAP if n >= BIG_N else max(spb_grid)
    for spb in spb_grid:
        if spb > cap_spb:
            dropped.append((spb, "over BIG_SPB_CAP"))
            continue
        steps = _effective_steps(n, spb)
        if steps in seen:
            dropped.append((spb, f"clamps to {steps} steps, already run"))
            continue
        seen.add(steps)
        kept.append(spb)
    return kept, dropped


def _tasks(graphs, configs, spb_grid, seeds, memory_only):
    for name in graphs:
        kept, dropped = _levels_for(len(graphs[name]["buffers"]), spb_grid)
        for spb, why in dropped:
            print(f"  skip {name} spb={spb}: {why}", flush=True)
        for spb in kept:
            for config in configs:
                for seed in seeds:
                    yield (name, spb, config, seed, memory_only)


def run_sweep(smoke=False, memory_only=False):
    graphs = _load_graphs()
    spb_grid = [160, 640] if smoke else SPB_GRID
    seeds = [0] if smoke else SEEDS
    cfgs = (
        ["reheat", "nest-greedy-convex", "nest-anneal-convex"]
        if smoke
        else list(CONFIGS)
    )
    names = ["swiglu", "flash_attention"] if smoke else list(graphs)
    objective = "memory-only" if memory_only else "BundleCostObjective"
    print(f"objective: {objective}", flush=True)
    tasks = list(
        _tasks({k: graphs[k] for k in names}, cfgs, spb_grid, seeds, memory_only)
    )

    results: dict = {c: {"n": len(graphs[c]["buffers"]), "levels": {}} for c in names}
    os.makedirs(_RESULTS, exist_ok=True)
    start = done = 0
    start = time.time()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, spb, config, seed, _), best, base, secs in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            lv = results[name]["levels"].setdefault(
                str(spb),
                {
                    "total_steps": _effective_steps(results[name]["n"], spb),
                    "baseline": base,
                    "configs": {},
                },
            )
            c = lv["configs"].setdefault(config, {"best": [], "secs": []})
            c["best"].append(best)
            c["secs"].append(secs)
            done += 1
            if done % 40 == 0 or done == len(tasks):
                with open(SWEEP_JSON, "w") as f:
                    json.dump(
                        {
                            "configs": list(CONFIGS),
                            "baseline": BASELINE,
                            "results": results,
                        },
                        f,
                        indent=1,
                    )
                print(
                    f"[{(time.time() - start) / 60:5.1f}m] {done}/{len(tasks)}",
                    flush=True,
                )
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    with open(SWEEP_JSON, "w") as f:
        json.dump(
            {"configs": list(CONFIGS), "baseline": BASELINE, "results": results},
            f,
            indent=1,
        )
    print(f"A/B DONE: {done} solves, {(time.time() - start) / 60:.1f} min", flush=True)


# --- report ----------------------------------------------------------------- #
def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def _load():
    d = json.load(open(SWEEP_JSON))
    res = {c: r for c, r in d["results"].items() if r["levels"]}
    return (
        list(d["configs"]),
        d["baseline"],
        dict(sorted(res.items(), key=lambda kv: kv[1]["n"])),
    )


def _plot_delta(configs, baseline, data):
    """Heatmap per graph: each nested config vs baseline (% of baseline score)."""
    plt = _plt()
    nested = [c for c in configs if c not in ("crude", baseline)]
    spbs = sorted({int(s) for r in data.values() for s in r["levels"]})
    items = list(data.items())
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.4 * nrow))
    axes = axes.flatten()
    for ax, (gname, r) in zip(axes, items):
        m = []
        rows = ["crude"] + nested
        for cfg in rows:
            row = []
            for spb in spbs:
                lv = r["levels"].get(str(spb))
                if not lv or cfg not in lv["configs"] or baseline not in lv["configs"]:
                    row.append(float("nan"))
                    continue
                b = _mean(lv["configs"][baseline]["best"]) or 1
                row.append(100.0 * (_mean(lv["configs"][cfg]["best"]) - b) / b)
            m.append(row)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
        ax.set_xticks(range(len(spbs)))
        ax.set_xticklabels(spbs, fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=6)
        ax.set_title(f"{gname} (n={r['n']})", fontsize=9)
        for i in range(len(rows)):
            for j in range(len(spbs)):
                v = m[i][j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(f"Config vs '{baseline}' baseline, % (blue = better than incumbent)")
    fig.supxlabel("steps per buffer")
    fig.tight_layout()
    fig.savefig(DELTA_PNG, dpi=110)
    plt.close(fig)


def _plot_frontier(configs, baseline, data):
    """Per graph: quality vs wall-clock (mean over seeds), one line per config over
    the spb grid. Lower-left is better -- shows the efficiency frontier."""
    plt = _plt()
    spbs = sorted({int(s) for r in data.values() for s in r["levels"]})
    items = list(data.items())
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow))
    axes = axes.flatten()
    for ax, (gname, r) in zip(axes, items):
        for cfg in configs:
            xs, ys = [], []
            for spb in spbs:
                lv = r["levels"].get(str(spb))
                if lv and cfg in lv["configs"]:
                    xs.append(_mean(lv["configs"][cfg]["secs"]))
                    ys.append(_mean(lv["configs"][cfg]["best"]))
            if xs:
                style = dict(marker="o", lw=1.2, ms=3)
                if cfg == baseline:
                    style = dict(marker="s", lw=2.4, ms=5, color="k")
                elif cfg == "crude":
                    style = dict(marker="^", lw=1.2, ms=3, color="gray")
                ax.plot(xs, ys, label=cfg, **style)
        ax.set_title(f"{gname} (n={r['n']})", fontsize=9)
        ax.set_xscale("log")
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=5)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle("Efficiency frontier: quality (y) vs wall-clock seconds (x, log)")
    fig.supxlabel("seconds per solve (mean over seeds)")
    fig.supylabel("best score (lower = better)")
    fig.tight_layout()
    fig.savefig(FRONTIER_PNG, dpi=110)
    plt.close(fig)


def _headline(baseline, data):
    """Generate the headline from the data rather than narrating it by hand.

    This section used to be a hand-written constant, and it went stale the first
    time the sweep was re-run under a different objective: the prose still
    claimed regressions on two graphs that the regenerated table showed at
    parity. Numbers that describe a run belong to the run.
    """
    rows, speedups, greedy_vs_anneal = [], [], []
    for name, r in data.items():
        level = r["levels"][max(r["levels"], key=lambda s: int(s))]
        cfgs = level["configs"]
        if baseline not in cfgs:
            continue
        base_score = _mean(cfgs[baseline]["best"])
        base_secs = _mean(cfgs[baseline]["secs"])
        nested = {c: v for c, v in cfgs.items() if c.startswith("nest-")}
        if not nested or not base_score:
            continue
        best = min(nested, key=lambda c: _mean(nested[c]["best"]))
        delta = 100.0 * (_mean(nested[best]["best"]) - base_score) / base_score
        speedup = base_secs / _mean(nested[best]["secs"])
        rows.append((name, r["n"], best, delta, speedup))
        speedups.append(speedup)
        greedy = [_mean(v["best"]) for c, v in nested.items() if "greedy" in c]
        anneal = [_mean(v["best"]) for c, v in nested.items() if "anneal" in c]
        if greedy and anneal and _mean(greedy):
            greedy_vs_anneal.append(
                100.0 * (_mean(anneal) - _mean(greedy)) / _mean(greedy)
            )

    ties = [r for r in rows if r[3] <= 0.0]
    losses = sorted((r for r in rows if r[3] > 0.0), key=lambda r: -r[3])

    # How many graphs the incumbent solves identically at every run length: the
    # measure of whether extra budget buys anything at all here. Counted, not
    # asserted, because it is the premise of the paragraph that reads the
    # speedups down.
    converged = 0
    for name, r in data.items():
        scores = {
            _mean(level["configs"][baseline]["best"])
            for level in r["levels"].values()
            if baseline in level["configs"]
        }
        converged += len(scores) == 1

    # The same comparison at the budget production actually uses. Reported
    # separately because it does not agree with the long-budget result, and the
    # long-budget result is the one that would mislead a defaulting decision.
    prod, prod_speed = [], []
    for name, r in data.items():
        level = r["levels"].get(str(DEFAULT_SPB))
        if not level or baseline not in level["configs"]:
            continue
        cfgs = level["configs"]
        base_score = _mean(cfgs[baseline]["best"])
        nested = {c: v for c, v in cfgs.items() if c.startswith("nest-")}
        if not nested or not base_score:
            continue
        best = min(nested, key=lambda c: _mean(nested[c]["best"]))
        prod.append(
            (
                name,
                best,
                100.0 * (_mean(nested[best]["best"]) - base_score) / base_score,
            )
        )
        prod_speed.append(_mean(cfgs[baseline]["secs"]) / _mean(nested[best]["secs"]))
    prod_losses = sorted((p for p in prod if p[2] > 0.0), key=lambda p: -p[2])
    prod_para = (
        f"""
### At the default step budget the result reverses

The grid above starts at `steps_per_buffer={DEFAULT_SPB}`, the engine's default and the only point production runs at. There the nested engine is **behind on {len(prod_losses)} of {len(prod)} graphs and ahead on none**, by {_mean([p[2] for p in prod]):+.2f}% on average -- worst {", ".join(f"`{n}` {d:+.2f}%" for n, _, d in prod_losses[:3])}. It is still {statistics.median(prod_speed):.1f}x cheaper in solver time, but the absolute saving is fractions of a second per graph.

The nested engine needs budget to amortize: each outer structural move spends a whole inner layout loop, so a small total budget buys few structural evaluations. Its advantage therefore appears only well above the operating point, and the equal-steps framing of the table above is what makes it look unconditional. **This is the cell to read before making nested a default.**
"""
        if prod
        else ""
    )
    curves: dict = {}
    for name, r in data.items():
        level = r["levels"][max(r["levels"], key=lambda s: int(s))]
        base_score = _mean(level["configs"].get(baseline, {}).get("best", []))
        for cfg, v in level["configs"].items():
            if cfg.startswith("nest-greedy") and base_score:
                curves.setdefault(cfg, []).append(
                    100.0 * (_mean(v["best"]) - base_score) / base_score
                )
    curve_line = ", ".join(
        f"`{c.replace('nest-greedy-', '')}` {_mean(v):+.2f}%"
        for c, v in sorted(curves.items(), key=lambda kv: _mean(kv[1]))
    )
    loss_line = (
        "none -- no graph favours the incumbent"
        if not losses
        else "; ".join(f"`{n}` {d:+.2f}%" for n, _, _, d, _ in losses)
    )
    return f"""## Headline finding

**The nested engine's win is wall-clock, not plan quality. Read both numbers together or this table will mislead you.** At each graph's longest run length the best nested config scores {_mean([r[3] for r in rows]):+.2f}% against `{baseline}` -- a tie, within noise, *not* an improvement -- while taking {statistics.median(speedups):.1f}x less time ({min(speedups):.1f}x-{max(speedups):.1f}x). It ties or beats on {len(ties)} of {len(rows)} graphs. Every "speedup" below is therefore the price of the same answer, never a better one; the largest are on the largest graphs, where the incumbent's per-step full rescore is most expensive.

That framing matters because of what the extra budget itself buys, which is almost nothing: `{baseline}` returns a bit-identical score at *every* run length on {converged} of {len(rows)} graphs, so the search has already converged at the default budget. The time the nested engine saves is therefore time spent on steps that do not change the answer. A speedup on a budget nobody needs is not a reason to adopt it.

Where the win comes from:

- **Skipping the per-step full rescore.** The incumbent scores the whole state
  every step; the nested inner layout loop drives the packer's incremental
  quality and computes the full score once per outer (structural) move. Under the
  cost-model objective a full score is far more expensive than it was under the
  memory-only one, so this advantage is *larger* here than when the nested engine
  was first measured.
- **Warm-started, rotate-based inner loops.** Layout re-adapts to each structural
  change from the persisted permutation, using single-buffer reinsertions (fast
  mixing), not adjacent swaps.

Where the incumbent still wins: {loss_line}.
{prod_para}
Secondary findings:

- **Greedy vs annealed inner loop:** the annealed inner loop is worse by {_mean(greedy_vs_anneal):+.2f}% on average, and 0.00% on the graphs where every config converges to the same score. Greedy-cold remains the robust choice.
- **The inner-length curve barely separates the greedy configs** ({curve_line}), so the simplest `constant` inner length is a fine default. The "grow the inner loop over the run" hypothesis is still not strongly supported -- warm-start and rescore-skipping carry the win, not the length schedule.

_Caveats: capacity = footprint//2; y is the cost model's fixed-point prediction, not measured hardware time (the seconds here are solver compute); flash_big capped at spb {BIG_SPB_CAP}. Several graphs converge to an identical score across every config and run length, so their +0.00% is a genuine tie, not a rounding artefact._
"""


def write_report():
    configs, baseline, data = _load()
    os.makedirs(_RESULTS, exist_ok=True)
    _plot_delta(configs, baseline, data)
    _plot_frontier(configs, baseline, data)
    out = ["# Nested two-timescale SA vs the single-loop engine (A/B)\n"]
    out.append(
        f"Nested variants (greedy/annealed inner x constant/linear/convex/adaptive "
        f"length curve) vs crude and the incumbent `{baseline}`, over run length, 5 "
        f"seeds, capacity `footprint//2`. `delta%` and the frontier are both vs "
        f"`{baseline}`. Wall-clock is recorded because the nested inner loop skips "
        f"the per-step rescore, so quality-vs-time is the honest efficiency axis.\n"
    )
    out.append(_headline(baseline, data))
    out.append(f"![delta](results/{os.path.basename(DELTA_PNG)})\n")
    out.append(f"![frontier](results/{os.path.basename(FRONTIER_PNG)})\n")
    out.append("## Best nested config vs incumbent, per (graph, run length)\n")
    out.append(
        f"| graph | n | spb | {baseline} score | {baseline} s | best nested | "
        "nested % | nested s | speedup |"
    )
    out.append("|---|--:|--:|--:|--:|---|--:|--:|--:|")
    for gname, r in data.items():
        for spb in sorted(int(s) for s in r["levels"]):
            lv = r["levels"][str(spb)]
            if baseline not in lv["configs"]:
                continue
            bscore = _mean(lv["configs"][baseline]["best"]) or 1
            bsecs = _mean(lv["configs"][baseline]["secs"])
            nested = {
                c: lv["configs"][c] for c in lv["configs"] if c.startswith("nest-")
            }
            if not nested:
                continue
            best_c = min(nested, key=lambda c: _mean(nested[c]["best"]))
            nscore = _mean(nested[best_c]["best"])
            nsecs = _mean(nested[best_c]["secs"])
            out.append(
                f"| {gname} | {r['n']} | {spb} | {bscore:,.0f} | {bsecs:.2f} | "
                f"{best_c} | {100.0 * (nscore - bscore) / bscore:+.2f} | {nsecs:.2f} | "
                f"{bsecs / nsecs:.2f}x |"
            )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument(
        "--memory-only",
        action="store_true",
        help="solve against the pre-cost-model objective instead of the default",
    )
    args = ap.parse_args()
    if not args.report:
        run_sweep(smoke=args.smoke, memory_only=args.memory_only)
    write_report()
