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

Run from the repo root::

    python3 benchmarks/profile_coopt_nested_ab.py            # full A/B + report
    python3 benchmarks/profile_coopt_nested_ab.py --smoke    # tiny subset
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
_LARGE = os.path.join(_REPO, "tests", "inductor", "cooptimization_captures_large.json")

SWEEP_JSON = os.path.join(_RESULTS, "coopt_nested_ab.json")
REPORT_MD = os.path.join(_BENCH, "coopt_nested_ab.md")
DELTA_PNG = os.path.join(_RESULTS, "coopt_nested_delta.png")
FRONTIER_PNG = os.path.join(_RESULTS, "coopt_nested_frontier.png")

SPB_GRID = [160, 640, 2560, 10240]
SEEDS = [0, 1, 2, 3, 4]
WORKERS = 32
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
    g = {c: gs[0].buffers for c, gs in load_captures().items()}
    g.update({c: gs[0].buffers for c, gs in load_captures(_LARGE).items()})
    return g


def _init_worker():
    global _GRAPHS
    _GRAPHS = _load_graphs()
    import atexit

    atexit.register(os._exit, 0)


def _work(task):
    """task = (name, spb, config, seed) -> (task, best, baseline, seconds)."""
    import copy

    name, spb, config, seed = task
    bufs = _GRAPHS[name]
    cap = max(1, _foot(bufs) // 2)
    t0 = time.time()
    s = SaCoOptimizingSolver(
        cap, 128, seed=seed, steps_per_buffer=spb, **CONFIGS[config]
    )
    s.plan_layout_and_core_divisions(copy.deepcopy(bufs))
    return task, s.best_score, s.baseline_score, time.time() - t0


def _tasks(graphs, configs, spb_grid, seeds):
    for name in graphs:
        cap_spb = BIG_SPB_CAP if len(graphs[name]) >= BIG_N else max(spb_grid)
        for spb in spb_grid:
            if spb > cap_spb:
                continue
            for config in configs:
                for seed in seeds:
                    yield (name, spb, config, seed)


def run_sweep(smoke=False):
    graphs = _load_graphs()
    spb_grid = [160, 640] if smoke else SPB_GRID
    seeds = [0] if smoke else SEEDS
    cfgs = (
        ["reheat", "nest-greedy-convex", "nest-anneal-convex"]
        if smoke
        else list(CONFIGS)
    )
    names = ["swiglu", "flash_attention"] if smoke else list(graphs)
    tasks = list(_tasks({k: graphs[k] for k in names}, cfgs, spb_grid, seeds))

    results: dict = {c: {"n": len(graphs[c]), "levels": {}} for c in names}
    os.makedirs(_RESULTS, exist_ok=True)
    start = done = 0
    start = time.time()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, spb, config, seed), best, base, secs in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            lv = results[name]["levels"].setdefault(
                str(spb),
                {
                    "total_steps": max(200, spb * results[name]["n"]),
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


_HEADLINE = """## Headline finding

**The nested two-timescale engine is substantially more time-efficient: on 9 of
11 graphs a single fixed config (`nest-greedy-constant`) matches or beats the
incumbent `reheat`'s *best* quality in 1.8-14.4x less wall-clock** (median ~2.7x;
flash_big 14.4x -- 13s vs 191s; sdpa 3.1x). The efficiency frontier (quality vs
seconds) shows the nested curves sitting left of the incumbent on those graphs.

Where the win comes from:

- **Skipping the per-step full rescore.** The incumbent scores the whole state
  every step; the nested inner layout loop drives the packer's incremental quality
  and only computes the full score once per outer (structural) move. That alone is
  most of the 2-14x speedup.
- **Warm-started, rotate-based inner loops.** Layout re-adapts to each structural
  change from the persisted permutation, using single-buffer reinsertions (fast
  mixing), not adjacent swaps.

Two honest caveats (the incumbent still wins these):

- **swiglu (+6.1%) and flash_attention (+7.7%): the incumbent's long *interleaved*
  layout refinement reaches a modestly better optimum that nested does not.** On
  the frontier their curves cross -- nested wins the short/mid-time regime, `reheat`
  wins the far right (long budget). The likely cause is that nested under-invests
  in layout on the final winning structure: layout only rides inside bursts (a
  rejected structural move's burst is discarded) plus a 20% final polish, whereas
  `reheat` refines layout continuously. A clear next lever: larger polish fraction
  or letting accepted structural moves carry deeper layout.

Secondary findings:

- **Greedy inner loop >> annealed.** The annealed inner loop is unreliable (it
  wanders when the early inner budget is small -- e.g. it missed sdpa's 40%
  division win at some seeds). Greedy-cold is the robust choice.
- **The inner-length curve barely matters among greedy configs** (constant /
  linear / convex / adaptive cluster together): the simplest `constant` inner
  length is a fine default. The "grow the inner loop over the run" hypothesis is
  *not* strongly supported by the data -- warm-start + rescore-skipping carry the
  win, not the length schedule.

_Caveats: capacity = footprint//2; y is the SA fixed-point objective, not hardware
wall-clock (the seconds here are solver compute); flash_big capped at spb 2560._
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
    out.append(_HEADLINE)
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
    args = ap.parse_args()
    if not args.report:
        run_sweep(smoke=args.smoke)
    write_report()
