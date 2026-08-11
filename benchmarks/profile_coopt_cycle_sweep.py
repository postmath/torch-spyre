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

"""Product sweep: reheating cycle count x total run length.

The reheating schedule splits the step budget into ``cycles`` equal reheat cycles
(cycle length = total_steps / cycles). This sweeps the product space of
``cycles`` x ``steps_per_buffer`` for the reheating schedule (plus a crude
reference), 5 seeds per cell, over every captured graph, to answer: does the
optimal cycle count depend on the run length, and does reheating (at its best
cycle setting) beat crude / the default 4 cycles?

Solves are independent and deterministic per seed, so the sweep is parallelized
across processes (the box has many cores). Results are written incrementally.

Run from the repo root::

    python3 benchmarks/profile_coopt_cycle_sweep.py            # full sweep + report
    python3 benchmarks/profile_coopt_cycle_sweep.py --smoke    # tiny subset
    python3 benchmarks/profile_coopt_cycle_sweep.py --report   # report only
"""

from __future__ import annotations

import os

# The SA solver is pure Python and never calls BLAS, but importing torch in each
# of the many worker processes would otherwise spin up a full OpenBLAS/OMP thread
# pool per worker (workers x cores threads -> RLIMIT_NPROC exhaustion). Pin every
# math backend to a single thread BEFORE torch is imported (re-run per spawned
# worker, since spawn re-imports this module).
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
    """Lazily import matplotlib (kept out of the solve workers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


_BENCH = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH)
_RESULTS = os.path.join(_BENCH, "results")
_LARGE = os.path.join(_REPO, "tests", "inductor", "cooptimization_captures_large.json")

SWEEP_JSON = os.path.join(_RESULTS, "coopt_cycle_sweep.json")
REPORT_MD = os.path.join(_BENCH, "coopt_cycle_sweep.md")
HEATMAP_PNG = os.path.join(_RESULTS, "coopt_cycle_heatmap.png")
LINES_PNG = os.path.join(_RESULTS, "coopt_cycle_lines.png")

CYCLES_GRID = [1, 2, 4, 8, 16]  # 1 = single long cool (no reheat) .. 16 short reheats
SPB_GRID = [40, 160, 640, 2560, 10240]  # total run length (x n buffers)
SEEDS = [0, 1, 2, 3, 4]
WORKERS = 32

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
    # Importing torch leaves native teardown that segfaults at worker shutdown.
    # Hard-exit on a graceful stop (after results are already sent back) to skip
    # it -- keeps the log clean. Fires only via pool.close()/join(), not SIGTERM.
    import atexit

    atexit.register(os._exit, 0)


def _work(task):
    """task = (name, spb, cycles or -1 for crude, seed) -> (task, best, baseline)."""
    import copy

    name, spb, cycles, seed = task
    bufs = _GRAPHS[name]
    cap = max(1, _foot(bufs) // 2)
    kw = dict(seed=seed, steps_per_buffer=spb)
    if cycles < 0:
        kw["schedule"] = "crude"
    else:
        kw["schedule"] = "reheating"
        kw["cycles"] = cycles
    s = SaCoOptimizingSolver(copy.deepcopy(bufs), cap, 128, **kw)
    s.plan_layout_and_core_divisions()
    return task, s.best_score, s.baseline_score


def _tasks(graphs, cycles_grid, spb_grid, seeds):
    for name in graphs:
        for spb in spb_grid:
            for seed in seeds:
                yield (name, spb, -1, seed)  # crude reference
                for cyc in cycles_grid:
                    yield (name, spb, cyc, seed)


def run_sweep(smoke=False):
    graphs = _load_graphs()
    cyc_grid = [1, 4] if smoke else CYCLES_GRID
    spb_grid = [40, 160] if smoke else SPB_GRID
    seeds = [0] if smoke else SEEDS
    names = ["swiglu", "flash_attention"] if smoke else list(graphs)
    tasks = list(_tasks({k: graphs[k] for k in names}, cyc_grid, spb_grid, seeds))

    # results[name][spb] = {"n":, "baseline":, "crude":[best...], "cycles":{c:[best...]}}
    results: dict = {}
    for name in names:
        results[name] = {"n": len(graphs[name]), "levels": {}}
    os.makedirs(_RESULTS, exist_ok=True)

    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, spb, cycles, seed), best, base in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            lv = results[name]["levels"].setdefault(
                str(spb),
                {"total_steps": None, "baseline": base, "crude": [], "cycles": {}},
            )
            lv["total_steps"] = max(200, spb * results[name]["n"])
            if cycles < 0:
                lv["crude"].append(best)
            else:
                lv["cycles"].setdefault(str(cycles), []).append(best)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                with open(SWEEP_JSON, "w") as f:
                    json.dump(
                        {
                            "cycles_grid": cyc_grid,
                            "spb_grid": spb_grid,
                            "results": results,
                        },
                        f,
                        indent=1,
                    )
                print(
                    f"[{(time.time() - start) / 60:5.1f}m] {done}/{len(tasks)} solves",
                    flush=True,
                )
        pool.close()
        pool.join()  # graceful stop -> worker atexit os._exit skips torch teardown
    finally:
        pool.terminate()
    with open(SWEEP_JSON, "w") as f:
        json.dump(
            {"cycles_grid": cyc_grid, "spb_grid": spb_grid, "results": results},
            f,
            indent=1,
        )
    print(
        f"SWEEP DONE: {done} solves, {(time.time() - start) / 60:.1f} min", flush=True
    )


# --- report ----------------------------------------------------------------- #
def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def _load_results():
    d = json.load(open(SWEEP_JSON))
    res = {c: r for c, r in d["results"].items() if r["levels"]}
    res = dict(sorted(res.items(), key=lambda kv: kv[1]["n"]))
    return d["cycles_grid"], d["spb_grid"], res


def _delta_matrix(r, cyc_grid, spb_grid):
    """reheating(cycles) vs crude, mean over seeds, as % (neg = reheating better)."""
    m = []
    for cyc in cyc_grid:
        row = []
        for spb in spb_grid:
            lv = r["levels"].get(str(spb))
            if not lv or str(cyc) not in lv["cycles"]:
                row.append(float("nan"))
                continue
            rh = _mean(lv["cycles"][str(cyc)])
            cr = _mean(lv["crude"]) or 1
            row.append(100.0 * (rh - cr) / cr)
        m.append(row)
    return m


def _plot_heatmaps(cyc_grid, spb_grid, data):
    plt = _plt()
    items = list(data.items())
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = axes.flatten()
    vmax = 8
    for ax, (c, r) in zip(axes, items):
        m = _delta_matrix(r, cyc_grid, spb_grid)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(spb_grid)))
        ax.set_xticklabels(spb_grid, fontsize=7)
        ax.set_yticks(range(len(cyc_grid)))
        ax.set_yticklabels(cyc_grid, fontsize=7)
        ax.set_title(f"{c} (n={r['n']})", fontsize=9)
        for i in range(len(cyc_grid)):
            for j in range(len(spb_grid)):
                v = m[i][j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.supxlabel("steps per buffer (run length)")
    fig.supylabel("reheating cycles")
    fig.suptitle(
        "Reheating vs crude, % (blue = reheating better) across cycles x run length"
    )
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=110)
    plt.close(fig)


def _plot_lines(cyc_grid, spb_grid, data):
    """Per graph: best score vs run length, one line per cycles value + crude."""
    plt = _plt()
    items = list(data.items())
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = axes.flatten()
    for ax, (c, r) in zip(axes, items):
        xs = [
            r["levels"][str(s)]["total_steps"]
            for s in spb_grid
            if str(s) in r["levels"]
        ]
        crude = [
            _mean(r["levels"][str(s)]["crude"])
            for s in spb_grid
            if str(s) in r["levels"]
        ]
        ax.plot(xs, crude, marker="s", color="k", lw=2, label="crude", zorder=5)
        for cyc in cyc_grid:
            ys = []
            for s in spb_grid:
                lv = r["levels"].get(str(s))
                if lv and str(cyc) in lv["cycles"]:
                    ys.append(_mean(lv["cycles"][str(cyc)]))
            if ys:
                ax.plot(xs[: len(ys)], ys, marker="o", lw=1.3, label=f"{cyc} cyc")
        ax.set_xscale("log")
        ax.set_title(f"{c} (n={r['n']})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=5, ncol=2)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.supxlabel("total steps (log)")
    fig.supylabel("best score (lower = better)")
    fig.tight_layout()
    fig.savefig(LINES_PNG, dpi=110)
    plt.close(fig)


_CYCLE_HEADLINE = """## Headline finding

**The cycle count only matters on the graphs where the schedule already mattered,
and there the answer is "fewer cycles, more so at longer runs" -- the default
`cycles=4` is mildly suboptimal.**

- **8 of 11 graphs are cycle-insensitive** (softmax, rms_norm, mlp, simple_attn,
  sdpa, block_x2/x3/x4): every cycle count 1..16 lands the same score (0% spread).
  Where both schedules already converge to the same optimum, splitting the budget
  into more/fewer reheats changes nothing.
- **flash_attention (n=43): fewer cycles is better, and the ranking sharpens with
  run length.** At the longest run (spb 10240) it is monotonic: `cycles=1` -8.9%
  vs crude, 2 -8.2%, 4 -7.0%, 8 -7.4%, 16 -5.7%. At short runs the best cycle
  count is finicky and higher (2 at spb 40, 8 at spb 160) and `cycles=16` is even
  *worse* than crude at spb 40 (+2%). So the optimal cycle count **depends on run
  length** -- the product-space interaction the sweep set out to find.
- **flash_big (n=79): same shape, noisier** (spreads 1-4%): `cycles=1` is best at
  the longest run (-3.0%) but the surface is bumpy across seeds.

**Interpretation.** The reheating *cycling itself* is not where the schedule's win
over crude comes from -- a single long cool (`cycles=1`, i.e. no reheating) already
captures the full advantage (flash_attention -8.9% at the longest run). The gain
is from the schedule's per-move self-calibrating bands + cycle-phase proposal mix;
extra reheats mildly *disrupt* an otherwise-converging search at long budgets.
**Actionable:** consider lowering the default `cycles` (4 -> 1-2), at least for
larger graphs / longer budgets, or making it adaptive; the current 4 leaves ~2%
on the table on the graphs that matter.

_Caveats: capacity = footprint//2; y is the SA fixed-point objective, not
wall-clock; flash_big is noisy across seeds; cycle count only bites where the
schedule is not already saturated._
"""


def write_report():
    cyc_grid, spb_grid, data = _load_results()
    os.makedirs(_RESULTS, exist_ok=True)
    _plot_heatmaps(cyc_grid, spb_grid, data)
    _plot_lines(cyc_grid, spb_grid, data)
    out = ["# Reheating cycle-count x run-length product sweep\n"]
    out.append(
        "Reheating schedule with `cycles` in "
        f"{cyc_grid} x `steps_per_buffer` in {spb_grid}, 5 seeds/cell, capacity "
        "`footprint//2`. `cycle length = total_steps / cycles`, so `cycles=1` is a "
        "single long cool (no reheating) and larger values are more, shorter "
        "reheats. Heatmap cells are reheating-vs-crude % (blue/negative = reheating "
        "better).\n"
    )
    out.append(_CYCLE_HEADLINE)
    out.append(f"![heatmap](results/{os.path.basename(HEATMAP_PNG)})\n")
    out.append(f"![lines](results/{os.path.basename(LINES_PNG)})\n")
    out.append("## Best cycle count per (graph, run length)\n")
    out.append(
        "| graph | n | spb | total steps | best cycles | reheat(best) vs crude % | cycles=4 vs crude % |"
    )
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for c, r in data.items():
        for spb in spb_grid:
            lv = r["levels"].get(str(spb))
            if not lv:
                continue
            cr = _mean(lv["crude"]) or 1
            per_cyc = {int(k): _mean(v) for k, v in lv["cycles"].items()}
            if not per_cyc:
                continue
            best_c = min(per_cyc, key=lambda k: per_cyc[k])
            bd = 100.0 * (per_cyc[best_c] - cr) / cr
            d4 = 100.0 * (per_cyc.get(4, float("nan")) - cr) / cr
            out.append(
                f"| {c} | {r['n']} | {spb} | {lv['total_steps']} | {best_c} | "
                f"{bd:+.2f} | {d4:+.2f} |"
            )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny subset to validate")
    ap.add_argument("--report", action="store_true", help="regenerate report only")
    args = ap.parse_args()
    if not args.report:
        run_sweep(smoke=args.smoke)
    write_report()
