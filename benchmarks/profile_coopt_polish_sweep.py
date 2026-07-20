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

"""Sweep the nested engine's ``polish_frac`` (final layout-only refinement).

The nested A/B found nested matches/beats the incumbent 2-14x faster on 9/11
graphs but falls short on swiglu / flash_attention at long budgets, traced to
under-investing in layout on the final structure. ``polish_frac`` reserves that
fraction of the step budget for a pure-layout anneal on the best structure found.
This sweeps it (fixed ``nest-greedy-constant``) against the incumbent ``reheat``
reference, at long run lengths, 5 seeds, all captured graphs, to find whether more
polish closes the two gaps and whether it hurts the graphs nested already matched.

Parallelized; records wall-clock; results written incrementally.

Run from the repo root::

    python3 benchmarks/profile_coopt_polish_sweep.py            # sweep + report
    python3 benchmarks/profile_coopt_polish_sweep.py --smoke
    python3 benchmarks/profile_coopt_polish_sweep.py --report
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

from tests.inductor.fake_cooptimization_substrate import load_captures  # noqa: E402
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

SWEEP_JSON = os.path.join(_RESULTS, "coopt_polish_sweep.json")
REPORT_MD = os.path.join(_BENCH, "coopt_polish_sweep.md")
LINES_PNG = os.path.join(_RESULTS, "coopt_polish_lines.png")

POLISH_GRID = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7]
SPB_GRID = [640, 2560, 10240]
SEEDS = [0, 1, 2, 3, 4]
WORKERS = 32
BIG_N = 70
BIG_SPB_CAP = 2560
# Fixed nested config the polish is swept over (the robust A/B winner).
NEST_BASE = dict(nested=True, inner_annealed=False, inner_curve="constant")

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
    """task = (name, spb, kind, seed) -> (task, best, baseline, seconds).

    ``kind`` is ``"reheat"`` or a polish fraction (float)."""
    import copy

    name, spb, kind, seed = task
    bufs = _GRAPHS[name]
    cap = max(1, _foot(bufs) // 2)
    if kind == "reheat":
        kw = dict(schedule="reheating")
    else:
        kw = dict(polish_frac=float(kind), **NEST_BASE)
    t0 = time.time()
    s = SaCoOptimizingSolver(cap, 128, seed=seed, steps_per_buffer=spb, **kw)
    s.plan_layout_and_core_divs(copy.deepcopy(bufs))
    return task, s.best_score, s.baseline_score, time.time() - t0


def _kinds():
    return ["reheat"] + [str(p) for p in POLISH_GRID]


def _tasks(graphs, spb_grid, seeds):
    for name in graphs:
        cap_spb = BIG_SPB_CAP if len(graphs[name]) >= BIG_N else max(spb_grid)
        for spb in spb_grid:
            if spb > cap_spb:
                continue
            for kind in _kinds():
                for seed in seeds:
                    yield (name, spb, kind, seed)


def run_sweep(smoke=False):
    graphs = _load_graphs()
    spb_grid = [640] if smoke else SPB_GRID
    seeds = [0] if smoke else SEEDS
    names = ["swiglu", "flash_attention"] if smoke else list(graphs)
    tasks = list(_tasks({k: graphs[k] for k in names}, spb_grid, seeds))

    results: dict = {c: {"n": len(graphs[c]), "levels": {}} for c in names}
    os.makedirs(_RESULTS, exist_ok=True)
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, spb, kind, seed), best, base, secs in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            lv = results[name]["levels"].setdefault(
                str(spb),
                {
                    "total_steps": max(200, spb * results[name]["n"]),
                    "baseline": base,
                    "kinds": {},
                },
            )
            k = lv["kinds"].setdefault(kind, {"best": [], "secs": []})
            k["best"].append(best)
            k["secs"].append(secs)
            done += 1
            if done % 40 == 0 or done == len(tasks):
                with open(SWEEP_JSON, "w") as f:
                    json.dump(
                        {"polish_grid": POLISH_GRID, "results": results}, f, indent=1
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
        json.dump({"polish_grid": POLISH_GRID, "results": results}, f, indent=1)
    print(
        f"POLISH SWEEP DONE: {done} solves, {(time.time() - start) / 60:.1f} min",
        flush=True,
    )


# --- report ----------------------------------------------------------------- #
def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def _load():
    d = json.load(open(SWEEP_JSON))
    res = {c: r for c, r in d["results"].items() if r["levels"]}
    return d["polish_grid"], dict(sorted(res.items(), key=lambda kv: kv[1]["n"]))


def _plot_lines(polish_grid, data):
    plt = _plt()
    items = list(data.items())
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow))
    axes = axes.flatten()
    spbs = sorted({int(s) for r in data.values() for s in r["levels"]})
    colors = {spb: c for spb, c in zip(spbs, ["tab:blue", "tab:orange", "tab:green"])}
    for ax, (gname, r) in zip(axes, items):
        for spb in spbs:
            lv = r["levels"].get(str(spb))
            if not lv:
                continue
            ys = [
                _mean(lv["kinds"][str(p)]["best"])
                if str(p) in lv["kinds"]
                else float("nan")
                for p in polish_grid
            ]
            ax.plot(
                polish_grid,
                ys,
                marker="o",
                color=colors[spb],
                lw=1.6,
                label=f"spb={spb}",
            )
            if "reheat" in lv["kinds"]:  # incumbent reference at this spb
                ax.axhline(
                    _mean(lv["kinds"]["reheat"]["best"]),
                    color=colors[spb],
                    ls=":",
                    lw=1.3,
                    alpha=0.8,
                )
        ax.set_title(f"{gname} (n={r['n']})", fontsize=9)
        ax.set_xlabel("polish_frac", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(
        "Nested best score vs polish_frac (solid); dotted = reheat incumbent, per spb"
    )
    fig.supylabel("best score (lower = better)")
    fig.tight_layout()
    fig.savefig(LINES_PNG, dpi=110)
    plt.close(fig)


_HEADLINE = """## Headline finding

**More polish does not help -- it hurts. The final layout polish should be off
(`polish_frac=0.0`); it was a wrong hypothesis.** The nested A/B suggested raising
`polish_frac` to close the swiglu / flash_attention shortfalls; this sweep refutes
that:

- **8 of 11 graphs are polish-insensitive** (flat vs `polish_frac`): the inner
  layout loops already reach the optimum, so a final frozen-structure polish adds
  nothing.
- **flash_attention: `polish_frac=0.0` is best and more polish steadily hurts.**
  At spb 10240, `0.0` lands within +1.0% of the incumbent while `0.2` (the current
  default) is +12.3% and `0.7` is +11%. The polish steals budget from the outer
  structural loop and freezes structure on the best-so-far too early. Dropping the
  polish essentially closes the flash_attention gap.
- **swiglu: flat +6.1% at every `polish_frac`.** This gap is *structural*, not a
  layout-investment problem -- nested's outer loop never reaches the better
  division the incumbent's interleaved search finds -- so no amount of layout
  polish touches it. A separate lever (outer-loop exploration / move mix / schedule)
  is needed, not polish.

**Actionable:** set the nested default `polish_frac = 0.0` (drop the final polish).
The inner-loop layout is sufficient; the polish is a mild-to-significant
pessimization on the graphs it was meant to help. The remaining swiglu shortfall
is an outer-loop structural-exploration issue for separate study.

_Caveats: capacity = footprint//2; flash_attention is noisy across seeds at short
spb (the polish=0.2 win at spb 640 does not survive to longer runs); flash_big
capped at spb 2560._
"""


def write_report():
    polish_grid, data = _load()
    os.makedirs(_RESULTS, exist_ok=True)
    _plot_lines(polish_grid, data)
    out = ["# Nested `polish_frac` sweep\n"]
    out.append(
        "Fixed `nest-greedy-constant`, sweeping `polish_frac` in "
        f"{polish_grid} at long run lengths {SPB_GRID} (flash_big capped at "
        f"{BIG_SPB_CAP}), 5 seeds, capacity `footprint//2`. Dotted lines in the plot "
        "are the `reheat` incumbent's score at each spb; solid lines are nested vs "
        "polish_frac. `polish_frac=0.2` is the current default.\n"
    )
    out.append(_HEADLINE)
    out.append(f"![lines](results/{os.path.basename(LINES_PNG)})\n")
    out.append("## Best polish_frac vs incumbent, per (graph, run length)\n")
    out.append(
        "| graph | n | spb | reheat | default(0.2) %vs | best polish | best % vs reheat |"
    )
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for gname, r in data.items():
        for spb in sorted(int(s) for s in r["levels"]):
            lv = r["levels"][str(spb)]
            if "reheat" not in lv["kinds"]:
                continue
            rq = _mean(lv["kinds"]["reheat"]["best"]) or 1
            polish = {
                p: _mean(lv["kinds"][str(p)]["best"])
                for p in polish_grid
                if str(p) in lv["kinds"]
            }
            if not polish:
                continue
            d02 = 100.0 * (polish.get(0.2, float("nan")) - rq) / rq
            bp = min(polish, key=lambda p: polish[p])
            bd = 100.0 * (polish[bp] - rq) / rq
            out.append(
                f"| {gname} | {r['n']} | {spb} | {rq:,.0f} | {d02:+.2f} | {bp} | {bd:+.2f} |"
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
