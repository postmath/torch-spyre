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

"""Locate the capacity at which ``crude`` overtakes ``reheating``.

``coopt_schedule_default.md`` found the two schedules swap places with scratchpad
capacity, not with the reorder move: at ``footprint//2`` reheating wins by
1.6-4.5%, at ``footprint//4`` crude wins by 2.5-5.7%, and both hold under the
sweep *and* the legacy random reorder. Two sampled capacities cannot say where
the crossover sits, so this scans it.

Capacity is swept as a *ratio* of the seed footprint rather than an integer
divisor, so the grid can be fine near the suspected crossover (0.5 -> 0.25).

Also records the **spill fraction** -- the share of buffers the solver left out
of LX. Capacity ratio is an input knob; spill fraction is the pressure the solver
actually experienced, and is comparable across graphs of different shapes. If the
crossover is sharper in that coordinate, it is the better basis for a conditional
default (and for explaining *why* the schedules swap).

Both reorder moves are swept: a threshold that only holds for the current default
move would silently rot the day the move changes again.

Run from the repo root::

    python3 benchmarks/profile_coopt_capacity_crossover.py
    python3 benchmarks/profile_coopt_capacity_crossover.py --smoke
    python3 benchmarks/profile_coopt_capacity_crossover.py --report
"""

from __future__ import annotations

import os
import sys

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
import copy  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

# Importable both as ``python3 benchmarks/profile_x.py`` (sys.path[0] is
# benchmarks/) and as ``python3 -m benchmarks.profile_x`` (sys.path[0] is the repo
# root); the sibling module has to resolve either way.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from coopt_corpus import (  # noqa: E402
    FRESH_SEED_BASE,
    announce,
    cost_objective_for,
    foot as _foot,
    load_graphs,
)
from torch_spyre._inductor.scratchpad.sa_cooptimizer import (  # noqa: E402
    SaCoOptimizingSolver,
)

_BENCH = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH)
_RESULTS = os.path.join(_BENCH, "results")

SWEEP_JSON = os.path.join(_RESULTS, "coopt_capacity_crossover.json")
REPORT_MD = os.path.join(_BENCH, "coopt_capacity_crossover.md")
PNG = os.path.join(_RESULTS, "coopt_capacity_crossover.png")

# capacity = seed_footprint * RATIO. 0.5 == the old "//2", 0.25 == "//4"; the
# grid is dense between them because that is where the sign change lives.
RATIOS = [0.80, 0.60, 0.50, 0.42, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
MOVES = ["sweep_quality", "random"]
SCHEDULES = ["crude", "reheating"]
SPB = 640  # one budget; the capacity axis is what this sweep varies
SEEDS = list(range(FRESH_SEED_BASE, FRESH_SEED_BASE + 20))
WORKERS = 24

_GRAPHS: dict = {}


def _load_graphs():
    """The corpus, keyed by graph name. Entries carry buffers + features +
    bundles; see ``coopt_corpus`` for why the objective cannot be left implicit."""
    return load_graphs()


def _init_worker():
    global _GRAPHS
    _GRAPHS = _load_graphs()
    import atexit

    atexit.register(os._exit, 0)


def _solve(name, ratio, move, schedule, seed):
    entry = _GRAPHS[name]
    bufs = copy.deepcopy(entry["buffers"])
    cap = max(1, int(_foot(bufs) * ratio))
    s = SaCoOptimizingSolver(
        bufs,
        cap,
        128,
        cost_objective=cost_objective_for(entry, bufs),
        seed=seed,
        steps_per_buffer=SPB,
        schedule=schedule,
        reorder_move=move,
    )
    out = s.plan_layout_and_core_divisions()
    spilled = sum(1 for b in out if b.address is None)
    return {"best": s.best_score, "spill_frac": spilled / len(out)}


def _work(task):
    return task, _solve(*task)


def run_sweep(smoke=False):
    graphs = _load_graphs()
    announce()
    names = ["sdpa", "flash_attention"] if smoke else list(graphs)
    ratios = [0.5, 0.25] if smoke else RATIOS
    seeds = SEEDS[:2] if smoke else SEEDS
    tasks = [
        (n, r, mv, sch, sd)
        for n in names
        for r in ratios
        for mv in MOVES
        for sch in SCHEDULES
        for sd in seeds
    ]
    results: dict = {}
    os.makedirs(_RESULTS, exist_ok=True)
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, ratio, mv, sch, sd), r in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            cell = (
                results.setdefault(
                    name, {"n": len(graphs[name]["buffers"]), "cells": {}}
                )["cells"]
                .setdefault(f"{ratio}|{mv}", {})
                .setdefault(sch, {"best": [], "spill_frac": []})
            )
            cell["best"].append(r["best"])
            cell["spill_frac"].append(r["spill_frac"])
            done += 1
            if done % 300 == 0 or done == len(tasks):
                with open(SWEEP_JSON, "w") as f:
                    json.dump(results, f, indent=1)
                print(
                    f"[{(time.time() - start) / 60:5.1f}m] {done}/{len(tasks)}",
                    flush=True,
                )
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    with open(SWEEP_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"DONE: {done} solves in {(time.time() - start) / 60:.1f} min", flush=True)


# --- report ----------------------------------------------------------------- #
def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def _boot(a, b, iters=8000, seed=3):
    rng = random.Random(seed)
    base = _mean(a)
    ds = []
    for _ in range(iters):
        ra = _mean([a[rng.randrange(len(a))] for _ in range(len(a))])
        rb = _mean([b[rng.randrange(len(b))] for _ in range(len(b))])
        ds.append(100.0 * (rb - ra) / base)
    ds.sort()
    return (
        100.0 * (_mean(b) - base) / base,
        ds[int(0.025 * iters)],
        ds[int(0.975 * iters)],
    )


def _by_ratio(data, mv):
    """ratio -> (delta%, lo, hi, mean spill frac, n non-tied cells)."""
    out = {}
    for ratio in RATIOS:
        pool_r, pool_c, spills, cells = [], [], [], 0
        for name, r in data.items():
            cell = r["cells"].get(f"{ratio}|{mv}")
            if not cell or "crude" not in cell or "reheating" not in cell:
                continue
            rh, cr = cell["reheating"]["best"], cell["crude"]["best"]
            spills += cell["reheating"]["spill_frac"]
            if _mean(rh) == _mean(cr) or _mean(rh) == 0:
                continue
            cells += 1
            pool_r += [x / _mean(rh) for x in rh]
            pool_c += [x / _mean(rh) for x in cr]
        if not pool_r:
            continue
        d, lo, hi = _boot(pool_r, pool_c)
        out[ratio] = (d, lo, hi, _mean(spills), cells)
    return out


def _plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, xcoord in zip(axes, ("ratio", "spill")):
        for mv, colour in zip(MOVES, ("tab:blue", "tab:orange")):
            by = _by_ratio(data, mv)
            xs, ys, los, his = [], [], [], []
            for ratio, (d, lo, hi, spill, _) in sorted(by.items()):
                xs.append(ratio if xcoord == "ratio" else spill)
                ys.append(d)
                los.append(lo)
                his.append(hi)
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            los = [los[i] for i in order]
            his = [his[i] for i in order]
            ax.plot(xs, ys, marker="o", color=colour, label=mv, lw=1.6)
            ax.fill_between(xs, los, his, color=colour, alpha=0.15)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(
            "capacity / seed footprint" if xcoord == "ratio" else "spill fraction"
        )
        ax.set_ylabel("crude - reheating, % (negative = crude better)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(
            "by capacity ratio" if xcoord == "ratio" else "by realized spill fraction"
        )
    fig.suptitle("Where crude overtakes reheating (shaded = 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    plt.close(fig)


def _crossing(by):
    """Interpolated ratio where the delta curve crosses zero, or None."""
    pts = sorted(by.items(), reverse=True)  # loose -> tight capacity
    for (r0, v0), (r1, v1) in zip(pts, pts[1:]):
        if v0[0] > 0 >= v1[0]:
            f = v0[0] / (v0[0] - v1[0])
            return r0 + f * (r1 - r0), v0[3] + f * (v1[3] - v0[3])
    return None


def write_report():
    data = json.load(open(SWEEP_JSON))
    _plot(data)
    out = ["# Where does `crude` overtake `reheating`?\n"]
    out.append(
        f"`crude` minus `reheating` as a percent of `reheating`; **negative means "
        f"crude is better**. Capacity swept as a fraction of the seed footprint "
        f"(0.5 is the old `//2`, 0.25 the old `//4`), steps-per-buffer {SPB}, seeds "
        f"{SEEDS[0]}-{SEEDS[-1]} (out-of-sample against every earlier run). Both "
        f"reorder moves swept, so the threshold does not depend on the current "
        f"default move.\n"
    )
    out.append(f"![crossover](results/{os.path.basename(PNG)})\n")
    for mv in MOVES:
        by = _by_ratio(data, mv)
        out.append(f"\n## reorder_move = `{mv}`\n")
        out.append(
            "| capacity / footprint | spill fraction | non-tied cells | delta % | 95% CI |"
        )
        out.append("|--:|--:|--:|--:|---|")
        for ratio in sorted(by, reverse=True):
            d, lo, hi, spill, cells = by[ratio]
            sig = "" if lo < 0 < hi else " **sig**"
            out.append(
                f"| {ratio:.2f} | {spill:.2f} | {cells} | {d:+.2f}{sig} | "
                f"[{lo:+.2f}, {hi:+.2f}] |"
            )
        cross = _crossing(by)
        if cross:
            out.append(
                f"\nZero crossing (linear interpolation): capacity ratio "
                f"**{cross[0]:.3f}**, spill fraction **{cross[1]:.2f}**.\n"
            )
        else:
            out.append("\nNo sign change over the swept range.\n")
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)
    for mv in MOVES:
        by = _by_ratio(data, mv)
        print(f"\n-- {mv} --")
        for ratio in sorted(by, reverse=True):
            d, lo, hi, spill, cells = by[ratio]
            print(
                f"  ratio={ratio:.2f} spill={spill:.2f} delta={d:+7.2f} "
                f"[{lo:+.2f},{hi:+.2f}] cells={cells}"
            )
        c = _crossing(by)
        print(f"  crossing: {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        run_sweep(smoke=args.smoke)
    write_report()
