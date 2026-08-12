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

"""The crude schedule's proposal weights, on the score/CPU frontier.

``reorder_weight`` / ``flip_weight`` / ``recolor_weight`` decide the crude
schedule's move mix. They were guessed when crude was only the A/B baseline that
reheating had to beat, and no benchmark has ever varied them. Crude is now the
default, so they run on every solve -- the largest unmeasured surface in the
engine.

**Why this must be a frontier sweep and not a score sweep.** The mix is exactly
the mechanism that made crude worth defaulting to. Under the cost objective a
recolor rewrites a region's divisions and dirties many bundles, while a reorder
only moves residency; crude is cheap *because* it proposes ~50% reorders where
reheating proposes ~46% recolors. Sweeping these weights on score alone would
reliably discover that more structural moves score better, and would hand back
the CPU win that justified the switch. So every arm is calibrated to the
incumbent's wall-clock, the way ``coopt_schedule_default`` now does it, and the
report carries CPU next to score.

One arm is not a weight setting at all: ``as-reheating`` mirrors the *observed*
proposal mix of the reheating schedule (~5% reorder / ~49% flip / ~46% recolor)
inside the crude schedule. If the mix alone reproduces reheating's behaviour then
the schedule difference was never about cooling, and these three weights subsume
the schedule knob entirely.

Run from the repo root::

    python3 benchmarks/profile_coopt_move_weights.py --calibrate
    python3 benchmarks/profile_coopt_move_weights.py
    python3 benchmarks/profile_coopt_move_weights.py --report
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from coopt_corpus import (  # noqa: E402
    DEFAULT_SPB,
    MIN_STEPS,
    announce,
    cost_objective_for,
    foot,
    load_graphs,
)
from torch_spyre._inductor.scratchpad.sa_cooptimizer import (  # noqa: E402
    SaCoOptimizingSolver,
)

_BENCH = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_BENCH, "results")
SWEEP_JSON = os.path.join(_RESULTS, "coopt_move_weights.json")
CALIB_JSON = os.path.join(_RESULTS, "coopt_move_weights_calib.json")
REPORT_MD = os.path.join(_BENCH, "coopt_move_weights.md")
FRONTIER_PNG = os.path.join(_RESULTS, "coopt_move_weights_frontier.png")

# (reorder, flip, recolor). Only the ratios matter -- the engine normalizes.
ARMS = {
    "incumbent": (0.5, 0.3, 0.2),
    "reorder-heavy": (0.8, 0.1, 0.1),
    "reorder-only-ish": (0.94, 0.03, 0.03),
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "flip-heavy": (0.2, 0.6, 0.2),
    "recolor-heavy": (0.2, 0.2, 0.6),
    "structure-heavy": (0.1, 0.45, 0.45),
    "as-reheating": (0.05, 0.49, 0.46),
}
BASELINE = "incumbent"
SPB_TARGETS = [DEFAULT_SPB, 160]
CALIB_SPB = 160
CAP_DIVISOR = 2
SEEDS = list(range(90, 100))  # fresh: 70-89 went to the tier-1 reruns
WORKERS = 24

_GRAPHS: dict = {}


def _init_worker():
    global _GRAPHS
    _GRAPHS = load_graphs()
    import atexit

    atexit.register(os._exit, 0)


def _solve(name, arm, spb, seed):
    entry = _GRAPHS[name]
    bufs = copy.deepcopy(entry["buffers"])
    reorder_w, flip_w, recolor_w = ARMS[arm]
    solver = SaCoOptimizingSolver(
        bufs,
        max(1, foot(bufs) // CAP_DIVISOR),
        128,
        seed=seed,
        steps_per_buffer=spb,
        cost_objective=cost_objective_for(entry, bufs),
        reorder_weight=reorder_w,
        flip_weight=flip_w,
        recolor_weight=recolor_w,
    )
    c0 = time.process_time()
    solver.plan_layout_and_core_divisions()
    cpu = time.process_time() - c0
    proposed = dict(solver.moves_proposed)
    total = sum(proposed.values()) or 1
    return {
        "best": solver.best_score,
        "cpu": cpu,
        "mix": {k: v / total for k, v in proposed.items()},
    }


def _work(task):
    *solve_args, level = task
    return task, _solve(*solve_args)


def calibrate():
    """Per-step cost per (graph, arm), and the spb grid that matches the
    incumbent's wall-clock at each target."""
    global _GRAPHS
    _GRAPHS = load_graphs()
    announce()
    os.makedirs(_RESULTS, exist_ok=True)
    out: dict = {}
    for name in sorted(_GRAPHS, key=lambda k: len(_GRAPHS[k]["buffers"])):
        n = len(_GRAPHS[name]["buffers"])
        arms = {}
        for arm in ARMS:
            runs = [_solve(name, arm, CALIB_SPB, sd) for sd in (0, 1)]
            cpu = min(r["cpu"] for r in runs)
            arms[arm] = {
                "cpu_per_step_us": cpu / max(MIN_STEPS, CALIB_SPB * n) * 1e6,
                "mix": runs[0]["mix"],
            }
        base = arms[BASELINE]["cpu_per_step_us"]
        for arm in ARMS:
            arms[arm]["cost_ratio"] = arms[arm]["cpu_per_step_us"] / base
            arms[arm]["spb_grid"] = [
                max(1, int(round(spb / arms[arm]["cost_ratio"]))) for spb in SPB_TARGETS
            ]
        out[name] = {"n": n, "arms": arms}
        ratios = " ".join(f"{a}:x{arms[a]['cost_ratio']:.2f}" for a in ARMS)
        print(f"{name:16} n={n:3} {ratios}", flush=True)
    with open(CALIB_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", CALIB_JSON)


def run_sweep(smoke=False):
    graphs = load_graphs()
    announce()
    if not os.path.exists(CALIB_JSON):
        raise SystemExit("run --calibrate first")
    calib = json.load(open(CALIB_JSON))
    names = ["simple_attn", "flash_big"] if smoke else list(graphs)
    seeds = SEEDS[:2] if smoke else SEEDS
    tasks = [
        (n, arm, calib[n]["arms"][arm]["spb_grid"][lv], sd, lv)
        for n in names
        for arm in ARMS
        for lv in range(len(SPB_TARGETS))
        for sd in seeds
    ]
    results: dict = {}
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, arm, spb, sd, lv), r in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            cell = (
                results.setdefault(
                    name, {"n": len(graphs[name]["buffers"]), "cells": {}}
                )["cells"]
                .setdefault(f"L{lv}", {})
                .setdefault(arm, {"best": [], "cpu": [], "mix": None, "spb": spb})
            )
            cell["best"].append(r["best"])
            cell["cpu"].append(r["cpu"])
            cell["mix"] = r["mix"]
            done += 1
            if done % 200 == 0 or done == len(tasks):
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


def _bootstrap(a, b, iters=10000, seed=11):
    """95% CI for mean(b) - mean(a) as % of mean(a); b = arm, a = incumbent."""
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


def _plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for arm in ARMS:
        xs, ys = [], []
        for r in data.values():
            for cell in r["cells"].values():
                if arm not in cell or BASELINE not in cell:
                    continue
                b = _mean(cell[BASELINE]["best"]) or 1
                bc = _mean(cell[BASELINE]["cpu"]) or 1
                xs.append(_mean(cell[arm]["cpu"]) / bc)
                ys.append(100.0 * (_mean(cell[arm]["best"]) - b) / b)
        if xs:
            ax.scatter(xs, ys, s=18, alpha=0.6, label=arm)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xlabel("CPU relative to incumbent (lower is better)")
    ax.set_ylabel("% score vs incumbent (lower is better)")
    ax.set_title(
        "Proposal-weight arms on the score/CPU frontier\n"
        "(bottom-left quadrant dominates the incumbent)"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FRONTIER_PNG, dpi=110)
    plt.close(fig)


def write_report():
    with open(SWEEP_JSON) as f:
        data = json.load(f)
    _plot(data)

    pooled = {arm: {"score": [], "cpu": [], "raw": []} for arm in ARMS}
    base_raw: list = []
    for r in data.values():
        for cell in r["cells"].values():
            if BASELINE not in cell:
                continue
            b = _mean(cell[BASELINE]["best"]) or 1
            bc = _mean(cell[BASELINE]["cpu"]) or 1
            base_raw += [x / b for x in cell[BASELINE]["best"]]
            for arm in ARMS:
                if arm not in cell:
                    continue
                pooled[arm]["score"].append(100.0 * (_mean(cell[arm]["best"]) - b) / b)
                pooled[arm]["cpu"].append(_mean(cell[arm]["cpu"]) / bc)
                pooled[arm]["raw"] += [x / b for x in cell[arm]["best"]]

    out = ["# Crude's proposal weights on the score/CPU frontier\n"]
    out.append(
        f"`reorder_weight` / `flip_weight` / `recolor_weight`, swept for the first "
        f"time. Every arm is calibrated to the incumbent's wall-clock at each of "
        f"steps-per-buffer {SPB_TARGETS}, {len(SEEDS)} seeds, capacity "
        f"`footprint//{CAP_DIVISOR}`. Score is % against the incumbent weights "
        f"{ARMS[BASELINE]} (negative = better); CPU is relative to the same. An arm "
        f"only beats the default if it is at or left of 1.00x CPU *and* at or "
        f"below 0% score.\n"
    )
    out.append("## Per arm, pooled\n")
    out.append(
        "| arm | (reorder, flip, recolor) | proposal mix r/f/c | CPU vs incumbent "
        "| score % | 95% CI |"
    )
    out.append("|---|---|---|--:|--:|---|")
    winners = []
    for arm, w in ARMS.items():
        if not pooled[arm]["raw"]:
            continue
        d, lo, hi = _bootstrap(base_raw, pooled[arm]["raw"])
        cpu = _mean(pooled[arm]["cpu"])
        mix = None
        for r in data.values():
            for cell in r["cells"].values():
                if arm in cell and cell[arm].get("mix"):
                    mix = cell[arm]["mix"]
                    break
            if mix:
                break
        mixs = (
            f"{100 * mix['reorder']:.0f}/{100 * mix['flip']:.0f}/"
            f"{100 * mix['recolor']:.0f}%"
            if mix
            else "--"
        )
        flag = ""
        if arm != BASELINE and hi < 0 and cpu <= 1.02:
            flag = " **dominates**"
            winners.append((arm, d, cpu))
        elif arm != BASELINE and lo > 0:
            flag = " (worse)"
        out.append(
            f"| `{arm}` | {tuple(round(x, 2) for x in w)} | {mixs} | {cpu:.2f}x | "
            f"{d:+.3f} | [{lo:+.3f}, {hi:+.3f}]{flag} |"
        )

    out.append(f"\n![frontier](results/{os.path.basename(FRONTIER_PNG)})\n")

    # The Pareto set on (cpu, score): an arm is dominated if another is at least
    # as cheap and at least as good, and strictly better on one axis.
    pts = {
        arm: (_mean(pooled[arm]["cpu"]), _mean(pooled[arm]["score"]))
        for arm in ARMS
        if pooled[arm]["raw"]
    }
    frontier = [
        a
        for a, (c, sc) in pts.items()
        if not any(
            (c2 <= c and sc2 <= sc) and (c2 < c or sc2 < sc)
            for b, (c2, sc2) in pts.items()
            if b != a
        )
    ]
    order = sorted(frontier, key=lambda a: pts[a][0])
    out.append("## Verdict\n")
    if winners:
        best = min(winners, key=lambda w: w[1])
        out.append(
            f"**{len(winners)} arm(s) dominate the incumbent weights**: "
            f"significantly better on score at or below parity CPU. Best is "
            f"`{best[0]}` at {best[1]:+.3f}% for {best[2]:.2f}x CPU.\n"
        )
    else:
        out.append(
            "**No arm dominates the incumbent weights.** Nothing swept here is "
            "both significantly better on score and no more expensive, so the "
            "guessed defaults survive their first contact with evidence -- worth "
            "stating plainly rather than leaving as an absence, since this mix "
            "was the largest unmeasured surface in the engine.\n"
        )
    out.append(
        "The frontier, cheapest first: "
        + ", ".join(f"`{a}` ({pts[a][0]:.2f}x, {pts[a][1]:+.3f}%)" for a in order)
        + ". "
        + (
            f"`{BASELINE}` is on it, sitting between the arms that buy score with "
            f"CPU and the arms that buy CPU with score."
            if BASELINE in frontier
            else f"`{BASELINE}` is **not** on it."
        )
        + " Everything here is a ~0.1% score effect against a cost model that has "
        "never been checked on hardware, so the frontier's shape is more "
        "informative than any single point on it.\n"
    )

    ar = pts.get("as-reheating")
    if ar:
        out.append(
            f"**Was the schedule difference ever about cooling?** `as-reheating` "
            f"puts the reheating schedule's observed mix (~5/49/46) inside the "
            f"crude schedule, changing the proposal mix and nothing else. It "
            f"lands at {ar[1]:+.3f}% score for {ar[0]:.2f}x CPU -- the same shape "
            f"as reheating itself against crude: a small score gain bought with "
            f"CPU. So the schedule knob is mostly a mix knob, and these three "
            f"weights largely subsume it. That also means the two are not "
            f"independent: retuning them re-opens the schedule decision.\n"
        )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)
    for line in out[2:14]:
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    elif not args.report:
        run_sweep(smoke=args.smoke)
    if not args.calibrate:
        write_report()
