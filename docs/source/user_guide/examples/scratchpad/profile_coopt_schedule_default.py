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

"""Decide the co-optimizer's default schedule: ``crude`` vs ``reheating``.

Retuning ``reheating`` for the best-first sweep failed (see
``coopt_band_retune_*.md``): neither the acceptance band, the cycle count, nor
the reorder proposal weight closes its gap to ``crude``, and the one promising
weight setting did not survive held-out seeds. What *did* replicate is that
``crude`` simply wins. This is the confirmation run behind promoting it.

Deliberately wider than the question strictly needs, because a default affects
every configuration rather than the one it was measured in:

* **both capacities** -- ``footprint//2`` (where most captures saturate) and
  ``footprint//4`` (where they discriminate). A default that only wins on tight
  LX is not a default.
* **both reorder moves** -- the new ``sweep_quality`` default *and* the legacy
  ``random``. ``crude`` was found to win in the sweep regime; if it loses under
  ``random`` then the right change is a conditional default, not a global one.

Fresh seeds (30-49), out-of-sample with respect to every earlier sweep in this
series. Schedule choice does not change per-step work, so equal steps are equal
time and no wall-clock calibration is needed.

Run from the repo root::

    python3 docs/source/user_guide/examples/scratchpad/profile_coopt_schedule_default.py
    python3 docs/source/user_guide/examples/scratchpad/profile_coopt_schedule_default.py --smoke
    python3 docs/source/user_guide/examples/scratchpad/profile_coopt_schedule_default.py --report
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

# Importable both as ``python3 docs/source/user_guide/examples/scratchpad/profile_x.py`` (sys.path[0] is
# docs/source/user_guide/examples/scratchpad/) and as ``python3 -m benchmarks.profile_x`` (sys.path[0] is the repo
# root); the sibling module has to resolve either way.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from coopt_corpus import (  # noqa: E402
    DEFAULT_SPB,
    MIN_STEPS as _MIN_STEPS,
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
# Repo root: docs/source/user_guide/examples/scratchpad -> five levels up.
_REPO = os.path.abspath(os.path.join(_BENCH, "..", "..", "..", "..", ".."))
# Reports are documentation pages; their raw data and images are not.
_DOCS = os.path.join(_REPO, "docs", "source")
_REPORTS = os.path.join(_DOCS, "compiler", "benchmarks")
_RESULTS = os.path.join(_REPORTS, "data")
_IMAGES = os.path.join(_DOCS, "_static", "images", "coopt")

SWEEP_JSON = os.path.join(_RESULTS, "coopt_schedule_default.json")
REPORT_MD = os.path.join(_REPORTS, "coopt_schedule_default.md")

CAPS = [2, 4]
MOVES = ["sweep_quality", "random"]
SCHEDULES = ["crude", "reheating"]
# Grid retuned for the cost objective. These sweeps were designed when a step was
# a packer update and ~20x cheaper; scoring now dominates it, so the old top
# levels cost minutes per solve. They also buy nothing: the incumbent returns a
# bit-identical score at every run length on 9 of 11 graphs (see
# coopt_nested_ab.md), so the informative range is at and just above the default
# budget, not two orders of magnitude past it. DEFAULT_SPB leads for the same
# reason it does in the nested A/B -- a knob is decided where it will run.
SPB_GRID = [DEFAULT_SPB, 160, 640]
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


def _solve(name, cap_div, move, schedule, spb, seed):
    entry = _GRAPHS[name]
    bufs = copy.deepcopy(entry["buffers"])
    cap = max(1, _foot(bufs) // cap_div)
    s = SaCoOptimizingSolver(
        bufs,
        cap,
        128,
        cost_objective=cost_objective_for(entry, bufs),
        seed=seed,
        steps_per_buffer=spb,
        schedule=schedule,
        reorder_move=move,
    )
    c0 = time.process_time()
    s.plan_layout_and_core_divisions()
    return {"best": s.best_score, "cpu": time.process_time() - c0}


def _work(task):
    """The trailing level index rides along so the result can be filed against
    the wall-clock target rather than the (now arm-specific) spb."""
    *solve_args, level = task
    return task, _solve(*solve_args)


CALIB_JSON = os.path.join(_RESULTS, "coopt_schedule_default_calib.json")
CALIB_SPB = 160
INCUMBENT = "reheating"


def calibrate():
    """Per-step CPU for every (graph, capacity, move, schedule), and the per-arm
    ``spb`` grid that puts each schedule on the incumbent's wall-clock targets.

    Needed because this sweep used to assume the schedules cost the same per
    step, which was true under the memory-only objective and is not true now: the
    two propose different move *types* (crude ~50% reorder, reheating ~46%
    recolor), and under the cost objective a recolor dirties far more bundles
    than a reorder. Measured, reheating costs ~1.7x per step -- so comparing at
    equal steps hands crude 1.7x less machine, and calls the result a schedule
    difference.
    """
    global _GRAPHS
    _GRAPHS = _load_graphs()
    announce()
    os.makedirs(_RESULTS, exist_ok=True)
    out: dict = {}
    for name in sorted(_GRAPHS, key=lambda k: len(_GRAPHS[k]["buffers"])):
        n = len(_GRAPHS[name]["buffers"])
        out[name] = {"n": n, "arms": {}}
        for cd in CAPS:
            for mv in MOVES:
                for sch in SCHEDULES:
                    # min-of-2: timing noise is one-sided, so the minimum is the
                    # better estimate of the true cost.
                    runs = [_solve(name, cd, mv, sch, CALIB_SPB, sd) for sd in (0, 1)]
                    cpu = min(r["cpu"] for r in runs)
                    steps = max(200, CALIB_SPB * n)
                    out[name]["arms"][f"cap{cd}|{mv}|{sch}"] = {
                        "cpu_per_step_us": cpu / steps * 1e6,
                        "cpu": cpu,
                    }
        for cd in CAPS:
            for mv in MOVES:
                base = out[name]["arms"][f"cap{cd}|{mv}|{INCUMBENT}"]["cpu_per_step_us"]
                for sch in SCHEDULES:
                    arm = out[name]["arms"][f"cap{cd}|{mv}|{sch}"]
                    arm["cost_ratio"] = arm["cpu_per_step_us"] / base
                    # Same wall-clock as the incumbent at each level: a cheaper
                    # step buys proportionally more of them.
                    arm["spb_grid"] = [
                        max(1, int(round(spb / arm["cost_ratio"]))) for spb in SPB_GRID
                    ]
        ratios = " ".join(
            f"{sch}:x{out[name]['arms'][f'cap2|sweep_quality|{sch}']['cost_ratio']:.2f}"
            for sch in SCHEDULES
        )
        print(f"{name:16} n={n:3} (cap2/sweep_quality) {ratios}", flush=True)
    with open(CALIB_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", CALIB_JSON)


def run_sweep(smoke=False):
    graphs = _load_graphs()
    announce()
    names = ["sdpa", "flash_attention"] if smoke else list(graphs)
    spbs = [160] if smoke else SPB_GRID
    seeds = SEEDS[:2] if smoke else SEEDS
    if not os.path.exists(CALIB_JSON):
        raise SystemExit("run --calibrate first")
    calib = json.load(open(CALIB_JSON))
    levels = range(len(spbs))
    tasks = [
        (
            n,
            cd,
            mv,
            sch,
            calib[n]["arms"][f"cap{cd}|{mv}|{sch}"]["spb_grid"][lv],
            sd,
            lv,
        )
        for n in names
        for cd in CAPS
        for mv in MOVES
        for sch in SCHEDULES
        for lv in levels
        for sd in seeds
    ]
    results: dict = {}
    os.makedirs(_RESULTS, exist_ok=True)
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, cd, mv, sch, spb, sd, lv), r in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            key = f"cap{cd}|{mv}|L{lv}"
            cell = (
                results.setdefault(
                    name, {"n": len(graphs[name]["buffers"]), "cells": {}}
                )["cells"]
                .setdefault(key, {})
                .setdefault(sch, {"best": [], "cpu": [], "spb": None})
            )
            cell["best"].append(r["best"])
            cell["cpu"].append(r["cpu"])
            cell["spb"] = spb
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


def _bootstrap(a, b, iters=10000, seed=5):
    """95% CI for mean(b) - mean(a) as % of mean(a); b = crude, a = reheating."""
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


def write_report():
    data = json.load(open(SWEEP_JSON))
    out = ["# Should `crude` be the co-optimizer's default schedule?\n"]
    out.append(
        f"`crude` minus `reheating`, as a percent of `reheating`. **Negative means "
        f"crude is better.** Seeds {SEEDS[0]}-{SEEDS[-1]} (out-of-sample with "
        f"respect to every earlier sweep in this series).\n\n**Compared at "
        f"matched wall-clock, not matched steps.** This sweep used to assume the "
        f"schedules cost the same per step, which held under the memory-only "
        f"objective. It does not hold under the cost objective: the two propose "
        f"different move types -- crude ~50% reorder, reheating ~46% recolor -- and "
        f"a recolor rewrites a region's divisions and dirties far more bundles "
        f"than a reorder does. Reheating costs ~1.7x per step, so equal steps "
        f"handed crude 1.7x less machine and called the difference a schedule "
        f"effect. Each arm's spb is now derived from a calibration pass so both "
        f"land on the same wall-clock targets, which the per-cell cpu column lets "
        f"you check. Incumbent targets: steps-per-buffer {SPB_GRID} for "
        f"`{INCUMBENT}`.\n\nCells where both schedules reach the same score under "
        f"every seed are counted as ties, not wins.\n"
    )

    out.append("## Headline: per capacity x reorder move\n")
    out.append(
        "| capacity | reorder move | cells | tied | crude better | reheat better "
        "| mean % | pooled 95% CI |"
    )
    out.append("|---|---|--:|--:|--:|--:|--:|---|")
    headline = {}
    for cd in CAPS:
        for mv in MOVES:
            deltas, pool_r, pool_c, tied = [], [], [], 0
            for name, r in data.items():
                for lv in range(len(SPB_GRID)):
                    key = f"cap{cd}|{mv}|L{lv}"
                    cell = r["cells"].get(key)
                    if not cell or "crude" not in cell or "reheating" not in cell:
                        continue
                    rh, cr = cell["reheating"]["best"], cell["crude"]["best"]
                    if _mean(rh) == _mean(cr):
                        tied += 1
                        continue
                    deltas.append(100.0 * (_mean(cr) - _mean(rh)) / _mean(rh))
                    pool_r += [x / _mean(rh) for x in rh]
                    pool_c += [x / _mean(rh) for x in cr]
            if not deltas:
                continue
            d, lo, hi = _bootstrap(pool_r, pool_c)
            headline[(cd, mv)] = (d, lo, hi)
            out.append(
                f"| footprint//{cd} | {mv} | {len(deltas) + tied} | {tied} | "
                f"{sum(1 for x in deltas if x < 0)} | "
                f"{sum(1 for x in deltas if x > 0)} | {_mean(deltas):+.2f} | "
                f"[{lo:+.2f}, {hi:+.2f}] |"
            )

    # Where the engine's min_steps floor binds, both arms run the same number of
    # steps whatever spb the calibration assigned, so the cheaper arm simply uses
    # less time and the cells are not matched. Counted rather than hidden: they
    # are exactly the cells that tilt the aggregate toward crude.
    floored = sum(
        1
        for name, r in data.items()
        for cd in CAPS
        for mv in MOVES
        for lv in range(len(SPB_GRID))
        if (cell := r["cells"].get(f"cap{cd}|{mv}|L{lv}"))
        and cell.get("crude", {}).get("spb")
        and max(_MIN_STEPS, cell["crude"]["spb"] * r["n"]) == _MIN_STEPS
    )
    achieved = [
        _mean(cell["crude"]["cpu"]) / _mean(cell["reheating"]["cpu"])
        for r in data.values()
        for cell in r["cells"].values()
        if cell.get("crude", {}).get("cpu") and cell.get("reheating", {}).get("cpu")
    ]
    out.append(
        f"\n**Match quality.** Achieved crude/reheating CPU ratio across cells: "
        f"mean {_mean(achieved):.2f} (min {min(achieved):.2f}, max "
        f"{max(achieved):.2f}); 1.00 is a perfect match. {floored} of "
        f"{len(achieved)} cells sit on the engine's `min_steps={_MIN_STEPS}` "
        f"floor, where both arms run the same steps whatever spb was assigned, so "
        f"the cheaper arm just uses less time. Those cells are unmatched by "
        f"construction and tilt the aggregate toward crude.\n"
    )
    out.append("\n## Per graph (non-tied cells only)\n")
    out.append(
        "| graph | n | capacity | move | spb rh/cr | cpu s rh/cr | reheating | "
        "crude | delta % |"
    )
    out.append("|---|--:|---|---|--:|--:|--:|--:|--:|")
    for name, r in sorted(data.items(), key=lambda kv: kv[1]["n"]):
        for cd in CAPS:
            for mv in MOVES:
                for lv in range(len(SPB_GRID)):
                    cell = r["cells"].get(f"cap{cd}|{mv}|L{lv}")
                    if not cell or "crude" not in cell:
                        continue
                    rh, cr = (
                        _mean(cell["reheating"]["best"]),
                        _mean(cell["crude"]["best"]),
                    )
                    if rh == cr:
                        continue
                    rh_spb = cell["reheating"].get("spb")
                    cr_spb = cell["crude"].get("spb")
                    rh_cpu = _mean(cell["reheating"]["cpu"])
                    cr_cpu = _mean(cell["crude"]["cpu"])
                    out.append(
                        f"| {name} | {r['n']} | //{cd} | {mv} | "
                        f"{rh_spb}/{cr_spb} | {rh_cpu:.2f}/{cr_cpu:.2f} | "
                        f"{rh:,.0f} | {cr:,.0f} | {100.0 * (cr - rh) / rh:+.2f} |"
                    )

    # An all-tie sweep contributes no headline rows at all, and `all()` over
    # nothing is True -- which promoted crude off an empty table. A sweep that
    # cannot separate the arms is a third outcome, not a win for either.
    if not headline:
        verdict = "NO EVIDENCE -- every cell tied"
    elif all(v[2] < 0 for v in headline.values()):
        verdict = "PROMOTE crude"
    else:
        verdict = "MIXED -- see per-cell table"
    out.append(f"\n## Verdict: {verdict}\n")
    out.append(
        (
            "Both schedules reach the same score under every seed, in every cell. "
            "This sweep runs at `steps_per_buffer` >= the default, i.e. entirely "
            "above the point the search converges, so it cannot separate the arms "
            "and says nothing about which default is right. The schedules do "
            "separate below convergence; a grid that starts at the operating "
            "point cannot see it.\n"
        )
        if not headline
        else (
            "A global default flip is justified only if crude wins (CI strictly "
            "below zero) in **every** capacity x move combination. If it wins "
            "under the sweep but not under `random`, the honest change is a "
            "conditional default.\n"
        )
    )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)
    for line in out[2:12]:
        print(line)
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="measure per-step cost and derive each arm's matched-time spb grid",
    )
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    elif not args.report:
        run_sweep(smoke=args.smoke)
    if not args.calibrate:
        write_report()
