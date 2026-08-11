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

    python3 benchmarks/profile_coopt_schedule_default.py
    python3 benchmarks/profile_coopt_schedule_default.py --smoke
    python3 benchmarks/profile_coopt_schedule_default.py --report
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
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

from tests.inductor.cooptimization_capture_loader import load_captures  # noqa: E402
from torch_spyre._inductor.scratchpad.sa_cooptimizer import (  # noqa: E402
    SaCoOptimizingSolver,
)

_BENCH = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH)
_RESULTS = os.path.join(_BENCH, "results")
_LARGE = os.path.join(_REPO, "tests", "inductor", "cooptimization_captures_large.json")

SWEEP_JSON = os.path.join(_RESULTS, "coopt_schedule_default.json")
REPORT_MD = os.path.join(_BENCH, "coopt_schedule_default.md")

CAPS = [2, 4]
MOVES = ["sweep_quality", "random"]
SCHEDULES = ["crude", "reheating"]
SPB_GRID = [160, 640, 2560]
SEEDS = list(range(30, 50))
WORKERS = 24

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


def _solve(name, cap_div, move, schedule, spb, seed):
    bufs = _GRAPHS[name]
    cap = max(1, _foot(bufs) // cap_div)
    s = SaCoOptimizingSolver(
        copy.deepcopy(bufs),
        cap,
        128,
        seed=seed,
        steps_per_buffer=spb,
        max_steps=10**9,
        schedule=schedule,
        reorder_move=move,
    )
    c0 = time.process_time()
    s.plan_layout_and_core_divisions()
    return {"best": s.best_score, "cpu": time.process_time() - c0}


def _work(task):
    return task, _solve(*task)


def run_sweep(smoke=False):
    graphs = _load_graphs()
    names = ["sdpa", "flash_attention"] if smoke else list(graphs)
    spbs = [160] if smoke else SPB_GRID
    seeds = SEEDS[:2] if smoke else SEEDS
    tasks = [
        (n, cd, mv, sch, spb, sd)
        for n in names
        for cd in CAPS
        for mv in MOVES
        for sch in SCHEDULES
        for spb in spbs
        for sd in seeds
    ]
    results: dict = {}
    os.makedirs(_RESULTS, exist_ok=True)
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, cd, mv, sch, spb, sd), r in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            key = f"cap{cd}|{mv}|{spb}"
            cell = (
                results.setdefault(name, {"n": len(graphs[name]), "cells": {}})["cells"]
                .setdefault(key, {})
                .setdefault(sch, {"best": [], "cpu": []})
            )
            cell["best"].append(r["best"])
            cell["cpu"].append(r["cpu"])
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
        f"respect to every earlier sweep in this series), steps-per-buffer "
        f"{SPB_GRID}. Schedule choice does not change per-step work, so equal steps "
        f"are equal time.\n\nCells where both schedules reach the same score under "
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
                for spb in SPB_GRID:
                    key = f"cap{cd}|{mv}|{spb}"
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

    out.append("\n## Per graph (non-tied cells only)\n")
    out.append("| graph | n | capacity | move | spb | reheating | crude | delta % |")
    out.append("|---|--:|---|---|--:|--:|--:|--:|")
    for name, r in sorted(data.items(), key=lambda kv: kv[1]["n"]):
        for cd in CAPS:
            for mv in MOVES:
                for spb in SPB_GRID:
                    cell = r["cells"].get(f"cap{cd}|{mv}|{spb}")
                    if not cell or "crude" not in cell:
                        continue
                    rh, cr = (
                        _mean(cell["reheating"]["best"]),
                        _mean(cell["crude"]["best"]),
                    )
                    if rh == cr:
                        continue
                    out.append(
                        f"| {name} | {r['n']} | //{cd} | {mv} | {spb} | {rh:,.0f} | "
                        f"{cr:,.0f} | {100.0 * (cr - rh) / rh:+.2f} |"
                    )

    verdict = (
        "PROMOTE crude"
        if all(v[2] < 0 for v in headline.values())
        else "MIXED -- see per-cell table"
    )
    out.append(f"\n## Verdict: {verdict}\n")
    out.append(
        "A global default flip is justified only if crude wins (CI strictly below "
        "zero) in **every** capacity x move combination. If it wins under the sweep "
        "but not under `random`, the honest change is a conditional default.\n"
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
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        run_sweep(smoke=args.smoke)
    write_report()
