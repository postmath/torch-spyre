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

"""High-power follow-up to ``profile_coopt_reorder_move.py``.

The 5-seed A/B ties on 9 of 11 captures -- every arm reaches the same score at
every budget, so those graphs carry no signal about the move at all. All the
movement is on ``flash_attention`` and ``flash_big``. This reruns just those with
many more seeds and reports a bootstrap confidence interval on the difference,
so the handful of non-tied cells can be called signal or noise rather than
eyeballed.

Run from the repo root::

    python3 docs/source/user_guide/examples/scratchpad/profile_coopt_reorder_focus.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import statistics
import time

from benchmarks.profile_coopt_reorder_move import (  # noqa: E402
    CALIB_JSON,
    CONFIGS,
    SWEEP_JSON as MAIN_JSON,
    _init_worker,
    _solve,
    _SUF,
    _REPORTS,
    _RESULTS,
    CAP_DIVISOR,
)


def _focus_graphs():
    """The graphs that actually discriminate between arms, read off the main A/B.

    This used to be the hardcoded pair ``["flash_attention", "flash_big"]`` (plus
    ``sdpa`` at tighter capacity), which was correct under the memory-only
    objective: everything else tied exactly under every arm at every budget, so
    focusing spent the seed budget where it could matter. Under the cost
    objective far more graphs respond, and a hardcoded pair would now silently
    discard most of the evidence -- the high-power check would be high-power over
    the wrong cells.

    So it is derived: any graph whose arms differ at all in the main sweep is in
    scope. If that sweep has not been run, every graph is, which is slower and
    never wrong.
    """
    try:
        with open(MAIN_JSON) as fh:
            main = json.load(fh)["results"]
    except (OSError, KeyError, ValueError):
        return sorted(json.load(open(CALIB_JSON)))
    focus = []
    for name, r in main.items():
        scores = {
            statistics.mean(cell["best"])
            for arm in r.get("arms", {}).values()
            for cell in arm.get("levels", {}).values()
            if cell.get("best")
        }
        if len(scores) > 1:
            focus.append(name)
    return focus or sorted(json.load(open(CALIB_JSON)))


SEEDS = list(range(20))
# 24, matching the main A/B. A 48-worker pool stalled on this box without making
# progress; the extra parallelism is not worth the flakiness for a run this size.
WORKERS = 24
FOCUS_JSON = os.path.join(_RESULTS, f"coopt_reorder_focus{_SUF}.json")
REPORT_MD = os.path.join(_REPORTS, f"coopt_reorder_focus{_SUF}.md")


def _work(task):
    name, spb, config, seed = task
    return task, _solve(name, spb, config, seed)


def _bootstrap_ci(a, b, iters=20000, seed=12345):
    """Percentile bootstrap CI for mean(b) - mean(a), as a % of mean(a).

    Two independent samples (the arms' RNG streams diverge once their move sets
    differ, so seeds do not pair), resampled with replacement.
    """
    rng = random.Random(seed)
    base = statistics.mean(a)
    diffs = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(100.0 * (statistics.mean(rb) - statistics.mean(ra)) / base)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    return 100.0 * (statistics.mean(b) - base) / base, lo, hi


def main():
    calib = json.load(open(CALIB_JSON))
    focus = _focus_graphs()
    tasks = []
    for name in focus:
        for config in CONFIGS:
            for spb in calib[name]["arms"][config]["spb_grid"]:
                for seed in SEEDS:
                    tasks.append((name, spb, config, seed))
    results: dict = {}
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, spb, config, seed), r in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            cell = (
                results.setdefault(name, {})
                .setdefault(config, {})
                .setdefault(str(spb), {"best": [], "cpu": []})
            )
            cell["best"].append(r["best"])
            cell["cpu"].append(r["cpu"])
            done += 1
            if done % 25 == 0 or done == len(tasks):
                with open(FOCUS_JSON, "w") as f:
                    json.dump(results, f, indent=1)
                print(
                    f"[{(time.time() - start) / 60:5.1f}m] {done}/{len(tasks)}",
                    flush=True,
                )
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    with open(FOCUS_JSON, "w") as f:
        json.dump(results, f, indent=1)

    # Report: for each (graph, incumbent level) compare each arm at the matched
    # spb its calibration assigned to that same time target.
    named = ", ".join(f"`{g}`" for g in focus)
    out = ["# Sweep vs random reorder: high-power check on the graphs that move\n"]
    out.append(
        f"{len(SEEDS)} seeds per cell (the main A/B used 5), capacity "
        f"`footprint//{CAP_DIVISOR}`. Graphs in scope ({len(focus)}): {named} -- "
        f"derived from the main sweep as the ones whose arms differ at all, not "
        f"fixed in advance, because which graphs discriminate is a property of the "
        f"objective and changed when the objective did. `delta%` is the arm's mean "
        f"score minus the incumbent's, as a percent of the incumbent's, with a 95% "
        f"percentile-bootstrap CI. Negative = the sweep is better.\n"
    )
    out.append("| graph | level | arm | cpu s | mean score | delta % | 95% CI |")
    out.append("|---|--:|---|--:|--:|--:|---|")
    verdicts = []
    for name in focus:
        base_grid = calib[name]["arms"]["random"]["spb_grid"]
        for level, base_spb in enumerate(base_grid):
            base = results[name]["random"][str(base_spb)]
            out.append(
                f"| {name} | {base_spb} | random | "
                f"{statistics.mean(base['cpu']):.2f} | "
                f"{statistics.mean(base['best']):,.0f} | -- | -- |"
            )
            for config in CONFIGS:
                if config == "random":
                    continue
                spb = calib[name]["arms"][config]["spb_grid"][level]
                cell = results[name][config][str(spb)]
                d, lo, hi = _bootstrap_ci(base["best"], cell["best"])
                sig = "sig" if (lo > 0 or hi < 0) else "ns"
                verdicts.append((name, base_spb, config, d, lo, hi, sig))
                out.append(
                    f"| {name} | {base_spb} | {config} | "
                    f"{statistics.mean(cell['cpu']):.2f} | "
                    f"{statistics.mean(cell['best']):,.0f} | {d:+.2f} | "
                    f"[{lo:+.2f}, {hi:+.2f}] {sig} |"
                )
    out.append("\n## Verdict per arm\n")
    out.append(
        "| arm | cells | sig better | sig worse | not significant | mean delta % |"
    )
    out.append("|---|--:|--:|--:|--:|--:|")
    for config in CONFIGS:
        if config == "random":
            continue
        rows = [v for v in verdicts if v[2] == config]
        better = sum(1 for v in rows if v[6] == "sig" and v[3] < 0)
        worse = sum(1 for v in rows if v[6] == "sig" and v[3] > 0)
        out.append(
            f"| {config} | {len(rows)} | {better} | {worse} | "
            f"{len(rows) - better - worse} | "
            f"{statistics.mean([v[3] for v in rows]):+.2f} |"
        )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)
    print("\n".join(out[-12:]))


if __name__ == "__main__":
    main()
