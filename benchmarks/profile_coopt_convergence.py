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

"""Which option reaches the best score *soonest*, not which ends up lowest.

Every other benchmark in this series compares endpoints, and an endpoint goes
blind the moment every arm converges. On this corpus that happens early: 8 of 11
graphs reach their final score by ``steps_per_buffer`` 20, against a default of
40. So the sweeps have been comparing searches that had already arrived, and
reporting the resulting ties as "this knob does not matter".

For one knob that reading was wrong. ``schedule`` looks like a dead tie at the
shipping budget and is worth 23% at ``spb=2`` -- reheating converges faster, not
better. Nothing in an endpoint sweep can show that; it took suspecting it and
re-running below the default to find. This benchmark makes it visible by
construction instead, by recording the whole curve (``trace_every``) rather than
its last point.

Reported per arm:

* **steps-to-target** -- the first sampled step at which the arm's best-seen
  score is within a tolerance of the *global* best for that graph (over every arm
  and seed, so arms are measured against the same bar rather than each against
  its own plateau). Lower is faster. An arm that never reaches the bar is
  reported as such rather than dropped, since "never converges" is the finding
  the mean would hide.
* **final gap** -- where the arm ends up, for the endpoint question the rest of
  the series answers.

The two together separate the three cases a tie can be: same speed and same
destination (a genuine non-difference), same destination reached at different
speeds (what ``schedule`` turned out to be), and different destinations.

Run from the repo root::

    python3 benchmarks/profile_coopt_convergence.py
    python3 benchmarks/profile_coopt_convergence.py --smoke
    python3 benchmarks/profile_coopt_convergence.py --report
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
import statistics  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from coopt_corpus import (  # noqa: E402
    DEFAULT_SPB,
    FRESH_SEED_BASE,
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
SWEEP_JSON = os.path.join(_RESULTS, "coopt_convergence.json")
REPORT_MD = os.path.join(_BENCH, "coopt_convergence.md")
CURVES_PNG = os.path.join(_RESULTS, "coopt_convergence_curves.png")

# One arm per knob setting worth distinguishing, incumbent first. ``cycles`` is
# absent deliberately: it only shapes the reheating schedule, which stopped being
# the default, so cycle arms would now be indistinguishable copies of the
# incumbent rather than a comparison.
ARMS = {
    "incumbent": {},
    "reheating": dict(schedule="reheating"),
    "reorder=random": dict(reorder_move="random"),
    "nested": dict(nested=True, inner_annealed=False, inner_curve="constant"),
}
BASELINE = "incumbent"
# Tolerances for "has it arrived": 1% is the coarse question, 0.1% the sharp one.
TOLERANCES = [1.0, 0.1]
SEEDS = list(range(FRESH_SEED_BASE, FRESH_SEED_BASE + 5))
WORKERS = 24

_GRAPHS: dict = {}


def _init_worker():
    global _GRAPHS
    _GRAPHS = load_graphs()
    import atexit

    atexit.register(os._exit, 0)


def _solve(name, arm, seed):
    entry = _GRAPHS[name]
    bufs = copy.deepcopy(entry["buffers"])
    n = len(bufs)
    # Sample ~50 points across the run whatever the graph size, so curves are
    # comparable in shape and the trace stays small enough to serialize.
    every = max(1, (DEFAULT_SPB * n) // 50)
    solver = SaCoOptimizingSolver(
        bufs,
        max(1, foot(bufs) // 2),
        128,
        seed=seed,
        steps_per_buffer=DEFAULT_SPB,
        cost_objective=cost_objective_for(entry, bufs),
        trace_every=every,
        **ARMS[arm],
    )
    solver.plan_layout_and_core_divisions()
    return {"trace": solver.trace, "best": solver.best_score}


def _work(task):
    return task, _solve(*task)


def run_sweep(smoke=False):
    graphs = load_graphs()
    announce()
    names = ["simple_attn", "flash_attention"] if smoke else list(graphs)
    seeds = SEEDS[:1] if smoke else SEEDS
    tasks = [(n, a, s) for n in names for a in ARMS for s in seeds]
    results: dict = {}
    os.makedirs(_RESULTS, exist_ok=True)
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, arm, seed), r in pool.imap_unordered(_work, tasks, chunksize=1):
            entry = results.setdefault(
                name, {"n": len(graphs[name]["buffers"]), "arms": {}}
            )
            entry["arms"].setdefault(arm, []).append(r)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(
                    f"[{(time.time() - start) / 60:5.1f}m] {done}/{len(tasks)}",
                    flush=True,
                )
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    with open(SWEEP_JSON, "w") as f:
        json.dump({"arms": list(ARMS), "results": results}, f)
    print(f"DONE: {done} solves in {(time.time() - start) / 60:.1f} min", flush=True)


# --- report ----------------------------------------------------------------- #
def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def _steps_to(trace, target):
    """First sampled step whose best-seen score is at or below ``target``."""
    for steps, score in trace:
        if score <= target:
            return steps
    return None


def _analyse(data):
    """``{graph: {arm: {tol: mean steps or None, "final": mean end score}}}``."""
    out = {}
    for name, r in data.items():
        best = min(run["best"] for runs in r["arms"].values() for run in runs)
        per_arm = {}
        for arm, runs in r["arms"].items():
            row = {"final": _mean([run["best"] for run in runs])}
            for tol in TOLERANCES:
                target = best * (1.0 + tol / 100.0)
                hits = [_steps_to(run["trace"], target) for run in runs]
                row[str(tol)] = None if any(h is None for h in hits) else _mean(hits)
            per_arm[arm] = row
        out[name] = {"n": r["n"], "best": best, "arms": per_arm}
    return out


def _plot(analysis, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(data.items(), key=lambda kv: kv[1]["n"])
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = axes.flatten()
    for ax, (name, r) in zip(axes, items):
        best = analysis[name]["best"]
        for arm in ARMS:
            runs = r["arms"].get(arm)
            if not runs:
                continue
            trace = runs[0]["trace"]
            xs = [t[0] for t in trace]
            ys = [100.0 * (t[1] - best) / best if best else 0.0 for t in trace]
            ax.plot(
                xs,
                ys,
                lw=1.6 if arm == BASELINE else 1.0,
                ls="-" if arm == BASELINE else "--",
                label=arm,
            )
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"{name} (n={r['n']})", fontsize=9)
        ax.set_xlabel("steps", fontsize=7)
        ax.set_ylabel("% above best", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(items) :]:
        ax.axis("off")
    axes[0].legend(fontsize=6)
    fig.suptitle("Convergence: % above the best score any arm reached (seed 0)")
    fig.tight_layout()
    fig.savefig(CURVES_PNG, dpi=110)
    plt.close(fig)


def write_report():
    with open(SWEEP_JSON) as f:
        blob = json.load(f)
    data = blob["results"]
    analysis = _analyse(data)
    _plot(analysis, data)

    # Aggregate: median steps-to-target per arm, and how often each arm is the
    # first to arrive.
    agg = {arm: {str(t): [] for t in TOLERANCES} for arm in ARMS}
    firsts = {arm: 0 for arm in ARMS}
    never = {arm: 0 for arm in ARMS}
    finals = {arm: [] for arm in ARMS}
    tol_key = str(TOLERANCES[0])
    for name, r in analysis.items():
        arrivals = {}
        for arm, row in r["arms"].items():
            finals[arm].append(
                100.0 * (row["final"] - r["best"]) / r["best"] if r["best"] else 0.0
            )
            for tol in TOLERANCES:
                v = row[str(tol)]
                if v is None:
                    never[arm] += 1
                else:
                    agg[arm][str(tol)].append(v)
            if row[tol_key] is not None:
                arrivals[arm] = row[tol_key]
        if arrivals:
            best_arm = min(arrivals, key=lambda a: arrivals[a])
            firsts[best_arm] += 1

    out = ["# Convergence: which option gets there soonest\n"]
    out.append(
        f"Steps until an arm's best-seen score is within a tolerance of the best "
        f"score *any* arm reached on that graph, at the default "
        f"`steps_per_buffer={DEFAULT_SPB}`, {len(SEEDS)} seeds, capacity "
        f"`footprint//2`. Lower is sooner. `final gap` is where the arm ends up, "
        f"which is what every other benchmark in this series reports.\n"
    )
    out.append("## Per arm\n")
    out.append(
        "| arm | median steps to within 1% | to within 0.1% | mean final gap % "
        "| first to arrive (graphs) |"
    )
    out.append("|---|--:|--:|--:|--:|")
    total = len(analysis)
    for arm in ARMS:
        cells = []
        for tol in TOLERANCES:
            vals = agg[arm][str(tol)]
            # The count travels with the median. Without it a tighter tolerance
            # can read as *faster* than a looser one -- not because any run got
            # there sooner (impossible: the curve is monotone) but because the
            # graphs that never reached the tighter bar drop out of the median
            # and take their slow arrivals with them.
            cells.append(
                f"{statistics.median(vals):.0f} ({len(vals)}/{total})"
                if vals
                else f"never ({0}/{total})"
            )
        out.append(
            f"| `{arm}` | {cells[0]} | {cells[1]} | {_mean(finals[arm]):+.2f} | "
            f"{firsts[arm]} |"
        )

    out.append("\n## Per graph: steps to within 1% of the best any arm reached\n")
    out.append("| graph | n | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
    out.append("|---|--:|" + "--:|" * len(ARMS))
    for name, r in sorted(analysis.items(), key=lambda kv: kv[1]["n"]):
        cells = []
        for arm in ARMS:
            v = r["arms"].get(arm, {}).get(str(TOLERANCES[0]))
            cells.append("never" if v is None else f"{v:.0f}")
        out.append(f"| {name} | {r['n']} | " + " | ".join(cells) + " |")

    out.append(f"\n![curves](results/{os.path.basename(CURVES_PNG)})\n")
    out.append("## Reading this\n")
    out.append(
        "A tie in `final gap` with a difference in `steps to within 1%` is an arm "
        "that reaches the same place faster -- invisible to every endpoint sweep in "
        "this series, and the state `schedule` turned out to be in. A tie in both "
        "is a genuine non-difference. `never` means the arm did not reach the bar "
        "inside the default budget on some graph, which is the case a mean over "
        "the arrivals would quietly drop.\n"
    )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)
    for line in out[2:12]:
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        run_sweep(smoke=args.smoke)
    write_report()
