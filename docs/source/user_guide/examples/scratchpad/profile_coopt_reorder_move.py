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

"""A/B: best-first reinsertion sweep vs the random ``(i, j)`` rotation.

The co-optimizer's ``reorder`` move rotates a random buffer to a random position
and judges that one sample. The layout-only annealer instead lifts one buffer and
scans *every* reinsertion position, trying them best-first. This asks whether
porting that move into the co-optimizer pays off **at equal wall-clock**, since a
sweep step is strictly more expensive than a random one and the public API budgets
steps, not time.

Arms (``reorder_move`` / ``sweep_*`` knobs on ``SaCoOptimizingSolver``):

- ``random``        -- the incumbent.
- ``sweep-q``       -- sweep ranked by the packer's O(1) ``quality()`` proxy.
- ``sweep-s``       -- sweep ranked by the true objective at every position.
- ``sweep-q-unbi``  -- ``sweep-q`` without the unallocated-biased choice of ``i``,
                       the control that separates "sweep over j" from "bias over i".

Two phases:

``--calibrate`` measures seconds per step for every (graph, arm) so the sweep
arms' ``steps_per_buffer`` grids can be chosen to land on the *same* wall-clock
targets as the incumbent's. Comparing arms at a shared spb grid would compare
different time budgets, which is exactly the confound the question is about.

The sweep proper then runs every arm over its calibrated spb grid x 5 seeds and
reports the quality-vs-time frontier plus an iso-time table: each arm's score
interpolated to the incumbent's wall-clock at each of its spb levels.

Run from the repo root::

    python3 docs/source/user_guide/examples/scratchpad/profile_coopt_reorder_move.py --calibrate
    python3 docs/source/user_guide/examples/scratchpad/profile_coopt_reorder_move.py
    python3 docs/source/user_guide/examples/scratchpad/profile_coopt_reorder_move.py --report
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
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

# Importable both as ``python3 docs/source/user_guide/examples/scratchpad/profile_x.py`` (sys.path[0] is
# docs/source/user_guide/examples/scratchpad/) and as ``python3 -m benchmarks.profile_x`` (sys.path[0] is the repo
# root); the sibling module has to resolve either way.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from coopt_corpus import (  # noqa: E402
    DEFAULT_SPB,
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

_CAP_DIVISOR_ENV = "COOPT_CAP_DIVISOR"
# Suffix every artifact when the capacity is not the default, so a tighter-LX run
# sits alongside the baseline one instead of overwriting it.
_SUF = (
    ""
    if os.environ.get(_CAP_DIVISOR_ENV, "2") == "2"
    else ("_cap" + os.environ[_CAP_DIVISOR_ENV])
)

CALIB_JSON = os.path.join(_RESULTS, f"coopt_reorder_move_calib{_SUF}.json")
SWEEP_JSON = os.path.join(_RESULTS, f"coopt_reorder_move{_SUF}.json")
REPORT_MD = os.path.join(_REPORTS, f"coopt_reorder_move{_SUF}.md")
FRONTIER_PNG = os.path.join(_IMAGES, f"coopt_reorder_frontier{_SUF}.png")
ISOTIME_PNG = os.path.join(_IMAGES, f"coopt_reorder_isotime{_SUF}.png")

CONFIGS = {
    "random": dict(reorder_move="random"),
    "sweep-q": dict(reorder_move="sweep_quality"),
    "sweep-s": dict(reorder_move="sweep_score"),
    "sweep-q-unbi": dict(reorder_move="sweep_quality", sweep_biased_i=False),
}
BASELINE = "random"

# The incumbent's step-budget grid. It defines the wall-clock targets; every
# other arm's grid is derived from the calibration so the times line up.
# Grid retuned for the cost objective. These sweeps were designed when a step was
# a packer update and ~20x cheaper; scoring now dominates it, so the old top
# levels cost minutes per solve. They also buy nothing: the incumbent returns a
# bit-identical score at every run length on 9 of 11 graphs (see
# coopt_nested_ab.md), so the informative range is at and just above the default
# budget, not two orders of magnitude past it. DEFAULT_SPB leads for the same
# reason it does in the nested A/B -- a knob is decided where it will run.
BASE_SPB_GRID = [DEFAULT_SPB, 160, 640]
CALIB_SPB = 640
# Scratchpad capacity as footprint // CAP_DIVISOR. 2 is what the sibling coopt
# benchmarks use; a larger divisor tightens LX and keeps the easier captures from
# saturating (at //2 nine of eleven reach the same score under every arm, which
# leaves the move-set comparison resting on two graphs).
CAP_DIVISOR = int(os.environ.get(_CAP_DIVISOR_ENV, "2"))
SEEDS = list(range(FRESH_SEED_BASE, FRESH_SEED_BASE + 5))
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


def _solve(name, spb, config, seed):
    """One solve -> (best, baseline, seconds, cpu_seconds, steps, sweep counters)."""
    entry = _GRAPHS[name]
    bufs = copy.deepcopy(entry["buffers"])
    n = len(bufs)
    cap = max(1, _foot(bufs) // CAP_DIVISOR)
    s = SaCoOptimizingSolver(
        bufs,
        cap,
        128,
        seed=seed,
        # Priced into the wall-clock calibration below, not bolted on after it:
        # the cost objective is most of the per-step cost now, so an arm's
        # spb grid must be derived with it in place or the arms are compared at
        # matched *step* budgets while claiming matched time.
        cost_objective=cost_objective_for(entry, bufs),
        steps_per_buffer=spb,
        **CONFIGS[config],
    )
    t0, c0 = time.perf_counter(), time.process_time()
    s.plan_layout_and_core_divisions()
    secs = time.perf_counter() - t0
    cpu = time.process_time() - c0
    # Mirrors the engine's own budget, so cost-per-step is priced against the
    # steps that actually ran rather than the ones asked for.
    steps = max(200, spb * n)
    return {
        "best": s.best_score,
        "baseline": s.baseline_score,
        "secs": secs,
        "cpu": cpu,
        "steps": steps,
        "sweep_steps": s.sweep_steps,
        "sweep_probes": s.sweep_probes,
        "sweep_evals": s.sweep_evals,
        "reorder_proposed": s.moves_proposed["reorder"],
        "reorder_accepted": s.moves_accepted["reorder"],
    }


# --- calibration ------------------------------------------------------------ #
def run_calibration():
    """Seconds per step for every (graph, arm), run serially so the timings are
    uncontended, then the derived spb grid that puts each arm on the incumbent's
    wall-clock targets."""
    global _GRAPHS
    _GRAPHS = _load_graphs()
    os.makedirs(_RESULTS, exist_ok=True)
    out: dict = {}
    for name in sorted(_GRAPHS, key=lambda k: len(_GRAPHS[k]["buffers"])):
        n = len(_GRAPHS[name]["buffers"])
        out[name] = {"n": n, "arms": {}}
        for config in CONFIGS:
            # Two seeds, min-of: timing noise is one-sided, so the minimum is the
            # better estimate of the true cost than a mean over a noisy machine.
            runs = [_solve(name, CALIB_SPB, config, seed) for seed in (0, 1)]
            r = min(runs, key=lambda r: r["cpu"])
            per_step = r["cpu"] / r["steps"]
            out[name]["arms"][config] = {
                "cpu_per_step_us": per_step * 1e6,
                "secs": r["secs"],
                "cpu": r["cpu"],
                "steps": r["steps"],
                "probes_per_sweep": (
                    r["sweep_probes"] / r["sweep_steps"] if r["sweep_steps"] else 0.0
                ),
                "evals_per_sweep": (
                    r["sweep_evals"] / r["sweep_steps"] if r["sweep_steps"] else 0.0
                ),
                "reorder_accept": (
                    r["reorder_accepted"] / r["reorder_proposed"]
                    if r["reorder_proposed"]
                    else 0.0
                ),
            }
        base = out[name]["arms"][BASELINE]["cpu_per_step_us"]
        for config in CONFIGS:
            arm = out[name]["arms"][config]
            arm["cost_ratio"] = arm["cpu_per_step_us"] / base
            # Match the incumbent's *time* at each of its spb levels: an arm that
            # costs r x more per step gets 1/r the steps.
            arm["spb_grid"] = [
                max(1, int(round(spb / arm["cost_ratio"]))) for spb in BASE_SPB_GRID
            ]
        print(
            f"{name:16s} n={n:3d} "
            + "  ".join(
                f"{c}:{out[name]['arms'][c]['cpu_per_step_us']:6.1f}us"
                f"(x{out[name]['arms'][c]['cost_ratio']:.2f})"
                for c in CONFIGS
            ),
            flush=True,
        )
    with open(CALIB_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", CALIB_JSON)
    return out


# --- sweep ------------------------------------------------------------------ #
def _work(task):
    name, spb, config, seed = task
    return task, _solve(name, spb, config, seed)


def _tasks(calib, seeds):
    for name, c in calib.items():
        for config in CONFIGS:
            for spb in c["arms"][config]["spb_grid"]:
                for seed in seeds:
                    yield (name, spb, config, seed)


def run_sweep(smoke=False):
    if not os.path.exists(CALIB_JSON):
        raise SystemExit("run --calibrate first")
    calib = json.load(open(CALIB_JSON))
    announce()
    if smoke:
        calib = {k: v for k, v in calib.items() if k in ("swiglu", "sdpa")}
    seeds = [0] if smoke else SEEDS
    tasks = list(_tasks(calib, seeds))
    results: dict = {name: {"n": c["n"], "arms": {}} for name, c in calib.items()}
    os.makedirs(_RESULTS, exist_ok=True)
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, spb, config, seed), r in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            lv = (
                results[name]["arms"]
                .setdefault(config, {})
                .setdefault(
                    str(spb),
                    {
                        "best": [],
                        "secs": [],
                        "cpu": [],
                        "steps": r["steps"],
                        "baseline": r["baseline"],
                    },
                )
            )
            lv["best"].append(r["best"])
            lv["secs"].append(r["secs"])
            lv["cpu"].append(r["cpu"])
            done += 1
            if done % 50 == 0 or done == len(tasks):
                with open(SWEEP_JSON, "w") as f:
                    json.dump({"baseline": BASELINE, "results": results}, f, indent=1)
                print(
                    f"[{(time.time() - start) / 60:5.1f}m] {done}/{len(tasks)}",
                    flush=True,
                )
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    with open(SWEEP_JSON, "w") as f:
        json.dump({"baseline": BASELINE, "results": results}, f, indent=1)
    print(f"DONE: {done} solves, {(time.time() - start) / 60:.1f} min", flush=True)


# --- report ----------------------------------------------------------------- #
def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _curve(arm_levels, xkey="cpu"):
    """(times, scores) for one arm, sorted by time, means over seeds."""
    pts = sorted((_mean(v[xkey]), _mean(v["best"])) for v in arm_levels.values())
    return [p[0] for p in pts], [p[1] for p in pts]


def _interp(xs, ys, x):
    """Linear interpolation of the score curve in log-time; clamped at the ends.
    Returns None when x is outside the arm's measured range by more than 15%, so
    the iso-time table never extrapolates a win out of thin air."""
    if not xs:
        return None
    lx = math.log(x)
    lxs = [math.log(v) for v in xs]
    if lx <= lxs[0]:
        return ys[0] if lx > lxs[0] - 0.14 else None
    if lx >= lxs[-1]:
        return ys[-1] if lx < lxs[-1] + 0.14 else None
    for k in range(1, len(lxs)):
        if lx <= lxs[k]:
            f = (lx - lxs[k - 1]) / (lxs[k] - lxs[k - 1])
            return ys[k - 1] + f * (ys[k] - ys[k - 1])
    return ys[-1]


def _isotime_rows(data):
    """Per (graph, incumbent time level): each arm's score interpolated to that
    exact wall-clock, as % of the incumbent's score there."""
    rows = []
    for name, r in data.items():
        base = r["arms"].get(BASELINE)
        if not base:
            continue
        bx, by = _curve(base)
        for spb in sorted(base, key=int):
            t = _mean(base[spb]["cpu"])
            cells = {}
            for config in CONFIGS:
                if config not in r["arms"]:
                    continue
                xs, ys = _curve(r["arms"][config])
                v = _interp(xs, ys, t)
                cells[config] = v
            ref = cells.get(BASELINE)
            if not ref:
                continue
            rows.append((name, r["n"], int(spb), t, ref, cells))
    return rows


def _plot_frontier(data):
    plt = _plt()
    items = sorted(data.items(), key=lambda kv: kv[1]["n"])
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow))
    axes = axes.flatten()
    for ax, (name, r) in zip(axes, items):
        for config in CONFIGS:
            if config not in r["arms"]:
                continue
            xs, ys = _curve(r["arms"][config])
            style = dict(marker="o", lw=1.3, ms=3)
            if config == BASELINE:
                style = dict(marker="s", lw=2.4, ms=5, color="k")
            ax.plot(xs, ys, label=config, **style)
        ax.set_title(f"{name} (n={r['n']})", fontsize=9)
        ax.set_xscale("log")
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=6)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle("Efficiency frontier: score (y) vs CPU seconds (x, log)")
    fig.supxlabel("CPU seconds per solve (mean over seeds)")
    fig.supylabel("best score (lower = better)")
    fig.tight_layout()
    fig.savefig(FRONTIER_PNG, dpi=110)
    plt.close(fig)


def _plot_isotime(rows):
    plt = _plt()
    arms = [c for c in CONFIGS if c != BASELINE]
    names = sorted(
        {r[0] for r in rows}, key=lambda nm: [r[1] for r in rows if r[0] == nm][0]
    )
    levels = sorted({r[2] for r in rows})
    fig, axes = plt.subplots(
        1, len(arms), figsize=(5.0 * len(arms), 4.4), squeeze=False
    )
    for ax, arm in zip(axes[0], arms):
        m = []
        for name in names:
            row = []
            for lv in levels:
                cell = [r for r in rows if r[0] == name and r[2] == lv]
                if not cell or cell[0][5].get(arm) is None:
                    row.append(float("nan"))
                else:
                    ref = cell[0][4]
                    row.append(100.0 * (cell[0][5][arm] - ref) / ref if ref else 0.0)
            m.append(row)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{arm} vs {BASELINE}, iso-time %", fontsize=10)
        for a in range(len(names)):
            for b in range(len(levels)):
                if not math.isnan(m[a][b]):
                    ax.text(
                        b, a, f"{m[a][b]:+.1f}", ha="center", va="center", fontsize=6
                    )
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.supxlabel("incumbent steps-per-buffer level (defines the time target)")
    fig.suptitle("Iso-time comparison (blue = sweep better at the same wall-clock)")
    fig.tight_layout()
    fig.savefig(ISOTIME_PNG, dpi=110)
    plt.close(fig)


def write_report():
    d = json.load(open(SWEEP_JSON))
    data = {k: v for k, v in d["results"].items() if v["arms"]}
    calib = json.load(open(CALIB_JSON)) if os.path.exists(CALIB_JSON) else {}
    rows = _isotime_rows(data)
    _plot_frontier(data)
    _plot_isotime(rows)

    out = ["# Best-first reinsertion sweep vs random rotation (co-optimizer A/B)\n"]
    out.append(
        "The co-optimizer's `reorder` move rotates a random buffer to a random "
        "position; the layout-only annealer sweeps every reinsertion position and "
        "takes them best-first. Arms are compared at **equal wall-clock**: each "
        "arm's `steps_per_buffer` grid is derived from a calibration pass so its "
        "solve times land on the incumbent's. CPU time (`process_time`) is the "
        "cost axis, so pool contention cannot flatter an arm.\n"
    )

    if calib:
        out.append("## Calibration: cost per step\n")
        out.append(
            "| graph | n | "
            + " | ".join(f"{c} us/step" for c in CONFIGS)
            + " | sweep probes | sweep evals | reorder accept |"
        )
        out.append("|---|--:|" + "--:|" * (len(CONFIGS) + 3))
        for name, c in sorted(calib.items(), key=lambda kv: kv[1]["n"]):
            a = c["arms"]
            out.append(
                f"| {name} | {c['n']} | "
                + " | ".join(
                    f"{a[k]['cpu_per_step_us']:.1f} (x{a[k]['cost_ratio']:.2f})"
                    for k in CONFIGS
                )
                + f" | {a['sweep-q']['probes_per_sweep']:.1f} "
                f"| {a['sweep-q']['evals_per_sweep']:.2f} "
                f"| {a['sweep-q']['reorder_accept']:.2f} |"
            )
        out.append("")

    out.append(
        f"![frontier](../../_static/images/coopt/{os.path.basename(FRONTIER_PNG)})\n"
    )
    out.append(
        f"![isotime](../../_static/images/coopt/{os.path.basename(ISOTIME_PNG)})\n"
    )

    out.append("## Iso-time comparison\n")
    out.append(
        f"| graph | n | level | cpu s | {BASELINE} score | "
        + " | ".join(f"{c} %" for c in CONFIGS if c != BASELINE)
        + " |"
    )
    out.append("|---|--:|--:|--:|--:|" + "--:|" * (len(CONFIGS) - 1))
    tally: dict = {c: [] for c in CONFIGS if c != BASELINE}
    for name, n, lv, t, ref, cells in rows:
        cols = []
        for c in CONFIGS:
            if c == BASELINE:
                continue
            v = cells.get(c)
            if v is None:
                cols.append("--")
            else:
                pct = 100.0 * (v - ref) / ref if ref else 0.0
                tally[c].append(pct)
                cols.append(f"{pct:+.2f}")
        out.append(
            f"| {name} | {n} | {lv} | {t:.2f} | {ref:,.0f} | " + " | ".join(cols) + " |"
        )
    out.append("")
    out.append("## Aggregate\n")
    out.append("| arm | cells | mean % | median % | better | worse | tied |")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for c, xs in tally.items():
        if not xs:
            continue
        better = sum(1 for x in xs if x < -0.005)
        worse = sum(1 for x in xs if x > 0.005)
        out.append(
            f"| {c} | {len(xs)} | {statistics.mean(xs):+.2f} | "
            f"{statistics.median(xs):+.2f} | {better} | {worse} | "
            f"{len(xs) - better - worse} |"
        )
    out.append(
        f"\n_Negative % = the sweep arm reaches a better (lower) score than the "
        f"incumbent at the same CPU time. Capacity = footprint//{CAP_DIVISOR}; scores are the "
        "SA fixed-point objective. Cells where an arm's measured time range does "
        "not cover the target are left blank rather than extrapolated._\n"
    )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        run_calibration()
    elif args.report:
        write_report()
    else:
        run_sweep(smoke=args.smoke)
        write_report()
