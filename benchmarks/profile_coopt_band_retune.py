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

"""Retune the reheating schedule's ``reorder`` band for the best-first sweep.

With the sweep as the reorder move, the ``crude`` schedule beats ``reheating``
outright -- the reverse of the Plan §5.1 claim (see
``test_sweep_reverses_the_schedule_ordering``). The cause is *geometry*, not
feedback: ``update()`` ignores its ``accepted`` argument entirely, so the band is
not a control loop. ``(accept_hi, accept_lo)`` only sets the temperature range in
units of the streamed move scale ``d_hat``::

    T_top = d_hat / -ln(accept_hi)      T_bottom = d_hat / -ln(accept_lo)

The shipped ``reorder`` band (0.6, 0.02) therefore cycles between 1.96*d_hat and
0.256*d_hat, while ``crude`` cools from ~1.0*d_hat to 0.001*d_hat. Reheating
never reaches a genuinely cold phase, and a strong reorder move is exactly what
profits from one.

Because the mapping is logarithmic, ``accept_lo`` alone cannot buy a cold end --
even 1e-6 only reaches 0.072*d_hat. So this sweeps the band *and* ``cycles``
(fewer cycles = longer uninterrupted descent, more steps spent near the bottom).

Band and cycle changes do not alter per-step work, so unlike the reorder-move A/B
this needs no wall-clock calibration: equal steps are equal time. Verified by
recording CPU seconds per arm anyway.

Run from the repo root::

    python3 benchmarks/profile_coopt_band_retune.py
    python3 benchmarks/profile_coopt_band_retune.py --smoke
    python3 benchmarks/profile_coopt_band_retune.py --report
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

CAP_DIVISOR = int(os.environ.get("COOPT_CAP_DIVISOR", "4"))
_SUF = "" if CAP_DIVISOR == 4 else f"_cap{CAP_DIVISOR}"
SWEEP_JSON = os.path.join(
    _RESULTS,
    f"coopt_band_retune_{os.environ.get('COOPT_RETUNE_MODE', 'scale')}{_SUF}.json",
)
REPORT_MD = os.path.join(
    _BENCH, f"coopt_band_retune_{os.environ.get('COOPT_RETUNE_MODE', 'scale')}{_SUF}.md"
)
LINES_PNG = os.path.join(
    _RESULTS,
    f"coopt_band_retune_{os.environ.get('COOPT_RETUNE_MODE', 'scale')}{_SUF}.png",
)

# The shipped band, and progressively colder ones. accept_hi is swept too: a
# lower top means the whole cycle sits colder, not just its bottom.
BANDS = {
    "b(.6,.02)": (0.6, 0.02),  # the shipped default
    "b(.6,1e-3)": (0.6, 1e-3),
    "b(.6,1e-6)": (0.6, 1e-6),
    "b(.3,1e-6)": (0.3, 1e-6),
}
CYCLES = [1, 2, 4]  # 4 is the shipped default

# Phase 2. The band sweep came back flat -- no reheating arm's CI excluded zero,
# while crude sat at -3.9%. The measured proposal mix explains why the band was
# the wrong knob: reheating spends only 7-12% of its proposals on `reorder`
# against crude's ~50%, because the cycle-phase mix weights each move by its
# neighborhood and `reorder`'s (n) is dwarfed by the flip/recolor menus. A
# stronger reorder move cannot pay off in a schedule that rarely proposes it, at
# any temperature. So sweep the reorder weight itself.
SCALES = [1, 2, 4, 8, 16, 32]

BASELINE = "reheat b(.6,.02) c4"  # the current shipping configuration
CRUDE = "crude"  # the schedule to beat
MODE = os.environ.get("COOPT_RETUNE_MODE", "scale")  # "band" | "scale" | "validate"


def _configs():
    """arm name -> solver kwargs."""
    out = {CRUDE: dict(schedule="crude")}
    if MODE == "band":
        for bname, band in BANDS.items():
            for c in CYCLES:
                out[f"reheat {bname} c{c}"] = dict(
                    schedule="reheating", move_bands={"reorder": band}, cycles=c
                )
        return out
    if MODE == "validate":
        # Out-of-sample check of the scale sweep's shortlist. The winner there was
        # picked as the best of 12 arms on 9 cells, so it is exactly the kind of
        # result that shrinks on fresh seeds; rerun only the shortlist and see.
        for scale, c in ((16, 4), (8, 2), (2, 4)):
            out[f"reheat x{scale} c{c}"] = dict(
                schedule="reheating", reorder_neighborhood_scale=scale, cycles=c
            )
        out[BASELINE] = dict(schedule="reheating")
        return out
    # scale mode: the shipped band, sweeping the reorder proposal weight x cycles
    for scale in SCALES:
        for c in (2, 4):
            name = BASELINE if (scale == 1 and c == 4) else f"reheat x{scale} c{c}"
            out[name] = dict(
                schedule="reheating", reorder_neighborhood_scale=scale, cycles=c
            )
    return out


CONFIGS = _configs()
SPB_GRID = [160, 640, 2560]
SEEDS = list(
    range(
        int(os.environ.get("COOPT_SEED_LO", "0")),
        int(os.environ.get("COOPT_SEED_HI", "10")),
    )
)
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


def _solve(name, spb, config, seed):
    bufs = _GRAPHS[name]
    cap = max(1, _foot(bufs) // CAP_DIVISOR)
    s = SaCoOptimizingSolver(
        cap, 128, seed=seed, steps_per_buffer=spb, max_steps=10**9, **CONFIGS[config]
    )
    c0 = time.process_time()
    s.plan_layout_and_core_divisions(copy.deepcopy(bufs))
    return {"best": s.best_score, "cpu": time.process_time() - c0}


def _work(task):
    return task, _solve(*task)


def run_sweep(smoke=False):
    graphs = _load_graphs()
    names = ["sdpa", "flash_attention"] if smoke else list(graphs)
    spbs = [160] if smoke else SPB_GRID
    seeds = [0, 1] if smoke else SEEDS
    tasks = [
        (n, spb, c, s) for n in names for spb in spbs for c in CONFIGS for s in seeds
    ]
    results: dict = {}
    os.makedirs(_RESULTS, exist_ok=True)
    start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(WORKERS, initializer=_init_worker)
    try:
        for (name, spb, config, seed), r in pool.imap_unordered(
            _work, tasks, chunksize=1
        ):
            cell = (
                results.setdefault(name, {"n": len(graphs[name]), "levels": {}})[
                    "levels"
                ]
                .setdefault(str(spb), {})
                .setdefault(config, {"best": [], "cpu": []})
            )
            cell["best"].append(r["best"])
            cell["cpu"].append(r["cpu"])
            done += 1
            if done % 200 == 0 or done == len(tasks):
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


def _bootstrap(a, b, iters=10000, seed=99):
    """95% CI for mean(b)-mean(a) as % of mean(a)."""
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


def _discriminating(data):
    """Graphs where at least one arm differs from the baseline at some budget."""
    out = []
    for name, r in data.items():
        moved = False
        for lv in r["levels"].values():
            if BASELINE not in lv:
                continue
            b = _mean(lv[BASELINE]["best"])
            if any(abs(_mean(c["best"]) - b) > 1e-9 for c in lv.values()):
                moved = True
        if moved:
            out.append(name)
    return out


def _plot(data, disc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spbs = sorted({int(s) for r in data.values() for s in r["levels"]})
    items = [(n, data[n]) for n in disc]
    ncol = min(3, max(1, len(items)))
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False
    )
    axes = axes.flatten()
    for ax, (name, r) in zip(axes, items):
        for config in CONFIGS:
            ys = []
            for spb in spbs:
                lv = r["levels"].get(str(spb), {})
                ys.append(_mean(lv[config]["best"]) if config in lv else float("nan"))
            style = dict(lw=1.1, marker="o", ms=3, alpha=0.75)
            if config == BASELINE:
                style = dict(lw=2.6, marker="s", ms=6, color="k")
            elif config == CRUDE:
                style = dict(lw=2.2, marker="^", ms=6, color="tab:red", ls="--")
            ax.plot(spbs, ys, label=config, **style)
        ax.set_xscale("log")
        ax.set_title(f"{name} (n={r['n']})", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=5, ncol=2)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(
        f"reorder band / cycles retune, capacity footprint//{CAP_DIVISOR} "
        "(black = shipped, red dashed = crude)"
    )
    fig.supxlabel("steps per buffer")
    fig.supylabel("best score (lower = better)")
    fig.tight_layout()
    fig.savefig(LINES_PNG, dpi=110)
    plt.close(fig)


def write_report():
    data = json.load(open(SWEEP_JSON))
    disc = _discriminating(data)
    _plot(data, disc)
    out = ["# Retuning the reheating schedule's `reorder` band for the sweep\n"]
    out.append(
        f"Capacity `footprint//{CAP_DIVISOR}`, {len(SEEDS)} seeds, "
        f"`reorder_move` at its default (`sweep_quality`). Band and cycle count do "
        f"not change per-step work, so equal steps are equal time here and no "
        f"wall-clock calibration is needed.\n\n"
        f"Baseline is the shipped `{BASELINE}`; `{CRUDE}` is the schedule that "
        f"currently beats it. Negative delta = better than the shipped default. "
        f"Only graphs where some arm moves are shown: {', '.join(disc)}.\n"
    )
    out.append(f"![lines](results/{os.path.basename(LINES_PNG)})\n")

    # Aggregate over discriminating graphs x budgets, per arm.
    out.append("## Aggregate over discriminating graphs\n")
    out.append("| arm | cells | mean % | median % | better | worse | 95% CI (pooled) |")
    out.append("|---|--:|--:|--:|--:|--:|---|")
    rows = {}
    for config in CONFIGS:
        deltas, pool_a, pool_b = [], [], []
        for name in disc:
            for spb, lv in data[name]["levels"].items():
                if BASELINE not in lv or config not in lv:
                    continue
                b = _mean(lv[BASELINE]["best"])
                if b == 0:
                    continue
                deltas.append(100.0 * (_mean(lv[config]["best"]) - b) / b)
                pool_a += [x / b for x in lv[BASELINE]["best"]]
                pool_b += [x / b for x in lv[config]["best"]]
        if not deltas:
            continue
        d, lo, hi = _bootstrap(pool_a, pool_b)
        rows[config] = statistics.mean(deltas)
        out.append(
            f"| {config} | {len(deltas)} | {statistics.mean(deltas):+.2f} | "
            f"{statistics.median(deltas):+.2f} | "
            f"{sum(1 for x in deltas if x < -1e-9)} | "
            f"{sum(1 for x in deltas if x > 1e-9)} | [{lo:+.2f}, {hi:+.2f}] |"
        )
    best = min(rows, key=rows.get) if rows else None
    out.append(f"\n**Best arm: `{best}` ({rows[best]:+.2f}% vs shipped).**\n")

    out.append("\n## Per graph and budget\n")
    out.append("| graph | n | spb | " + " | ".join(CONFIGS) + " |")
    out.append("|---|--:|--:|" + "--:|" * len(CONFIGS))
    for name in disc:
        r = data[name]
        for spb in sorted(r["levels"], key=int):
            lv = r["levels"][spb]
            cells = [
                f"{_mean(lv[c]['best']):,.0f}" if c in lv else "--" for c in CONFIGS
            ]
            out.append(f"| {name} | {r['n']} | {spb} | " + " | ".join(cells) + " |")
    out.append(
        "\n_Scores are the SA fixed-point objective, mean over seeds. Graphs whose "
        "score is identical under every arm at every budget are omitted._\n"
    )
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REPORT_MD)
    for line in out[3:14]:
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        run_sweep(smoke=args.smoke)
    write_report()
