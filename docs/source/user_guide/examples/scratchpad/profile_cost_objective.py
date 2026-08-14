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

"""How affordable is the cost-model objective in an annealer's inner loop?

The engine's memory-only score costs ~3us for a whole 79-buffer graph, and an
anneal spends 10^5-10^6 evaluations. ``predict_ops`` costs 3-10us for a *single
bundle*, so scoring naively is 1-2 orders of magnitude too slow;
:class:`BundleCostObjective` answers with memoization plus dirty tracking. This
measures whether that is enough, because the answer decides the step budget --
and therefore whether a cost-model solve can be compared against the incumbent at
equal wall-clock at all.

Measured per move type, because they dirty very different amounts:

* ``flip`` changes one buffer's division -> the bundles containing it.
* ``recolor`` changes a region's divisions -> several bundles.
* ``reorder`` changes no division at all, only which buffers are LX-resident ->
  every bundle that reads or writes a buffer whose residency moved, which fans
  out further than the division map.

Also measures the cost of ``invalidate()``. The engine restores a snapshot on
every *rejected* move, which resets the diff baseline and forces the next score
to walk all bundles. Cached values survive, so the walk is mostly dict hits --
but if rejections are frequent the dirty tracking buys much less than it appears
to, and that shows up here rather than as a surprise later.

CPU-only: it runs off the captured feature fixture and never needs a Spyre card.

Run from the repo root::

    python3 docs/source/user_guide/examples/scratchpad/profile_cost_objective.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import json  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

from torch_spyre._inductor.cost_model import op_from_dict  # noqa: E402
from torch_spyre._inductor.scratchpad.cost_objective import (  # noqa: E402
    BundleCostObjective,
)

# Repo root: docs/source/user_guide/examples/scratchpad -> five levels up.
_REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5))
)
FIXTURE = os.path.join(_REPO, "tests", "inductor", "cooptimization_op_features.json")

# The memory-only objective this has to be affordable *relative to*, measured
# earlier on the solver corpus: whole-graph score, native packer.
MEMORY_ONLY_US = {25: 1.5, 43: 2.1, 79: 3.1}


def _load():
    with open(FIXTURE) as fh:
        graphs = json.load(fh)["graphs"]
    out = {}
    for name, g in graphs.items():
        names = list(g["buffers"])
        feats = {
            n: [None if f is None else op_from_dict(f) for f in b["features"]]
            for n, b in g["buffers"].items()
        }
        out[name] = (names, feats)
    return out


def _bundlings(names):
    """Groupings to test. The fixture carries no real grouping yet (that needs
    live IR), so bracket the plausible range instead of pretending to one."""
    mid = max(1, len(names) // 3)
    return {
        "one-per-op": [[n] for n in names],
        "runs-of-3": [names[i : i + 3] for i in range(0, len(names), 3)],
        "whole-graph": [list(names)],
        "runs-of-n/3": [names[i : i + mid] for i in range(0, len(names), mid)],
    }


def _time_moves(obj, names, feats, move, reps=400, invalidate_rate=0.0, seed=0):
    """Mean microseconds per ``score`` call under a given move type."""
    rng = random.Random(seed)
    menu = {n: max(1, len(feats[n])) for n in names}
    chosen = [0] * len(names)
    resident = set(names[: len(names) // 2])
    obj.score(chosen, frozenset(resident))  # warm the first full pass
    samples = []
    for _ in range(reps):
        if move == "flip":
            i = rng.randrange(len(names))
            chosen[i] = rng.randrange(menu[names[i]])
        elif move == "recolor":
            for _ in range(max(2, len(names) // 3)):
                i = rng.randrange(len(names))
                chosen[i] = rng.randrange(menu[names[i]])
        elif move == "reorder":
            for _ in range(2):  # a rotation typically moves a couple of buffers
                n = names[rng.randrange(len(names))]
                resident.symmetric_difference_update({n})
        if invalidate_rate and rng.random() < invalidate_rate:
            obj.invalidate()
        frozen = frozenset(resident)
        t0 = time.perf_counter()
        obj.score(chosen, frozen)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.mean(samples), statistics.median(samples)


def _synthesize(base_names, base_feats, n_target, per_bundle):
    """Replicate real features up to ``n_target`` buffers.

    Argument names are renamed per copy so distinct buffers do not alias each
    other in the residency dirty map -- otherwise every copy would dirty every
    bundle and the measurement would be meaningless.
    """
    import dataclasses

    names: list[str] = []
    feats: dict = {}
    copy_index = 0
    while len(names) < n_target:
        for base in base_names:
            if len(names) >= n_target:
                break
            new = f"{base}_r{copy_index}"
            names.append(new)
            feats[new] = [
                None
                if f is None
                else dataclasses.replace(
                    f,
                    args=[
                        dataclasses.replace(a, name=f"{a.name}_r{copy_index}")
                        for a in f.args
                    ],
                )
                for f in base_feats[base]
            ]
        copy_index += 1
    bundles = [names[i : i + per_bundle] for i in range(0, len(names), per_bundle)]
    return names, feats, bundles


def main():
    graphs = _load()
    print("Per-call cost of BundleCostObjective.score (microseconds).")
    print("The memory-only objective it replaces: ~1.5-3.1us for a WHOLE graph.\n")

    print(
        f"{'graph':13s} {'n':>3s} {'bundling':>12s} {'bundles':>8s} "
        f"{'flip':>8s} {'recolor':>8s} {'reorder':>8s} {'hit%':>6s}"
    )
    for name, (names, feats) in sorted(graphs.items(), key=lambda kv: len(kv[1][0])):
        for label, bundles in _bundlings(names).items():
            obj = BundleCostObjective(names, feats, bundles)
            row = []
            for move in ("flip", "recolor", "reorder"):
                mean, _ = _time_moves(obj, names, feats, move)
                row.append(mean)
            hit = 100.0 * (1 - obj.evaluations / max(1, obj.lookups))
            nb = len(obj._bundles)
            print(
                f"{name:13s} {len(names):3d} {label:>12s} {nb:8d} "
                f"{row[0]:8.1f} {row[1]:8.1f} {row[2]:8.1f} {hit:6.0f}"
            )

    print("\n--- cost of a rejected move (invalidate forces a full walk) ---")
    print(f"{'graph':13s} {'reject rate':>12s} {'flip us':>9s} {'vs 0%':>8s}")
    for name, (names, feats) in sorted(graphs.items(), key=lambda kv: len(kv[1][0])):
        bundles = _bundlings(names)["runs-of-3"]
        base = None
        for rate in (0.0, 0.25, 0.75):
            obj = BundleCostObjective(names, feats, bundles)
            mean, _ = _time_moves(obj, names, feats, "flip", invalidate_rate=rate)
            if base is None:
                base = mean
            print(f"{name:13s} {rate:12.0%} {mean:9.1f} {mean / base:7.2f}x")

    print("\n--- scaling, measured on synthetic graphs ---")
    print("Real features replicated up to n buffers (argument names kept distinct)")
    print("so the dirty maps stay realistic. Multiplying a whole-graph bundle cost")
    print("by a bundle count would double-count; this measures instead.\n")
    names0, feats0 = graphs["simple_attn"]
    print(
        f"{'n':>4s} {'bundle sz':>10s} {'bundles':>8s} {'flip us':>9s} "
        f"{'reorder us':>11s} {'hit%':>6s} {'vs mem-only':>12s}"
    )
    for n in sorted(MEMORY_ONLY_US):
        for per in (1, 3, 6):
            names, feats, bundles = _synthesize(names0, feats0, n, per)
            obj = BundleCostObjective(names, feats, bundles)
            flip, _ = _time_moves(obj, names, feats, "flip", reps=250)
            reorder, _ = _time_moves(obj, names, feats, "reorder", reps=250)
            hit = 100.0 * (1 - obj.evaluations / max(1, obj.lookups))
            worst = max(flip, reorder)
            print(
                f"{n:4d} {per:10d} {len(obj._bundles):8d} {flip:9.1f} "
                f"{reorder:11.1f} {hit:6.0f} {worst / MEMORY_ONLY_US[n]:11.1f}x"
            )


if __name__ == "__main__":
    main()
