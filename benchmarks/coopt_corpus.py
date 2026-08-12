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

"""Shared corpus + objective loading for the co-optimizer benchmarks.

Every benchmark in this series solves the same captured graphs and needs the
same three things per graph: the solver's buffers, the per-division
``OpFeatures``, and the estimated fused bundles. This is the one place that
assembles them.

**Why this is not a convenience.** The engine's objective defaults to the string
``"bundle"``, which builds itself from the live Inductor graph. A benchmark
driving *serialized* captures has no live graph, so a solver constructed the
obvious way silently falls back to the memory-only objective and measures the
engine that used to exist. The failure is invisible: the run completes, the
report regenerates, and every number in it answers a question nobody asked. That
happened once already (see ``coopt_nested_ab.md``'s objective note), across a
benchmark whose loader had been copy-pasted. Duplicating that loader eight more
times is how it happens again, so it lives here and every benchmark imports it.

The corpus is ``cooptimization_captures_regen.json`` rather than the original
``cooptimization_captures.json``, which carries no features and therefore cannot
support the cost objective at all. The two were captured from different pipeline
revisions, so scores are not comparable across that boundary -- a report
regenerated against this corpus is a new measurement, not an update of an old
one.

Usage::

    from coopt_corpus import announce, foot, load_graphs, objective

    GRAPHS = load_graphs()                     # {name: {buffers, features, bundles}}
    announce(memory_only=False)                # print which objective is in use
    bufs = copy.deepcopy(GRAPHS[name]["buffers"])
    solver = SaCoOptimizingSolver(
        bufs, foot(bufs) // 2, 128,
        cost_objective=objective(GRAPHS[name], bufs),
    )
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

_BENCH = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH)
_FIXTURES = os.path.join(_REPO, "tests", "inductor")

CAPTURES = os.path.join(_FIXTURES, "cooptimization_captures_regen.json")
FEATURES = os.path.join(_FIXTURES, "cooptimization_op_features.json")

# Seed ranges already spent in this series: 0-4, 20-29, 30-49, 50-69. A rerun
# that reuses them is reporting in-sample numbers, so new waves start here.
FRESH_SEED_BASE = 70


def _default_spb() -> int:
    """The engine's own ``steps_per_buffer`` default, read off its signature.

    Every sweep in this series should include the point production runs at, and
    should learn it from the solver rather than restate it -- see the nested A/B,
    whose grid started at 4x the default and reversed below that range.
    """
    import inspect

    from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver

    return (
        inspect.signature(SaCoOptimizingSolver).parameters["steps_per_buffer"].default
    )


DEFAULT_SPB = _default_spb()


def _min_steps() -> int:
    """The engine's ``min_steps`` floor, read off its signature.

    Matters to any sweep that varies ``steps_per_buffer``: below the floor every
    arm runs the same number of steps regardless, so a grid derived per arm stops
    being a grid.
    """
    import inspect

    from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver

    return inspect.signature(SaCoOptimizingSolver).parameters["min_steps"].default


MIN_STEPS = _min_steps()


def load_graphs() -> dict[str, dict]:
    """``{graph_name: {"buffers", "features", "bundles"}}`` for the whole corpus.

    ``buffers`` are freshly parsed :class:`CoreDivisionBuffer` objects. The
    solver **mutates them in place**, so a caller running more than one solve per
    graph must ``deepcopy`` before each.
    """
    from tests.inductor.cooptimization_capture_loader import load_captures
    from torch_spyre._inductor.cost_model import op_from_dict

    with open(FEATURES) as fh:
        raw = json.load(fh)["graphs"]
    out: dict[str, dict] = {}
    for name, graphs in load_captures(CAPTURES).items():
        entry = raw[name]
        out[name] = {
            "buffers": graphs[0].buffers,
            "features": {
                n: [None if f is None else op_from_dict(f) for f in b["features"]]
                for n, b in entry["buffers"].items()
            },
            "bundles": entry["bundles"],
        }
    return out


def objective(entry: dict, buffers: Any) -> Any:
    """A fresh :class:`BundleCostObjective` over ``buffers``.

    One per solve, never shared: it memoizes per bundle and tracks dirty state
    against the last ``(chosen, resident)`` it scored, so reusing one across
    solves would carry another solve's state into the next.
    """
    from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective

    return BundleCostObjective(
        [b.name for b in buffers], entry["features"], entry["bundles"]
    )


def cost_objective_for(entry: dict, buffers: Any, memory_only: bool = False) -> Any:
    """``objective(...)``, or ``None`` when the caller asked for memory-only.

    Spelt out as a helper so a benchmark's ``--memory-only`` arm reads as an
    explicit opt-out at the call site rather than as an absent argument -- an
    absent argument is what silently produced the wrong objective before.
    """
    return None if memory_only else objective(entry, buffers)


def foot(buffers: Any) -> int:
    """Total per-core footprint at the seed (index-0) divisions: the scale every
    benchmark in this series derives its capacities from."""
    return sum(
        math.ceil(b.size / b.core_divisions[0].output_partition) for b in buffers
    )


def announce(memory_only: bool = False, extra: Optional[str] = None) -> None:
    """Print which objective and corpus the run is using.

    Printed unconditionally, at the top of every sweep, because the failure mode
    this module exists to prevent is a run that looks right and is not.
    """
    which = "memory-only (pre-cost-model)" if memory_only else "BundleCostObjective"
    print(f"objective: {which}", flush=True)
    print(f"corpus:    {os.path.basename(CAPTURES)}", flush=True)
    if extra:
        print(extra, flush=True)
