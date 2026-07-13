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

"""Fake co-optimization substrate: replays captured dumps as lightweight,
dependency-free objects that mirror the real substrate's read-side API.

The joint SA engine is developed and tested against THIS fake so it never
depends on the (unlanded, moving) co-optimizing allocator during early phases.
By design we reconstruct plain dataclasses from JSON, NOT the real
``CoreDivision`` / ``CoreDivisionBuffer`` -- so substrate churn touches only the
real adapter, never the fake.

The fixture JSON (``cooptimization_captures.json``) is produced by running the
real ``CoOptimizingAllocator`` over softmax/mlp/swiglu/sdpa and serializing the
buffers handed to the solver (candidate menus + ``cd_parent_matches`` + placement
/ cost fields) plus the solver's chosen division + address per buffer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_CAPTURE_PATH = Path(__file__).parent / "cooptimization_captures.json"


# --- mirror of CoreDivision (with the derived API the SA engine calls) ------ #
@dataclass(frozen=True)
class FakeCoreDivision:
    output_splits: dict[int, int]
    reduction_splits: dict[int, int]

    @property
    def cores_used(self) -> int:
        return math.prod(self.output_splits.values()) * math.prod(
            self.reduction_splits.values()
        )

    @property
    def is_clean(self) -> bool:
        return not self.reduction_splits

    @property
    def output_partition(self) -> int:
        return math.prod(self.output_splits.values())

    def signature_key(self):
        return tuple(sorted(self.output_splits.items())) if self.is_clean else None

    @property
    def is_trivial(self) -> bool:
        """Whole-buffer / undivided (the seed's index-0 fixed division)."""
        return not self.output_splits and not self.reduction_splits


# --- mirror of CoreDivisionBuffer (input side) ------------------------------ #
@dataclass
class FakeCoreDivisionBuffer:
    name: str
    size: int
    uses: list[int]
    first_use_is_read: bool
    in_place_parents: list[str]
    placement: bool
    residency_reason: Optional[str]
    boundary_cost: int
    spill_write_cost: int
    parents: list[str]
    core_divisions: list[FakeCoreDivision]
    # {parent_name: [(parent_div_idx, this_div_idx), ...]} -- the slicing gate.
    cd_parent_matches: dict[str, list[tuple[int, int]]]
    # Solver-written outputs (Plan §7.2 / §7.3 output contract). Left unset on
    # load; the SA engine writes them during solve. ``address is None`` == spill.
    chosen_division: Optional[int] = None
    address: Optional[int] = None

    @property
    def residency_allowed(self) -> bool:
        return self.placement

    @property
    def start_time(self) -> int:
        return self.uses[0]

    @property
    def end_time(self) -> int:
        return self.uses[-1] + 1


@dataclass
class FakeGraph:
    """One captured graph: the solver INPUT (buffers) + the SOLVED reference."""

    buffers: list[FakeCoreDivisionBuffer]
    # name -> {"chosen_division": int|None, "address": int|None, "resident": bool}
    solved: dict[str, dict] = field(default_factory=dict)

    def by_name(self) -> dict[str, FakeCoreDivisionBuffer]:
        return {b.name: b for b in self.buffers}


# --- loading ---------------------------------------------------------------- #
def _cd(d: dict) -> FakeCoreDivision:
    # keys were serialized as strings; the real substrate keys them by int stride.
    return FakeCoreDivision(
        output_splits={int(k): v for k, v in d["output_splits"].items()},
        reduction_splits={int(k): v for k, v in d["reduction_splits"].items()},
    )


def _buf(d: dict) -> FakeCoreDivisionBuffer:
    return FakeCoreDivisionBuffer(
        name=d["name"],
        size=d["size"],
        uses=list(d["uses"]),
        first_use_is_read=d["first_use_is_read"],
        in_place_parents=list(d["in_place_parents"]),
        placement=d["placement"],
        residency_reason=d["residency_reason"],
        boundary_cost=d["boundary_cost"],
        spill_write_cost=d["spill_write_cost"],
        parents=list(d["parents"]),
        core_divisions=[_cd(cd) for cd in d["core_divisions"]],
        cd_parent_matches={
            p: [tuple(pair) for pair in pairs]
            for p, pairs in d["cd_parent_matches"].items()
        },
    )


def load_captures(
    path: str | Path = DEFAULT_CAPTURE_PATH,
) -> dict[str, list[FakeGraph]]:
    """Parse the capture JSON -> ``{case_name: [FakeGraph, ...]}``."""
    with open(path) as f:
        raw = json.load(f)
    out: dict[str, list[FakeGraph]] = {}
    for case, graphs in raw.items():
        out[case] = [
            FakeGraph(
                buffers=[_buf(b) for b in g["inputs"]],
                solved={s["name"]: s for s in g["solved"]},
            )
            for g in graphs
        ]
    return out


# The capture pins the committed/legacy division at ``core_divisions[0]`` (the
# allocator's fixed-division seed), so the SA seed is ``chosen_division = 0`` for
# every buffer (plan Section 8.2), with pi from a FirstFit pass over the sizes.
SEED_DIVISION_INDEX = 0


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CAPTURE_PATH
    cases = load_captures(path)
    print(f"{'case':8} {'graphs':>6} {'buffers':>7} {'pinned':>6} {'resident*':>9}")
    for case, graphs in cases.items():
        for gi, g in enumerate(graphs):
            pinned = sum(1 for b in g.buffers if not b.residency_allowed)
            resident = sum(1 for s in g.solved.values() if s["resident"])
            assert set(g.solved) <= set(g.by_name()), (
                f"{case}[{gi}] solved/name mismatch"
            )
            tag = f"{case}[{gi}]" if len(graphs) > 1 else case
            print(
                f"{tag:8} {len(graphs):>6} {len(g.buffers):>7} {pinned:>6} {resident:>9}"
            )
    print("(*resident = solved reference; the SA engine must re-derive it)")
