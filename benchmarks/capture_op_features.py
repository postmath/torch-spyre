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

"""Capture cost-model ``OpFeatures`` per (buffer, candidate division).

The co-optimizer's objective is memory-only, so a division only matters if it
changes what fits in LX. Measured consequence: four of the eleven existing
captures sit at the objective's floor and can never distinguish any move set,
schedule or capacity, and the reduction node term alone does not help (its
minimum is "never split" -- see ``coopt_band_retune_*.md`` and the node_term
commit). The missing piece is the matmul *reward*, which needs op-level metadata
the buffer-centric captures never carried.

This produces it. For every buffer handed to the solver it emits one
``OpFeatures`` per entry of that buffer's candidate division menu, so a menu index
selects its features directly -- the same indexing ``CoreDivisionBuffer.
core_divisions`` uses. Residency is left unresolved (``is_lx`` is applied at
scoring time by ``op_features.with_residency``), since that is the other half of
what the search decides.

Requires real hardware: the extractor reads live Inductor IR (committed layouts
and ``op_it_space_splits``), so it cannot run against the serialized captures.

Run from the repo root, on a Spyre machine::

    python3 benchmarks/capture_op_features.py
    python3 benchmarks/capture_op_features.py --out /tmp/features.json --only mlp

Note: the co-optimizing solver currently raises on a freshly compiled graph --
``parent 'argN_1' of 'bufM' is not in the solver's buffer set`` -- because
``_build_cd_bound_buffers`` sets ``parents=info["op_inputs"]`` without
intersecting the solver's buffer set, while ``SaCoOptimizingSolver._precompute_
topology`` asserts that intersection holds. Capture only needs the buffers, which
are built before that point, so the solve is allowed to fail and is recorded as
``solver_error``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# Must precede the torch import: the capture path is the SA co-optimizer's, and
# only ``layout_solver=simulated_annealing`` routes to CoOptimizingAllocator (the
# default ``greedy`` goes to StrategyBCoOptimizingAllocator, which has no
# ``_build_cd_bound_buffers``).
os.environ.setdefault("CO_OPTIMIZING_LX_PLANNING", "1")
os.environ.setdefault("LX_PLANNING", "1")
os.environ.setdefault("LAYOUT_SOLVER", "simulated_annealing")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch._inductor.config as _inductor_config  # noqa: E402

import torch_spyre  # noqa: F401,E402
from torch_spyre.constants import DEVICE_NAME  # noqa: E402
from torch_spyre._inductor.cost_model import _jsonable, op_to_dict  # noqa: E402
from torch_spyre._inductor.scratchpad import allocator as _allocator  # noqa: E402
from torch_spyre._inductor.scratchpad.op_features import (  # noqa: E402
    features_for_menu,
)

_inductor_config.force_disable_caches = True

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(
    _REPO, "tests", "inductor", "cooptimization_op_features.json"
)


# --- models -------------------------------------------------------------- #
# Named to line up with the existing capture corpus where the shapes match, so
# the two fixtures can be cross-referenced by graph name.
def _models():
    H, D = 192, 256

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(D, 4 * D, bias=False)
            self.down = torch.nn.Linear(4 * D, D, bias=False)

        def forward(self, x):
            return self.down(torch.nn.functional.gelu(self.up(x)))

    class SwiGLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(D, 2 * D, bias=False)
            self.up = torch.nn.Linear(D, 2 * D, bias=False)
            self.down = torch.nn.Linear(2 * D, D, bias=False)

        def forward(self, x):
            return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

    class Softmax(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    class RmsNorm(torch.nn.Module):
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    class SimpleAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(D, D, bias=False)
            self.k = torch.nn.Linear(D, D, bias=False)
            self.v = torch.nn.Linear(D, D, bias=False)

        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v

    return {
        "mlp": (MLP(), (H, D)),
        "swiglu": (SwiGLU(), (H, D)),
        "softmax": (Softmax(), (H, D)),
        "rms_norm": (RmsNorm(), (H, D)),
        "simple_attn": (SimpleAttn(), (H, D)),
    }


def _capture_one(name, model, shape):
    """Compile ``model`` and return its per-(buffer, division) features."""
    captured: dict = {}
    original = _allocator.CoOptimizingAllocator._build_cd_bound_buffers

    def hooked(self, graph, in_place, divisions):
        buffers = original(self, graph, in_place, divisions)
        for buf in buffers:
            try:
                op = graph.get_buffer(buf.name)
            except Exception:  # noqa: BLE001 - not every buffer is a graph buffer
                continue
            feats = features_for_menu(op, buf.core_divisions)
            captured[buf.name] = {
                "menu_size": len(buf.core_divisions),
                "output_partitions": [cd.output_partition for cd in buf.core_divisions],
                # index-aligned with core_divisions; null where the op could not
                # be featurized, so a menu index still selects the right entry.
                "features": [None if f is None else op_to_dict(f) for f in feats],
            }
        return buffers

    _allocator.CoOptimizingAllocator._build_cd_bound_buffers = hooked
    error = None
    try:
        m = model.to(torch.float16).to(DEVICE_NAME)
        x = torch.randn(*shape, dtype=torch.float16, device=DEVICE_NAME)
        torch.compile(m, backend="inductor")(x)
    except Exception as exc:  # noqa: BLE001 - the solve may fail; buffers still captured
        error = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
    finally:
        _allocator.CoOptimizingAllocator._build_cd_bound_buffers = original
    return captured, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", action="append", help="capture only these graphs")
    args = ap.parse_args()

    models = _models()
    if args.only:
        models = {k: v for k, v in models.items() if k in args.only}

    out: dict = {"graphs": {}}
    for name, (model, shape) in models.items():
        print(f"[{name}] compiling {tuple(shape)} ...", flush=True)
        try:
            buffers, error = _capture_one(name, model, shape)
        except Exception:  # noqa: BLE001 - one bad graph must not lose the rest
            traceback.print_exc()
            continue
        featurized = sum(
            1 for b in buffers.values() for f in b["features"] if f is not None
        )
        total = sum(b["menu_size"] for b in buffers.values())
        out["graphs"][name] = {
            "input_shape": list(shape),
            "buffers": buffers,
            "solver_error": error,
        }
        print(
            f"[{name}] buffers={len(buffers)} divisions={total} "
            f"featurized={featurized}/{total}"
            + (f"  (solve failed: {error[:60]})" if error else ""),
            flush=True,
        )

    # Sizes off the IR are frequently sympy ``Integer``; the cost model ships the
    # coercer its own serializer uses, so reuse it rather than a second one.
    # Written compact and key-sorted: the fixture is machine-generated and
    # machine-read, pretty-printing doubles it to 1.3 MB (over jj's default
    # new-file limit), and sorted keys keep regeneration diffs meaningful.
    with open(args.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=True, default=_jsonable)
    print(f"\nwrote {args.out}")
    return 0 if out["graphs"] else 1


if __name__ == "__main__":
    sys.exit(main())
