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

"""Capture a co-optimizer corpus: solver buffers, bundles, and ``OpFeatures``.

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
core_divisions`` uses. Residency is left unresolved (``ArgTraffic.mem`` is applied
at scoring time by ``op_features.with_residency``), since that is the other half
of what the search decides.

**Three artifacts from one compile, deliberately.** ``BundleCostObjective`` needs
all three -- the solver's buffers, the estimated fused bundles, and the features
-- and they only line up if they come from the same graph. The first version of
this script emitted features alone, against models whose shapes had drifted from
the ones behind ``cooptimization_captures.json``: the buffer *names* still
collided (Inductor numbers them ``buf0..``, so any graph produces the same
names), but ``softmax`` was captured at 1024x512 and featurized at 192x256, and
two graphs did not even agree on buffer count. Scoring the captured corpus with
those features would have been silently, undetectably wrong. So the hook now
serializes the buffers it featurizes, in the schema
``cooptimization_capture_loader`` already reads, and the bundle grouping
``fusion.estimate_bundles`` derives from the same operation list.

Requires real hardware: the extractor reads live Inductor IR (committed layouts
and ``op_it_space_splits``), so it cannot run against the serialized captures.

Run from the repo root, on a Spyre machine::

    python3 docs/source/user_guide/examples/scratchpad/capture_op_features.py
    python3 docs/source/user_guide/examples/scratchpad/capture_op_features.py --out /tmp/features.json --only mlp

Note: the solve used to raise on every freshly compiled graph --
``parent 'argN_1' of 'bufM' is not in the solver's buffer set``, because
``_build_cd_bound_buffers`` sets ``parents=info["op_inputs"]`` without
intersecting the solver's buffer set while ``_precompute_topology`` asserted that
intersection held. The solver now skips such parents instead, so the solve
succeeds; ``solver_error`` stays in the output because capture only needs the
buffers (built before the solve) and must not lose a graph to an unrelated
failure.
"""

from __future__ import annotations

import argparse
import json
import math
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
from torch_spyre._inductor.fusion import estimate_bundles  # noqa: E402
from torch_spyre._inductor.scratchpad import allocator as _allocator  # noqa: E402
from torch_spyre._inductor.scratchpad.op_features import (  # noqa: E402
    features_for_menu,
)
from torch_spyre._inductor.scratchpad.plan_solver import BufferType  # noqa: E402

_inductor_config.force_disable_caches = True

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(
    _REPO, "tests", "inductor", "cooptimization_op_features.json"
)
DEFAULT_CAPTURES_OUT = os.path.join(
    _REPO, "tests", "inductor", "cooptimization_captures_regen.json"
)


# --- models -------------------------------------------------------------- #
# One entry per graph in the co-optimizer corpus, under the corpus's own name.
# Shapes are reverse-engineered from the buffer sizes in
# ``cooptimization_captures.json`` -- that fixture's generator was never
# committed, so the sizes are the only surviving record of what was compiled.
# ``--verify`` reports, per graph, how close the result lands to the original.
def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float16)


def _module(model):
    """A module as a plain callable, moved to the device on first call."""
    holder: list = []

    def call(*args):
        if not holder:
            holder.append(model.to(torch.float16).to(DEVICE_NAME))
        return holder[0](*args)

    return call


def _softmax():
    # Capture: buf1 = 1048576 B = 524288 elems, buf0 = 2048 B = 1024 rows.
    return (lambda x: torch.softmax(x, dim=-1), lambda: (_rand(1024, 512),))


def _mlp():
    # Capture: weights 262144 elems (D x 4D), activations H x 4D = 131072 elems.
    # ``Linear`` rather than a raw matmul because the corpus carries a buffer per
    # weight (a materialized transpose), which ``x @ w`` does not produce.
    D, H = 256, 128

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(D, 4 * D, bias=False)
            self.down = torch.nn.Linear(4 * D, D, bias=False)

        def forward(self, x):
            return self.down(torch.nn.functional.gelu(self.up(x)))

    return _module(MLP()), lambda: (_rand(H, D),)


def _swiglu():
    # Capture: 8 buffers, two D x D weights, no down projection -- the captured
    # graph stops at the gate/up product.
    D, H = 512, 256

    class SwiGLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(D, D, bias=False)
            self.up = torch.nn.Linear(D, D, bias=False)

        def forward(self, x):
            return torch.nn.functional.silu(self.gate(x)) * self.up(x)

    return _module(SwiGLU()), lambda: (_rand(H, D),)


def _rms_norm():
    # tests/inductor/test_building_blocks.py::test_rms_norm (T=128, D=256),
    # reducing along dim 0 as that test does.
    T, D = 128, 256

    def fn(x, weight, eps):
        x_mean_sq = (x * x).mean(dim=0)
        return x * torch.rsqrt(x_mean_sq + eps)[None, :] * weight

    return fn, lambda: (
        _rand(D, T),
        _rand(D, T),
        torch.full([T], 1e-6, dtype=torch.float16),
    )


def _simple_attn():
    # tests/inductor/test_building_blocks.py::test__simple_attn.
    H, Q, L, D = 4, 64, 256, 128

    def fn(q, k, v, sm_scale):
        qk = (q @ k.transpose(-1, -2).contiguous()) * sm_scale
        return qk.softmax(dim=-1) @ v

    return fn, lambda: (
        _rand(H * Q, D),
        _rand(L, D),
        _rand(L, D),
        torch.full([L], D**-0.5, dtype=torch.float16),
    )


def _sdpa():
    # tests/inductor/test_building_blocks.py::test_causal_sdpa_unpadded_kv_no_inf,
    # written out rather than called: ``F.scaled_dot_product_attention`` lowers to
    # a multi-output extern (``NotImplementedError: MultiOutputLayout``) that never
    # reaches the allocator, so the hook sees no buffers at all. The corpus's
    # 25-buffer sdpa capture is a decomposition, which is what this reproduces.
    B, H, S, D = 1, 8, 64, 128

    def fn(q, k, v, mask):
        scores = (q @ k.transpose(-1, -2).contiguous()) + mask
        return scores.softmax(dim=-1) @ v

    shape = (B, H, S, D)
    return fn, lambda: (
        _rand(*shape),
        _rand(*shape),
        _rand(*shape),
        torch.triu(torch.full((S, S), -1e4, dtype=torch.float16), diagonal=1),
    )


def _flash(seq_len):
    # tests/inductor/test_building_blocks.py::test_flash_attention. ``flash_big``
    # is the same decomposition over twice the sequence, i.e. twice the blocks.
    B, H, D = 1, 8, 64
    block_size = 128

    def fn(Q, K, V):
        output = torch.zeros_like(Q)
        M = torch.full(
            (B, H, seq_len), float("-inf"), device=Q.device, dtype=torch.float16
        )
        denominator = torch.zeros((B, H, seq_len), device=Q.device, dtype=torch.float16)
        scale = 1.0 / math.sqrt(D)
        for start in range(0, seq_len, block_size):
            K_block = K[:, :, start : start + block_size, :]
            V_block = V[:, :, start : start + block_size, :]
            scores = torch.matmul(Q, K_block.transpose(-1, -2).contiguous()) * scale
            scores = scores.transpose(-1, -2).contiguous()
            max_running = torch.maximum(M, torch.amax(scores, dim=-2))
            exp_scores = torch.exp(scores - max_running.unsqueeze(-2))
            correction = torch.exp(M - max_running)
            denominator = denominator * correction + exp_scores.sum(dim=-2)
            output = output * correction.unsqueeze(-1) + torch.bmm(
                exp_scores.transpose(-1, -2).flatten(0, 1), V_block.flatten(0, 1)
            ).unflatten(0, (B, H))
            M = max_running
        return output / denominator.unsqueeze(-1)

    shape = (B, H, seq_len, D)
    return fn, lambda: (_rand(*shape), _rand(*shape), _rand(*shape))


def _blocks(n_blocks):
    # Capture: 14 buffers per repeat -- attention (256x128, as ``simple_attn``)
    # followed by a gate/up/silu/mul/down MLP with a 4x hidden dimension.
    rows, D, hidden = 256, 128, 512

    def fn(x, *weights):
        for i in range(n_blocks):
            k, v, gate, up, down = weights[5 * i : 5 * i + 5]
            p = (x @ k.transpose(-1, -2).contiguous()).softmax(dim=-1)
            attn = p @ v
            x = (torch.nn.functional.silu(attn @ gate) * (attn @ up)) @ down
        return x

    def args():
        out = [_rand(rows, D)]
        for _ in range(n_blocks):
            out += [
                _rand(rows, D),
                _rand(rows, D),
                _rand(D, hidden),
                _rand(D, hidden),
                _rand(hidden, D),
            ]
        return tuple(out)

    return fn, args


def _models():
    """``{graph_name: (callable, args_factory)}`` for the whole corpus."""
    return {
        "softmax": _softmax(),
        "mlp": _mlp(),
        "swiglu": _swiglu(),
        "rms_norm": _rms_norm(),
        "simple_attn": _simple_attn(),
        "sdpa": _sdpa(),
        "flash_attention": _flash(256),
        "block_x2": _blocks(2),
        "block_x3": _blocks(3),
        "block_x4": _blocks(4),
        "flash_big": _flash(512),
    }


def _serialize_buffer(buf):
    """One solver buffer in ``cooptimization_captures.json``'s schema.

    ``placement`` / ``boundary_cost`` / ``spill_write_cost`` predate the landed
    data model and are reconstructed here rather than dropped, so the existing
    ``cooptimization_capture_loader`` reads a fresh capture unchanged (it maps
    all three back on load; see its module docstring for why the mapping is
    exact).
    """
    return {
        "name": buf.name,
        "size": buf.size,
        "uses": list(buf.uses),
        "first_use_is_read": buf.first_use_is_read,
        "in_place_parents": list(buf.in_place_parents),
        "placement": buf.residency_reason is None,
        "residency_reason": buf.residency_reason,
        "boundary_cost": buf.size if buf.boundary is BufferType.Output else 0,
        "spill_write_cost": buf.size,
        "parents": list(buf.parents),
        # Split keys are strides off the IR and are frequently sympy integers,
        # which json refuses as keys; the loader coerces them back with int().
        "core_divisions": [
            {
                "output_splits": {
                    str(int(k)): int(v) for k, v in cd.output_splits.items()
                },
                "reduction_splits": {
                    str(int(k)): int(v) for k, v in cd.reduction_splits.items()
                },
            }
            for cd in buf.core_divisions
        ],
        "cd_parent_matches": {
            parent: [list(pair) for pair in pairs]
            for parent, pairs in buf.cd_parent_matches.items()
        },
    }


def _drop_foreign_parents(buffer_list):
    """Drop ``parents`` entries naming buffers the solver does not own.

    ``_build_cd_bound_buffers`` sets ``parents = info["op_inputs"]`` without
    intersecting the solver's buffer set, so an op's graph inputs appear there
    even when they are not solver buffers.

    The solver no longer needs this -- ``_precompute_topology`` skips unowned
    parents rather than asserting on them -- so this is now about the *fixture*
    rather than about solvability: the historical corpus records only in-set
    edges, and filtering keeps a fresh capture directly comparable to it. Returns
    the number of edges dropped, so the caller can report it rather than hide it.
    """
    owned = {b["name"] for b in buffer_list}
    dropped = 0
    for buf in buffer_list:
        keep = [p for p in buf["parents"] if p in owned]
        dropped += len(buf["parents"]) - len(keep)
        buf["parents"] = keep
        buf["cd_parent_matches"] = {
            p: pairs for p, pairs in buf["cd_parent_matches"].items() if p in owned
        }
    return dropped


def _capture_one(name, fn, make_args):
    """Compile ``fn`` and return its buffers, bundles and per-division features."""
    captured: dict = {}
    buffers_out: list = []
    bundles_out: list = []
    original = _allocator.CoOptimizingAllocator._build_cd_bound_buffers

    def hooked(self, graph, in_place, divisions):
        buffers = original(self, graph, in_place, divisions)
        buffers_out[:] = [_serialize_buffer(b) for b in buffers]
        # The grouping the cost model has to score against. Estimated from the
        # same operation list the allocator holds, because fusion has not run
        # yet at this point in the pipeline (see fusion.estimate_bundles).
        bundles_out[:] = [
            [op.get_name() for op in group]
            for group in estimate_bundles(graph.operations)
        ]
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
        args = tuple(a.to(DEVICE_NAME) for a in make_args())
        torch.compile(fn, backend="inductor")(*args)
    except Exception as exc:  # noqa: BLE001 - the solve may fail; buffers still captured
        error = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
    finally:
        _allocator.CoOptimizingAllocator._build_cd_bound_buffers = original
    return captured, buffers_out, bundles_out, error


def _verify(name, buffers):
    """Compare a freshly captured buffer set against the historical corpus.

    Reports rather than asserts: the original generator was never committed, so
    a mismatch means the reconstructed model differs from whatever produced the
    old fixture -- useful to know, not a reason to lose the capture.
    """
    from tests.inductor.cooptimization_capture_loader import (
        DEFAULT_CAPTURE_PATH,
        LARGE_CAPTURE_PATH,
        load_captures,
    )

    for path in (DEFAULT_CAPTURE_PATH, LARGE_CAPTURE_PATH):
        graphs = load_captures(path)
        if name in graphs:
            old = {b.name: b for b in graphs[name][0].buffers}
            break
    else:
        return "no historical capture"
    new = {b["name"]: b for b in buffers}
    if set(old) != set(new):
        return f"buffer set differs ({len(old)} old vs {len(new)} new)"
    size_diff = [n for n in old if old[n].size != new[n]["size"]]
    menu_diff = [
        n for n in old if len(old[n].core_divisions) != len(new[n]["core_divisions"])
    ]
    if not size_diff and not menu_diff:
        return "exact"
    return f"{len(size_diff)} sizes, {len(menu_diff)} menus differ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--captures-out",
        default=DEFAULT_CAPTURES_OUT,
        help="where to write the regenerated solver-buffer corpus",
    )
    ap.add_argument("--only", action="append", help="capture only these graphs")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="compare each capture against the historical corpus",
    )
    args = ap.parse_args()

    models = _models()
    if args.only:
        models = {k: v for k, v in models.items() if k in args.only}

    out: dict = {"graphs": {}}
    captures: dict = {}
    for name, (fn, make_args) in models.items():
        print(f"[{name}] compiling ...", flush=True)
        try:
            buffers, buffer_list, bundles, error = _capture_one(name, fn, make_args)
        except Exception:  # noqa: BLE001 - one bad graph must not lose the rest
            traceback.print_exc()
            continue
        featurized = sum(
            1 for b in buffers.values() for f in b["features"] if f is not None
        )
        total = sum(b["menu_size"] for b in buffers.values())
        out["graphs"][name] = {
            "buffers": buffers,
            "bundles": bundles,
            "solver_error": error,
        }
        # The solve itself is what the co-optimizer will redo, so no reference
        # solution is recorded; the loader treats ``solved`` as optional.
        captures[name] = [{"inputs": buffer_list, "solved": []}]
        foreign = _drop_foreign_parents(buffer_list)
        verdict = f"  vs corpus: {_verify(name, buffer_list)}" if args.verify else ""
        print(
            f"[{name}] buffers={len(buffer_list)} featurized_buffers={len(buffers)} "
            f"divisions={total} featurized={featurized}/{total} "
            f"bundles={len(bundles)} foreign_parents_dropped={foreign}"
            + (f"  (solve failed: {error[:60]})" if error else "")
            + verdict,
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
    with open(args.captures_out, "w") as fh:
        json.dump(
            captures, fh, separators=(",", ":"), sort_keys=True, default=_jsonable
        )
    print(f"wrote {args.captures_out}")
    return 0 if out["graphs"] else 1


if __name__ == "__main__":
    sys.exit(main())
