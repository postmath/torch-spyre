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

"""Micro-driver that isolates ``NativePermutationLayoutSolver::RecomputeAll``.

Each ``rotate`` on an eligible buffer runs exactly one ``RecomputeAll``, so a
tight loop of random rotations hammers the recompute path with *no* Python SA
loop and no per-op glue -- the cleanest target for a callgrind line-by-line
profile:

    valgrind --tool=callgrind --collect-atstart=no \\
        --toggle-collect='*RecomputeAll*' --dump-instr=yes --cache-sim=yes \\
        --callgrind-out-file=cg.out \\
        python benchmarks/microbench_recompute.py

``--toggle-collect`` restricts *counting* to the RecomputeAll subtree, so torch
import and packer construction are instrumented but not counted. Read with
``callgrind_annotate --auto=yes cg.out`` (or kcachegrind).

Buffers are a minimal duck-typed object (the C++ ctor only reads name / size /
uses / first_use_is_read / in_place_parents), keeping the un-collected setup
light and free of scratchpad imports.
"""

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
import random as rnd  # noqa: E402

# Prefer the standalone, torch-free packer module (benchmarks/standalone_packer,
# put its build dir on PYTHONPATH) so profilers see no torch import; fall back to
# the shipping _C extension.
try:
    from packer_native import NativePermutationLayoutSolver  # noqa: E402
except ImportError:
    from torch_spyre._C import NativePermutationLayoutSolver  # noqa: E402

ALIGNMENT = 128


class Buf:
    """Duck-typed buffer exposing exactly the attributes the C++ ctor reads."""

    __slots__ = ("name", "size", "uses", "first_use_is_read", "in_place_parents")

    def __init__(self, name, size, uses):
        self.name = name
        self.size = size
        self.uses = uses
        self.first_use_is_read = True
        self.in_place_parents = []


def make_buffers(rng, n, horizon=12, max_size=200, inplace_prob=0.25):
    """Same shape as the SA/packer differential generator: heavy overlap
    (small horizon) with ``inplace_prob`` in-place children. Lowering
    ``inplace_prob`` raises fast-path coverage (buffers without an in-place
    partner skip the candidate scan), so sweeping it shows the optimization's
    win as a function of in-place density."""
    bufs = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        uses = [start] if end == start + 1 else [start, end - 1]
        bufs.append(Buf(f"b{i}", size, uses))
    for child_i in range(1, n):
        if rng.random() < inplace_prob:
            parent = bufs[rng.randrange(child_i)]
            child = bufs[child_i]
            new_start = parent.uses[-1]
            new_last = max(child.uses[-1], parent.uses[-1])
            child.uses = [new_start] if new_start == new_last else [new_start, new_last]
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return bufs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128, help="Buffer count.")
    ap.add_argument("--iters", type=int, default=1500, help="rotate() calls.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--inplace-prob",
        type=float,
        default=0.25,
        help="Fraction of buffers wired as in-place children; lower => higher "
        "fast-path coverage. Sweep to measure the optimization vs in-place density.",
    )
    args = ap.parse_args()

    rng = rnd.Random(args.seed)
    bufs = make_buffers(rng, args.n, inplace_prob=args.inplace_prob)
    capacity = max(b.size for b in bufs) * 3
    solver = NativePermutationLayoutSolver(
        bufs, list(range(args.n)), capacity, ALIGNMENT
    )

    # Precompute the moves so the timed loop is pure C++ (no RNG in the hot
    # path polluting the profile).
    moves = [(rng.randrange(args.n), rng.randrange(args.n)) for _ in range(args.iters)]

    # Tight rotate loop: one RecomputeAll per call (all buffers eligible).
    for i, j in moves:
        solver.rotate(i, j)

    # Touch a result so nothing is optimized away.
    print(f"n={args.n} iters={args.iters} final_quality={solver.quality():.1f}")


if __name__ == "__main__":
    main()
