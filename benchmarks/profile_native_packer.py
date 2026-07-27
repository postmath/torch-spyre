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

"""Profiling / A-B driver for the native (C++) permutation-layout packer.

Drives the *real* consumer -- ``SimulatedAnnealingLayoutSolver`` -- over a
deterministic set of randomly generated buffer problems spanning the captured
graph range (~30-320 buffers), with either the native C++ packer or the
pure-Python one selected via ``torch_spyre._inductor.config.native_layout_packer``.

Two uses:

  * **Wall-clock A/B** (``--packer both``): headline "how much faster is
    native" number, per size and overall.

  * **Flamegraph target** (``--packer native``): a long-running, deterministic
    workload to sample with py-spy, e.g.

        py-spy record --native --rate 250 -o native_packer.svg -- \\
            python benchmarks/profile_native_packer.py --packer native

    py-spy is the only sampler that sees into ``_C.so`` on this box (no perf,
    no root); ``--native`` unwinds the C++ stack via the always-on ``-g`` info.
    Launching the driver *under* py-spy keeps it same-uid, so no ptrace-scope
    permission is needed.

BLAS thread pinning happens before torch is imported (a shared machine would
otherwise oversubscribe); layout planning itself is single-threaded.
"""

import os

# Pin BLAS/OMP threads *before* importing torch -- see module docstring.
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
import random as rnd  # noqa: E402
import time  # noqa: E402

from torch_spyre._inductor import config  # noqa: E402
from torch_spyre._inductor.scratchpad.plan_solver import (  # noqa: E402
    LifetimeBoundBuffer,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (  # noqa: E402
    SimulatedAnnealingLayoutSolver,
)


def _random_buffers(rng, n, horizon=12, max_size=200):
    """Half-open lifetimes, some in-place children (parent.end == child.start+1).

    Mirrors the generator used by the SA / packer differential tests so the
    profiled workload matches what those suites exercise.
    """
    buffers = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        uses = [start] if end == start + 1 else [start, end - 1]
        buffers.append(LifetimeBoundBuffer(f"b{i}", size, uses))
    for child_i in range(1, n):
        if rng.random() < 0.25:
            parent = buffers[rng.randrange(child_i)]
            child = buffers[child_i]
            new_start = parent.uses[-1]
            new_last = max(child.uses[-1], parent.uses[-1])
            child.uses = [new_start] if new_start == new_last else [new_start, new_last]
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def _make_instances(sizes, seeds):
    """Deterministic (buffers, capacity, seed) instances across sizes x seeds.

    Capacity is a tight multiple of the largest buffer so the initial layout
    does *not* trivially fit -- forcing the SA search to actually iterate (the
    regime the native packer is meant to accelerate).
    """
    instances = []
    for n in sizes:
        for seed in seeds:
            rng = rnd.Random(seed * 1000 + n)
            buffers = _random_buffers(rng, n)
            capacity = max(b.size for b in buffers) * 3
            instances.append((n, seed, capacity, buffers))
    return instances


def _run_once(buffers, capacity, seed):
    """One full SA solve on a private copy of the buffers (default schedule)."""
    work = copy.deepcopy(buffers)
    solver = SimulatedAnnealingLayoutSolver(capacity, 128, random=rnd.Random(seed))
    solver.plan_layout(work)
    return work


def _time_config(instances, native, repeats):
    """Total wall-clock to solve every instance ``repeats`` times under the
    selected packer. Returns (total_seconds, per_size_seconds)."""
    per_size: dict[int, float] = {}
    total = 0.0
    with config.patch(native_layout_packer=native):
        for _ in range(repeats):
            for n, seed, capacity, buffers in instances:
                t0 = time.perf_counter()
                _run_once(buffers, capacity, seed)
                dt = time.perf_counter() - t0
                per_size[n] = per_size.get(n, 0.0) + dt
                total += dt
    return total, per_size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--packer",
        choices=("native", "python", "both"),
        default="both",
        help="Which packer to exercise (both = A/B comparison).",
    )
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Buffer counts to sweep.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="Distinct random instances per size.",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the whole instance set N times (raise to lengthen a "
        "py-spy sampling run).",
    )
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    instances = _make_instances(args.sizes, seeds)
    ninst = len(instances)
    print(
        f"instances: {ninst} ({len(args.sizes)} sizes x {args.seeds} seeds), "
        f"repeats={args.repeats}, sizes={args.sizes}"
    )

    configs = (
        [("native", True), ("python", False)]
        if args.packer == "both"
        else [(args.packer, args.packer == "native")]
    )

    results = {}
    for label, native in configs:
        total, per_size = _time_config(instances, native, args.repeats)
        results[label] = (total, per_size)
        solves = ninst * args.repeats
        print(
            f"\n[{label}] total={total:.3f}s over {solves} solves "
            f"({1000 * total / solves:.2f} ms/solve)"
        )
        for n in args.sizes:
            print(f"    n={n:4d}: {per_size[n]:.3f}s")

    if args.packer == "both":
        nt = results["native"][0]
        pt = results["python"][0]
        print(
            f"\nspeedup (python/native): {pt / nt:.2f}x  (native {nt:.3f}s vs python {pt:.3f}s)"
        )
        print("per-size speedup:")
        for n in args.sizes:
            ns = results["native"][1][n]
            ps = results["python"][1][n]
            print(f"    n={n:4d}: {ps / ns:.2f}x")


if __name__ == "__main__":
    main()
