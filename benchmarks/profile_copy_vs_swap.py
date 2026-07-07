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

"""Compare the cost of PermutationBasedLayoutSolver.copy() against swap() and a
full single-element sweep, across buffer counts and overlap densities.

This decides whether the Imanishi/Xu reinsertion sweep should add a second copy
(to avoid re-traversing) -- worth it only if a copy is cheaper than the swaps it
saves. Reports, per (n, density): the adjacent-pair overlap rate, build time,
copy time, no-op/overlapping swap time, full-sweep time, and "swaps/copy" -- how
many average swaps equal one copy. A second copy saves ~n/2 swaps, so it pays
off only when swaps/copy < n/2.

Run from the repository root::

    python benchmarks/profile_copy_vs_swap.py
"""

import random
import time

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
)


def make_buffers(rng, n, span_frac):
    """Half-open lifetimes over a horizon of n; lifetime length ~ uniform up to
    span_frac * n, which controls how many buffers are simultaneously live."""
    max_len = max(1, int(span_frac * n))
    buffers = []
    for i in range(n):
        start = rng.randint(0, n)
        length = rng.randint(1, max_len)
        # Half-open lifetime [start, start + length); uses records first and last.
        uses = [start] if length == 1 else [start, start + length - 1]
        buffers.append(LifetimeBoundBuffer(f"b{i}", rng.randint(1, 4096), uses))
    return buffers


def overlap_rate(plan):
    perm = plan.permutation
    pairs = len(perm) - 1
    live = sum(1 for i in range(pairs) if plan.overlaps(perm[i], perm[i + 1]))
    return live / max(1, pairs)


def find_positions(plan):
    """First adjacent pair that does / doesn't overlap (or None)."""
    perm = plan.permutation
    noop = ovlp = None
    for i in range(len(perm) - 1):
        if plan.overlaps(perm[i], perm[i + 1]):
            ovlp = i if ovlp is None else ovlp
        else:
            noop = i if noop is None else noop
    return noop, ovlp


def time_call(fn, repeats):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def time_swap_at(plan, i, repeats):
    if i is None:
        return float("nan")
    c = plan.copy()  # toggle on a copy; swap(i) twice returns to the same state
    return time_call(lambda: c.swap(i), repeats)


def time_sweep(plan, repeats):
    n = len(plan.permutation)
    i = n // 2
    total = 0.0
    for _ in range(repeats):
        c = plan.copy()  # copy outside the timed region: measure the sweep only
        t0 = time.perf_counter()
        c.rotate(i, 0)
        for p in range(1, n):
            c.swap(p - 1)
        total += time.perf_counter() - t0
    return total / repeats


SIZES = [10, 30, 100, 300, 1000]
DENSITIES = [("sparse", 0.05), ("medium", 0.2), ("dense", 0.5)]

hdr = (
    f"{'n':>5} {'density':>8} {'ovlp%':>6} | {'build':>8} {'copy':>9} "
    f"{'swap0':>9} {'swapX':>9} {'sweep':>9} | {'swaps/copy':>10} {'n/2':>6}"
)
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

# Skip configs whose one-time graph build (O(ticks * alive^2)) is intractable.
# This is setup cost, not what we are measuring.
BUILD_CAP = 5e7

for n in SIZES:
    for label, span in DENSITIES:
        if n * (span * n) ** 2 > BUILD_CAP:
            print(f"{n:>5} {label:>8}  (skipped: build too slow)", flush=True)
            continue
        rng = random.Random(1000 + n)
        buffers = make_buffers(rng, n, span)
        perm = list(range(n))
        rng.shuffle(perm)

        t_build = time_call(
            lambda: PermutationBasedLayoutSolver(buffers, perm, 10**9, 128),
            max(1, 200 // n),
        )
        plan = PermutationBasedLayoutSolver(buffers, list(perm), 10**9, 128)
        rate = overlap_rate(plan)
        noop_i, ovlp_i = find_positions(plan)

        t_copy = time_call(plan.copy, max(20, 20000 // n))
        t_swap0 = time_swap_at(plan, noop_i, max(50, 50000 // n))
        t_swapx = time_swap_at(plan, ovlp_i, max(50, 50000 // n))
        t_sweep = time_sweep(plan, max(3, 3000 // n))

        # Average swap cost given the measured overlap mix.
        parts = [(rate, t_swapx), (1 - rate, t_swap0)]
        t_swap_avg = sum(w * t for w, t in parts if t == t) or float("nan")
        swaps_per_copy = (
            t_copy / t_swap_avg if t_swap_avg == t_swap_avg else float("nan")
        )

        print(
            f"{n:>5} {label:>8} {rate * 100:>5.0f}% | "
            f"{t_build * 1e3:>7.2f}m {t_copy * 1e6:>8.1f}u "
            f"{t_swap0 * 1e9:>8.0f}n {t_swapx * 1e6:>8.2f}u {t_sweep * 1e6:>8.1f}u | "
            f"{swaps_per_copy:>10.0f} {n // 2:>6}",
            flush=True,
        )
