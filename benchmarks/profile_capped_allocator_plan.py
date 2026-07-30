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

"""Compare reference vs incremental CappedAllocatorPlan performance.

Run from the repository root so ``torch_spyre`` imports in place::

    python benchmarks/profile_capped_allocator_plan.py

Reports, per buffer count: build time for both plans, reference swap time
(reference rebuilds on every swap, so any swap is O(n^2)), and incremental
swap time for both a realistic random-swap mix (mostly O(1) no-ops) and the
worst case where every swap hits an overlapping pair (full propagation).
See capped_allocator_plan_results.md for a sample run and analysis.
"""

import random
import time

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    ReferencePermutationBasedLayoutSolver,
)


def make_buffers(rng, n):
    """Localized half-open lifetimes [start, end) (length ~ uniform 1..8) over a
    horizon of n ticks, so overlap density stays bounded and the workload looks
    like a real schedule rather than everything-alive-at-once."""
    buffers = []
    for i in range(n):
        start = rng.randint(0, n)
        end = start + rng.randint(1, 8)
        size = rng.randint(1, 4096)
        # Half-open lifetime [start, end); uses records the first and last tick.
        uses = [start] if end == start + 1 else [start, end - 1]
        buffers.append(LifetimeBoundBuffer(f"b{i}", size, uses, in_place_parents=[]))
    for ci in range(1, n):
        if rng.random() < 0.25:
            pi = rng.randrange(max(0, ci - 12), ci)
            parent = buffers[pi]
            child = buffers[ci]
            # In-place child: start_time == parent.end_time - 1 (the in-place
            # invariant parent.end_time == child.start_time + 1), ending a random
            # bit later. start_time/end_time derive from uses, so set uses.
            new_start = parent.end_time - 1
            new_end = parent.end_time + rng.randint(0, 8)
            if new_end == new_start + 1:
                child.uses = [new_start]
            else:
                child.uses = [new_start, new_end - 1]
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def time_it(fn, repeat):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def overlapping_i(plan, rng, tries=30):
    perm = plan.permutation
    for _ in range(tries):
        i = rng.randrange(len(perm) - 1)
        if plan.overlaps(perm[i], perm[i + 1]):
            return i
    return rng.randrange(len(perm) - 1)


def avg_swap_random(plan, rng, n_swaps):
    """Random adjacent swaps -- a realistic mix of no-ops and real work."""
    noops = 0
    t0 = time.perf_counter()
    for _ in range(n_swaps):
        i = rng.randrange(len(plan.permutation) - 1)
        x, y = plan.permutation[i], plan.permutation[i + 1]
        if not plan.overlaps(x, y):
            noops += 1
        plan.swap(i)
    return (time.perf_counter() - t0) / n_swaps, noops / n_swaps


def avg_swap_overlapping(plan, rng, n_swaps):
    """Only swaps of overlapping pairs -- the full propagation path."""
    t0 = time.perf_counter()
    for _ in range(n_swaps):
        plan.swap(overlapping_i(plan, rng))
    return (time.perf_counter() - t0) / n_swaps


SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
CAP_FRACTION = 0.3  # capacity = 30% of total size (forces eviction pressure)

hdr = (
    f"{'n':>6} | {'bld ref':>8} {'bld fast':>8} | "
    f"{'swap ref':>10} | {'rnd fast':>9} {'(noop%)':>7} {'spdup':>6} | "
    f"{'ovlp fast':>9} {'spdup':>6}"
)
print(hdr)
print("-" * len(hdr))

for n in SIZES:
    rng = random.Random(1234 + n)
    buffers = make_buffers(rng, n)
    total = sum(b.size for b in buffers)
    cap = int(total * CAP_FRACTION)
    perm = list(range(n))
    rng.shuffle(perm)

    build_repeat = max(1, 2000 // n)
    t_build_ref = time_it(
        lambda: ReferencePermutationBasedLayoutSolver(buffers, perm, cap, 128),
        build_repeat,
    )
    t_build_fast = time_it(
        lambda: PermutationBasedLayoutSolver(buffers, perm, cap, 128), build_repeat
    )

    n_swaps = max(50, min(3000, 300000 // n))
    # swap ref: reference rebuilds regardless, so any swap sequence is O(n^2).
    ref = ReferencePermutationBasedLayoutSolver(buffers, list(perm), cap, 128)
    s_ref = avg_swap_overlapping(ref, random.Random(7), n_swaps)

    fast_rnd = PermutationBasedLayoutSolver(buffers, list(perm), cap, 128)
    s_fast_rnd, noop = avg_swap_random(fast_rnd, random.Random(7), n_swaps)

    fast_ovlp = PermutationBasedLayoutSolver(buffers, list(perm), cap, 128)
    s_fast_ovlp = avg_swap_overlapping(fast_ovlp, random.Random(7), n_swaps)

    print(
        f"{n:>6} | {t_build_ref * 1e3:>7.2f}m {t_build_fast * 1e3:>7.2f}m | "
        f"{s_ref * 1e6:>8.1f}us | {s_fast_rnd * 1e6:>7.1f}us {noop * 100:>6.0f}% "
        f"{s_ref / s_fast_rnd:>5.0f}x | "
        f"{s_fast_ovlp * 1e6:>7.1f}us {s_ref / s_fast_ovlp:>5.0f}x"
    )
