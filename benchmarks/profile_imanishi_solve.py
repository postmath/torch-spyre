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

"""Profile a full ImanishiXu solve on the random_buffers example workload.

Reproduces examples/scratchpad/random_buffers.py exactly (seed 0, N=100,
first-fit init, 30x100 exponential schedule) and profiles ``solver.solve()``:

- a function-level cProfile pass (always), and
- a line-by-line pass over the placement / swap / profile hot functions, if
  ``line_profiler`` is installed (``uv pip install line_profiler``).

Run from the repository root::

    python benchmarks/profile_imanishi_solve.py
"""

import cProfile
import copy
import io
import math
import pstats
import random as rnd
import time

import torch  # noqa: F401  (importing torch_spyre without torch sometimes fails)

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad import permutation_layout as PL
from torch_spyre._inductor.scratchpad.imanishi_xu import (
    ImanishiXuSolverWithBuffers,
    peak_memory_load,
    ExponentialCoolingSchedule,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)


def _random_buffer(name, size_range, time_range, random):
    """Identical to the example: biased toward large sizes, short lifetimes."""
    duration = random.randrange((time_range - 1) // 2)
    duration = duration * duration // (time_range - 1)
    t_start = random.randrange(time_range - duration)
    t_end = t_start + duration + 1
    size = random.randrange(size_range)
    size = max(1, math.isqrt(size * size_range))
    # Live at ticks [t_start, t_end] inclusive; uses records the first and last.
    uses = [t_start] if t_end == t_start else [t_start, t_end]
    return LifetimeBoundBuffer(name, size, uses)


def build_solver():
    """Reproduce the example's solver (seed 0, N=100, first-fit init, 30x100)."""
    random = rnd.Random(0)
    n = 100
    buffers = [_random_buffer(f"B{i}", 1_000_000, n, random) for i in range(n)]
    capacity = peak_memory_load(buffers) // 2
    ff = copy.deepcopy(buffers)
    FirstFitLayoutSolver(capacity).plan_layout(ff)  # example parity (no rng use)
    return ImanishiXuSolverWithBuffers(
        buffers,
        capacity,
        alignment=1,
        initial="first_fit",
        random=random,
        schedule=ExponentialCoolingSchedule(
            t_initial=500000.0, t_final=50000.0, steps_per_epoch=30, epochs=100
        ),
    )


def function_level():
    solver = build_solver()
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    solver.solve()
    pr.disable()
    wall = time.perf_counter() - t0
    print("\n##### cProfile (function level) #####")
    print(f"wall {wall * 1e3:.0f} ms; best_quality={solver.best_quality}")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(25)
    print(s.getvalue())


def line_level():
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print(
            "\n(line_profiler not installed; skipping line-by-line pass. "
            "Install with `uv pip install line_profiler`.)"
        )
        return
    solver = build_solver()
    PBLS = PL.PermutationBasedLayoutSolver
    BASE = PL.PermutationBasedLayoutSolverBase
    IX = ImanishiXuSolverWithBuffers
    funcs = [
        IX.anneal,
        IX.annealing_step_rotate,
        IX.annealing_step_swap,
        PBLS.swap,
        PBLS._recompute_address,
        PBLS.contact_at,
        PBLS.copy,
        PBLS._update_profiles_for_swap,
        PBLS._splice_half,
        BASE._placement_decision,
        BASE.rotate,
        BASE._top,
        BASE.overlaps,
        BASE._in_place_pair,
        BASE.is_fully_allocated,
        BASE._align_up,
        PL.Profile.splice,
        PL.Profile.segments,
        PL.Profile.relabel,
        PL.Profile.label_at,
        PL.Profile.label_set,
        PL._coalesce_segments,
    ]
    lp = LineProfiler()
    for f in funcs:
        lp.add_function(f)
    t0 = time.perf_counter()
    lp.runcall(solver.solve)
    wall = time.perf_counter() - t0
    print("\n##### line_profiler (line level) #####")
    print(f"instrumented wall {wall * 1e3:.0f} ms; best_quality={solver.best_quality}")
    lp.print_stats(output_unit=1e-6, stripzeros=True)


if __name__ == "__main__":
    function_level()
    line_level()
