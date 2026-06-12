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

"""Plot the layout for a fixed ordering of four buffers (no annealing).

Lifetimes are half-open intervals: a buffer with start_time=s and end_time=e
is live at ticks s, s+1, ..., e-1.

The identity ordering [B0, B1, B2, B3] stacks buffers by arrival and produces
a peak height of 22.  Run::

    python toy_layout.py

Writes ``toy_layout.png`` to the current directory.  Requires matplotlib.
"""

# Importing torch_spyre without torch currently sometimes fails.
import torch  # noqa: F401

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.imanishi_xu import (
    ExponentialCoolingSchedule,
    ImanishiXuSolverWithBuffers,
)

buffers = [
    LifetimeBoundBuffer("B0", 8, 0, 2),
    LifetimeBoundBuffer("B1", 4, 1, 5),
    LifetimeBoundBuffer("B2", 2, 2, 6),
    LifetimeBoundBuffer("B3", 8, 3, 6),
]

solver = ImanishiXuSolverWithBuffers(
    buffers,
    size=14,
    alignment=1,
    initial=[0, 1, 2, 3],
    schedule=ExponentialCoolingSchedule(
        t0=10.0, t_end=1.0, steps_per_epoch=10, epochs=10
    ),
)
print("Solving...")
solver.solve()
solver.plot(max_height=22).savefig("toy_layout.png", dpi=300)
print("Saved toy_layout.png")
