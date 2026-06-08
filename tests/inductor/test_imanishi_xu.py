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

"""End-to-end tests for the Imanishi/Xu simulated-annealing layout solver."""

import copy
import random as rnd
from unittest import TestCase

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    PermutationBasedLayoutSolver,
)
from torch_spyre._inductor.scratchpad.imanishi_xu import (
    ExponentialCoolingSchedule,
    ImanishiXuLayoutSolver,
)


def _random_buffers(rng, n, horizon=12, max_size=200):
    """Half-open lifetimes, some in-place children (parent.end == child.start+1)."""
    buffers = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        buffers.append(LifetimeBoundBuffer(f"b{i}", size, start, end))
    for child_i in range(1, n):
        if rng.random() < 0.25:
            parent = buffers[rng.randrange(child_i)]
            child = buffers[child_i]
            child.start_time = parent.end_time - 1
            child.end_time = max(child.end_time, parent.end_time)
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def _short_schedule():
    return ExponentialCoolingSchedule(t0=100.0, t_end=1.0, steps_per_epoch=5, epochs=4)


def _assert_feasible(buffers, capacity):
    """Committed buffers fit below capacity and never address-overlap a
    time-overlapping peer (an in-place pair may share its base address)."""
    committed = [b for b in buffers if b.address is not None]
    for b in committed:
        assert b.address + b.size <= capacity, f"{b.name} exceeds capacity"
    for a in range(len(committed)):
        for c in range(a + 1, len(committed)):
            u, v = committed[a], committed[c]
            if not u.overlaps_in_time(v):
                continue
            if u.name in v.in_place_parents or v.name in u.in_place_parents:
                continue  # in-place pair may share an address
            assert u.address + u.size <= v.address or v.address + v.size <= u.address, (
                f"{u.name}@{u.address}+{u.size} overlaps {v.name}@{v.address}+{v.size}"
            )


def _committed_total(buffers):
    return sum(b.size for b in buffers if b.address is not None)


class ImanishiXuTests(TestCase):
    def _run(self, buffers, capacity, *, initial, seed, alignment=128):
        solver = ImanishiXuLayoutSolver(
            capacity,
            alignment,
            initial=initial,
            schedule=_short_schedule(),
            random=rnd.Random(seed),
        )
        return solver.plan_layout(buffers)

    def test_finalized_layout_is_feasible(self):
        for seed in range(60):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            self._run(buffers, cap, initial="first_fit", seed=seed)
            _assert_feasible(buffers, cap)

    def test_annealing_never_worse_than_initial(self):
        # Starting from a known permutation, the tracked best (and thus the
        # finalized committed total) can only improve on the initial layout.
        for seed in range(60):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            initial = list(range(n))
            rng.shuffle(initial)
            initial_quality = PermutationBasedLayoutSolver(
                copy.deepcopy(buffers), list(initial), cap, 128
            ).quality()

            self._run(buffers, cap, initial=initial, seed=seed)
            self.assertGreaterEqual(_committed_total(buffers), initial_quality, seed)
            _assert_feasible(buffers, cap)

    def test_deterministic_with_seed(self):
        rng = rnd.Random(0)
        n = 7
        base = _random_buffers(rng, n)
        cap = max(b.size for b in base) * 3

        first = copy.deepcopy(base)
        self._run(first, cap, initial="first_fit", seed=42)
        second = copy.deepcopy(base)
        self._run(second, cap, initial="first_fit", seed=42)

        self.assertEqual([b.address for b in first], [b.address for b in second])
