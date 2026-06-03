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

"""Unit tests for calculate_liveness and the Liveness dataclass."""

import unittest
from dataclasses import dataclass

from torch_spyre._inductor.scratchpad.utils import Liveness, calculate_liveness


@dataclass(frozen=True)
class _Dep:
    name: str


@dataclass
class _ReadWrites:
    reads: frozenset
    writes: frozenset


@dataclass
class _Op:
    read_names: list
    write_names: list

    def get_read_writes(self):
        return _ReadWrites(
            reads=frozenset(_Dep(n) for n in self.read_names),
            writes=frozenset(_Dep(n) for n in self.write_names),
        )


class _Graph:
    def __init__(self, input_names, ops):
        self.graph_input_names = input_names
        self.operations = ops


def _make_graph(input_names, op_specs):
    """Build a _Graph from (reads, writes) op specs."""
    return _Graph(input_names, [_Op(r, w) for r, w in op_specs])


class TestCalculateLiveness(unittest.TestCase):
    def test_unused_input_has_zero_liveness(self):
        graph = _make_graph(
            ["x", "unused"],
            [
                (["x"], ["y"]),  # op0: reads x, writes y; "unused" never touched
            ],
        )
        liveness = calculate_liveness(graph)
        u = liveness["unused"]
        self.assertEqual(u.start, 0)
        self.assertEqual(u.end, 0)
        self.assertEqual(u.reads, [])

    def test_input_single_read(self):
        graph = _make_graph(
            ["x"],
            [
                (["x"], ["y"]),  # op0
            ],
        )
        liveness = calculate_liveness(graph)
        x = liveness["x"]
        self.assertEqual(x.start, 0)
        self.assertEqual(x.end, 1)
        self.assertEqual(x.reads, [0])

    def test_input_multiple_reads(self):
        graph = _make_graph(
            ["x"],
            [
                (["x"], ["y"]),  # op0: reads x
                (["y"], ["z"]),  # op1: doesn't read x
                (["x", "z"], ["w"]),  # op2: reads x again
            ],
        )
        liveness = calculate_liveness(graph)
        x = liveness["x"]
        self.assertEqual(x.start, 0)
        self.assertEqual(x.end, 3)  # last touched at op2 (i=2), end = 3
        self.assertEqual(x.reads, [0, 2])

    def test_intermediate_start_and_end(self):
        graph = _make_graph(
            ["x"],
            [
                (["x"], ["y"]),  # op0: x→y
                (["y"], ["z"]),  # op1: y→z
            ],
        )
        liveness = calculate_liveness(graph)
        y = liveness["y"]
        self.assertEqual(y.start, 0)  # first seen at op0 (i=0)
        self.assertEqual(y.end, 2)  # last touched at op1 (i=1), end = 2
        self.assertEqual(y.reads, [1])

    def test_buffer_first_seen_at_later_op(self):
        graph = _make_graph(
            [],
            [
                ([], ["a"]),  # op0: creates a
                (["a"], ["b"]),  # op1: a→b  (b first appears here)
                (["b"], ["c"]),  # op2: b→c
            ],
        )
        liveness = calculate_liveness(graph)
        b = liveness["b"]
        self.assertEqual(b.start, 1)  # first seen at op1 (i=1)
        self.assertEqual(b.end, 3)  # last touched at op2 (i=2), end = 3
        self.assertEqual(b.reads, [2])

    def test_all_graph_inputs_present_in_output(self):
        """Every name in graph_input_names appears in the returned dict, even if never used."""
        graph = _make_graph(
            ["a", "b", "c"],
            [
                (["a"], ["x"]),  # b and c never touched
            ],
        )
        liveness = calculate_liveness(graph)
        self.assertIn("a", liveness)
        self.assertIn("b", liveness)
        self.assertIn("c", liveness)

    def test_return_type_is_liveness_dataclass(self):
        graph = _make_graph(["x"], [(["x"], ["y"])])
        liveness = calculate_liveness(graph)
        for v in liveness.values():
            self.assertIsInstance(v, Liveness)


if __name__ == "__main__":
    unittest.main()
