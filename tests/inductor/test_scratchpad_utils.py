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

"""Unit tests for calculate_liveness."""

import unittest
from dataclasses import dataclass

from torch_spyre._inductor.scratchpad.utils import calculate_liveness


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
    return _Graph(input_names, [_Op(r, w) for r, w in op_specs])


class TestCalculateLiveness(unittest.TestCase):
    def test_graph_inputs(self):
        """Unused inputs stay [], used inputs accumulate op indices, all appear in result."""
        graph = _make_graph(
            ["a", "b", "unused"],
            [
                (["a"], ["x"]),  # op0: reads a
                (["x"], ["y"]),  # op1: doesn't read a or b
                (["a", "b"], ["z"]),  # op2: reads a again, reads b once
            ],
        )
        liveness = calculate_liveness(graph)
        self.assertIn("unused", liveness)
        self.assertEqual(liveness["unused"], [])
        self.assertEqual(liveness["a"], [0, 2])  # read at op0 and op2
        self.assertEqual(liveness["b"], [2])  # read only at op2

    def test_intermediate_buffers(self):
        """Intermediates accumulate entries for write and all subsequent reads."""
        graph = _make_graph(
            [],
            [
                ([], ["a"]),  # op0: writes a
                (["a"], ["b"]),  # op1: reads a, writes b
                (["a", "b"], ["c"]),  # op2: reads a and b
            ],
        )
        liveness = calculate_liveness(graph)
        self.assertEqual(liveness["a"], [0, 1, 2])  # written op0, read op1 and op2
        self.assertEqual(liveness["b"], [1, 2])  # written op1, read op2
        self.assertEqual(liveness["c"], [2])  # written op2, never read


if __name__ == "__main__":
    unittest.main()
