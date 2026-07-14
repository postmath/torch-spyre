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

"""Cross-process regression test for FirstFit in-place parent determinism."""

import json
import os
import subprocess
import sys
from unittest import TestCase

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_SNIPPET = """
import json
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import FirstFitLayoutSolver
def b(n, s, st, en, ipp=None):
    return LifetimeBoundBuffer(name=n, size=s, uses=[st, en - 1], in_place_parents=ipp or [])
# c has two in-place parents at distinct addresses, both in-place candidates for
# its gap -> _build_gaps' set iteration decides in_place_parents[0].
bufs = [b("pA", 100, 0, 3), b("pB", 80, 1, 3), b("c", 50, 2, 5, ["pA", "pB"])]
FirstFitLayoutSolver(10_000, 1).plan_layout(bufs)
print("RESULT " + json.dumps({x.name: x.address for x in bufs}))
"""


def _run(hashseed):
    env = dict(
        os.environ,
        PYTHONHASHSEED=str(hashseed),
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
    )
    p = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO_ROOT,
        timeout=60,
    )
    assert p.returncode == 0, p.stderr
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


class FirstFitDeterminismTest(TestCase):
    """FirstFit placement must not depend on PYTHONHASHSEED (set-iteration order)."""

    def test_inplace_parent_choice_is_hashseed_independent(self):
        base = _run(0)
        for hs in range(1, 10):
            self.assertEqual(_run(hs), base, f"PYTHONHASHSEED={hs}")
