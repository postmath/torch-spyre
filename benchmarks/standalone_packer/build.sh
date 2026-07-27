#!/usr/bin/env bash
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

# Build the standalone, torch-free packer module (packer_native.so) for
# profiling. Needs only pybind11 (`pip install pybind11`) and a C++20 compiler
# -- no torch, no Spyre SDK, no Spyre card. Portable to any machine with perf.
#
# Mirrors the shipping build flags (setup.py): -O2 (implicit) + -g + -std=c++20.
# Use -fno-omit-frame-pointer so perf can unwind cheaply without DWARF.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC="$HERE/../../torch_spyre/csrc"
OUT="$HERE/packer_native$(python3-config --extension-suffix 2>/dev/null || echo .so)"

PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"

set -x
# shellcheck disable=SC2086
g++ -O2 -g -fno-omit-frame-pointer -std=c++20 -Wall -shared -fPIC \
    $PYBIND_INCLUDES \
    -I"$CSRC" \
    "$CSRC/perm_layout_native.cpp" \
    "$HERE/packer_module.cpp" \
    -o "$OUT"
set +x
echo "built: $OUT"
