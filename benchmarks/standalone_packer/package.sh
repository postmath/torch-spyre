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

# Assemble a self-contained profiling kit for a machine that has profiling tools
# but no torch / no Spyre SDK / no Spyre card (e.g. a laptop). Copies the live
# packer sources (never stale) + the driver into a flat directory and tars it.
#
#   bash benchmarks/standalone_packer/package.sh            # -> /tmp/packer-profiling-kit.tar.gz
#   bash benchmarks/standalone_packer/package.sh /some/dir  # choose output dir
#
# On the laptop:  tar xzf packer-profiling-kit.tar.gz && cd packer-profiling-kit
#                 python3 setup.py build_ext --inplace      # needs: pip install pybind11
#                 (then the perf / samply recipe in README.md)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC="$HERE/../../torch_spyre/csrc"
OUTDIR="${1:-/tmp}"
KIT="$OUTDIR/packer-profiling-kit"

rm -rf "$KIT"
mkdir -p "$KIT"

# Live sources -- copied fresh each run so the kit never drifts from the tree.
cp "$CSRC/perm_layout_native.cpp" "$KIT/"
cp "$CSRC/perm_layout_native.h" "$KIT/"
cp "$HERE/packer_module.cpp" "$KIT/"
cp "$HERE/../microbench_recompute.py" "$KIT/"
cp "$HERE/README.md" "$KIT/README.dev-box.md"  # dev-box notes, for reference

# Portable, cross-platform build via pybind11's setuptools helpers (handles the
# Linux/macOS shared-object flags and extension suffix automatically).
cat > "$KIT/setup.py" <<'PY'
# Build the standalone, torch-free packer module for profiling:
#   pip install pybind11
#   python3 setup.py build_ext --inplace
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "packer_native",
    ["perm_layout_native.cpp", "packer_module.cpp"],
    cxx_std=20,
    # -g keeps DWARF line info; frame pointers let perf unwind without DWARF.
    extra_compile_args=["-g", "-fno-omit-frame-pointer"],
)
setup(name="packer_native", ext_modules=[ext], cmdclass={"build_ext": build_ext})
PY

cp "$HERE/KIT_README.md" "$KIT/README.md"

TARBALL="$OUTDIR/packer-profiling-kit.tar.gz"
tar czf "$TARBALL" -C "$OUTDIR" packer-profiling-kit
echo "kit:     $KIT"
echo "tarball: $TARBALL"
echo "files:"
ls -1 "$KIT"
