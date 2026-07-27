/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Standalone pybind module exposing ONLY NativePermutationLayoutSolver, built
// from torch_spyre/csrc/perm_layout_native.cpp with no torch and no Spyre SDK
// (the packer depends only on pybind11 + the STL). Its purpose is profiling:
//
//   * it builds anywhere pybind11 + a C++20 compiler exist (e.g. a laptop with
//     perf but no Spyre card), enabling `perf record` / `perf annotate` for
//     real wall-clock line-level attribution;
//   * it has zero torch import, so it runs cleanly under callgrind (torch's
//     AVX instructions crash valgrind) and leaves no import noise in a profile.
//
// See benchmarks/standalone_packer/build.sh.

#include <pybind11/pybind11.h>

#include "perm_layout_native.h"

namespace py = pybind11;

PYBIND11_MODULE(packer_native, m) {
  torch_spyre::scratchpad::register_perm_layout_native(m);
}
