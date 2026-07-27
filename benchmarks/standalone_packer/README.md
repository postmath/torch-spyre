# Standalone packer — profiling harness

A torch-free, Spyre-free build of `NativePermutationLayoutSolver` for profiling.
`perm_layout_native.cpp` depends only on pybind11 + the STL, so it compiles into
a tiny standalone module anywhere a C++20 compiler and pybind11 exist — **no
torch, no Spyre SDK, no Spyre card required**.

Two reasons this exists:

- **Enables `perf` on a machine that has it** (e.g. a laptop). `perf record` +
  `perf annotate` give real wall-clock, cycle-sampled, *line-level* attribution
  inside `RecomputeAll` — which py-spy `--native` (function-level only, and
  throttled to ~10 Hz on the Spyre dev box) and callgrind (instruction-count,
  and it crashes under the dev box's valgrind/Python combo) cannot.
- **Clean profiles**: zero torch import, so ~all samples land in the packer.

## Build

```bash
pip install pybind11          # if not already present
bash benchmarks/standalone_packer/build.sh
```

Produces `packer_native.<abi>.so` in this directory. Flags mirror the shipping
build (`-O2 -g -std=c++20`) plus `-fno-omit-frame-pointer` for cheap perf
unwinding.

## Profile with perf (the wall-clock line-level view)

```bash
export PYTHONPATH=benchmarks/standalone_packer

# Record (frame-pointer call graph; the build keeps frame pointers).
perf record -g --call-graph fp -F 999 -o /tmp/packer.perf -- \
    python3 benchmarks/microbench_recompute.py --n 256 --iters 3000000

# Function-level breakdown within the rotate/RecomputeAll loop:
perf report -i /tmp/packer.perf --stdio

# Line-level, annotated onto the C++ source (THE wall-clock detail):
perf annotate -i /tmp/packer.perf --stdio \
    -s 'torch_spyre::scratchpad::NativePermutationLayoutSolver::GatherCandidates'
# ...repeat -s for RecomputeAll / PlaceDecision / MarkIntervals as wanted.
```

If `perf annotate` line mapping looks fused (aggressive `-O2` inlining), rebuild
with `-O2 -fno-inline-functions-called-once` or `-Og` by editing `build.sh` —
`-Og` gives the cleanest line mapping at a small realism cost.

## Or callgrind (exact, deterministic; instruction-count not wall-clock)

```bash
export PYTHONPATH=benchmarks/standalone_packer
valgrind --tool=callgrind --dump-instr=yes --cache-sim=yes \
    --callgrind-out-file=/tmp/cg.out \
    python3 benchmarks/microbench_recompute.py --n 256 --iters 4000
callgrind_annotate --auto=yes --inclusive=no /tmp/cg.out   # per-line Ir + cache
# or: kcachegrind /tmp/cg.out
```

`--cache-sim=yes` answers whether the candidate/overlap scans are memory-bound.

## The workload

`microbench_recompute.py` builds one representative problem (heavy lifetime
overlap, tight capacity) and runs a tight loop of random `rotate()` calls — each
triggers exactly one `RecomputeAll`, so the profile is almost entirely the
recompute path with no Python SA-loop or pybind-per-op noise. It imports
`packer_native` if on `PYTHONPATH`, else falls back to `torch_spyre._C`.
