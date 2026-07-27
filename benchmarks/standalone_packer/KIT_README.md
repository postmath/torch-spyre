# Packer profiling kit (torch-free, Linux x86-64 + perf)

Self-contained profiling harness for `NativePermutationLayoutSolver` — the C++
scratchpad-layout packer from torch-spyre. It depends only on **pybind11 + the
STL**, so it builds and runs here with **no torch, no Spyre SDK, no Spyre card**.

Purpose: get real **wall-clock, line-level** profiling with `perf`, which the
Spyre dev box can't do (no perf there; py-spy is function-level and throttled;
callgrind crashes under that box's valgrind/Python). The hot function is
`RecomputeAll`; its dominant phase is `GatherCandidates` (the candidate
overlap-scan).

## Contents

| file | what |
|---|---|
| `perm_layout_native.cpp` / `.h` | the packer (verbatim copies from the tree) |
| `packer_module.cpp` | pybind `PYBIND11_MODULE(packer_native, …)` shim |
| `microbench_recompute.py` | driver: tight `rotate` loop, one `RecomputeAll` each |
| `setup.py` | portable pybind11 build |
| `README.dev-box.md` | the dev-box notes (py-spy/callgrind), for reference |

## 1. Build (with uv)

```bash
uv venv                              # create .venv/ (add --python 3.12 to pin)
uv pip install pybind11 setuptools   # build deps -- a uv venv starts bare
uv run python setup.py build_ext --inplace
```

Produces `packer_native.<abi>.so` here, ABI-tagged for `.venv`'s interpreter.
Flags: `-O2` (shipping default) `-g` `-fno-omit-frame-pointer` (so perf unwinds
via frame pointers, no DWARF needed).

Prereqs: `uv`, a C++20 compiler (g++ ≥ 10 / clang ≥ 11), `perf`. uv supplies the
Python (its managed builds include the dev headers) and pybind11; no system
python3-dev needed. **Build and run with the same interpreter** — hence
`.venv/bin/python` throughout below (the `.so` is per-interpreter).

Common: `--n` = buffer count, `--iters` = number of `rotate` calls (bump until
the run is ~30–60 s for a solid sample count). `--inplace-prob` (default 0.25) is
the fraction of buffers with in-place partners; **lower it toward 0 to see the
fast-path win grow** (partner-free buffers skip the candidate scan). Always
target `.venv/bin/python` directly (not `uv run python`) so the profiler sees the
interpreter, not the uv launcher; the driver imports the local `packer_native.so`
automatically (cwd is on `sys.path`).

Both tools below use `perf_event_open`, so first unlock it once per boot:
`sudo sysctl kernel.perf_event_paranoid=1`.

## 2a. Record + read with samply (recommended)

`samply` is a self-contained Rust profiler — no `linux-tools`/`perf` packages
needed (handy since Ubuntu OEM kernels often don't package `perf`). It gives a
Firefox-Profiler UI with a **source view** (per-line sample counts).

```bash
cargo install samply     # or a release binary from github.com/mstange/samply
samply record .venv/bin/python microbench_recompute.py --n 256 --iters 4000000
```

In the browser: open the **Call Tree**, select
`…NativePermutationLayoutSolver::GatherCandidates` (or `RecomputeAll`), then the
**Source view** shows cycles per C++ source line. Because you build and profile
in this same directory, the source paths in the DWARF resolve locally, so the
line view works with no extra setup.

## 2b. Record + read with perf (if available)

```bash
perf record -e cycles:u -g --call-graph fp -F 999 -o packer.perf -- \
    .venv/bin/python microbench_recompute.py --n 256 --iters 4000000
perf report  -i packer.perf --stdio | head -40           # function level
perf annotate -i packer.perf --stdio \
    -s 'torch_spyre::scratchpad::NativePermutationLayoutSolver::GatherCandidates'
```

`cycles:u` = user-space only (no kernel symbols / `kptr_restrict` needed).
On an OEM kernel with no packaged perf, either `sudo apt install
linux-tools-generic` and call its binary directly
(`$(ls -d /usr/lib/linux-tools/*generic/perf | tail -1)`) — a version-mismatched
perf is fine for user-space sampling — or just use samply above.

`perf annotate` / the samply source view interleave source lines with their
cycle share, so you can see which lines of `GatherCandidates` (the overlap
iterate, the `position_ < pos && eligible_` filter, the `push_back`) cost what,
and whether it's branch- or memory-bound.

If `-O2` inlining fuses the annotation, rebuild with cleaner line mapping: add
`"-fno-inline-functions-called-once"` to `setup.py`'s `extra_compile_args`, or
drop to `"-Og"` (cleanest mapping, slightly less representative), and rebuild.

## Suggested experiments

- **Line-level of the hotspot**: `perf annotate` on `GatherCandidates` at
  `--n 512 --inplace-prob 0.25`. Confirms the overlap-scan line breakdown.
- **Fast-path win vs in-place density**: fixed `--n 512`, sweep
  `--inplace-prob 0.0 0.1 0.25 0.5`; compare `RecomputeAll` self-time and total.
  Real graphs sit somewhere on this curve.
- **Scaling**: fixed `--inplace-prob`, sweep `--n 128 256 512 1024 2048`.

## Interpreting vs the dev-box numbers

On the dev box the optimization (per-interval max-top fast path) measured ~2×
on the `RecomputeAll` loop and ~19% on full SA, at ~57% fast-path coverage.
perf here should confirm the `GatherCandidates` self-time drop and expose the
per-line cost of the remaining scan (the in-place-partner path).
