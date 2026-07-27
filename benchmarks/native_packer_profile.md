# Native packer profile

Profiling data for the C++ `NativePermutationLayoutSolver`, driven through its
real consumer (`SimulatedAnnealingLayoutSolver`) by
`benchmarks/profile_native_packer.py`. Workload: deterministic random buffer
problems (`horizon=12` — heavy lifetime overlap; capacity = 3× largest buffer —
tight enough to force SA search) across 64–256 buffers, default reheating
schedule. BLAS threads pinned; layout planning is single-threaded.

## Headline: wall-clock A/B (native vs pure-Python packer)

Same instances, `config.native_layout_packer` toggled:

| n (buffers) | speedup (python / native) |
|---|---|
| 32  | 10.9× |
| 64  | 9.3× |
| 128 | 9.0× |

The native packer is ~**9–11× faster** end-to-end on the SA layout workload,
easing slightly as `n` grows (RecomputeAll is super-linear in `n`).

## Where the time goes (py-spy `--native`, self-time within the SA solve)

Sampled with `py-spy record --native` (the only sampler that sees into `_C.so`
on this box — no `perf`, no root). About half of a short run is one-time torch
import; the table below is self-time **within the SA solve subtree** only:

| self-time | frame | side |
|---|---|---|
| **39.6%** | `NativePermutationLayoutSolver::RecomputeAll` | C++ |
| 9.2% | `annealing_step_rotate` | Python |
| 4.6% | `swap` | Python |
| 4.2% | `is_fully_allocated` | Python |
| 3.0% | `quality` | Python |
| 2.9% | `PlaceDecision` | C++ |
| 2.4% | `std::vector<int>::push_back` | C++ |
| 1.8%+1.0%+0.9% | `std::_Hash_bytes` / `_Hashtable` / `_Mod_range_hashing` | C++ (pybind dispatch) |
| 1.6% | `overlaps` | Python |

Split: **~60% C++, ~40% Python SA-loop.**

## Reading

- **`RecomputeAll` is the one hotspot that matters** (~40% of solve self-time,
  and the whole point of the port). Every accepted move recomputes placement
  from scratch; it's super-linear in `n`, which is why the A/B speedup tapers
  and why the truly-incremental port (`packer-native-incremental`) overtakes
  this one past n≈640. If large graphs become common, that's the lever.
- **The per-op pybind boundary overhead the port set out to kill is gone.** The
  boundary marshalling itself is a thin slice; time is now in real C++ compute,
  not per-call Python glue.
- **~40% is still Python** — the SA driver's own move logic (`annealing_step_*`)
  plus per-call round-trips (`is_fully_allocated`, `quality`, `overlaps`). Each
  is a separate pybind crossing; `is_fully_allocated` at 4.2% stands out as a
  per-buffer query in a Python loop. Batching those queries (return an array
  once instead of one call per index) would trim boundary crossings — a
  second-order win, well behind RecomputeAll.
- **~3–4% is pybind type-index hashing** (`_Hash_bytes`/`_Hashtable`) — inherent
  to pybind's per-call argument dispatch; not worth chasing.

Flamegraph: `benchmarks/results/native_packer_flamegraph.svg`.

## Inside RecomputeAll (phase-level wall-clock)

`RecomputeAll` was split into `[[gnu::noinline]]` phase helpers (behaviour
bit-identical, differential-tested; the split is free at `-O2`, A/B confirmed)
so py-spy can attribute wall-clock self-time per phase. Profiled via the
torch-free standalone module (`benchmarks/standalone_packer`) driven by
`microbench_recompute.py` — a tight `rotate` loop, one `RecomputeAll` per call.
py-spy `--native`, 1,859 samples, 80% in the packer subtree:

| self-time | phase | what it does |
|---|---|---|
| **67.3%** | **GatherCandidates** | for each position, scan idx's full time-overlap set and keep earlier-positioned, eligible buffers |
| 6.2% | RecomputeAll (self) | outer loop control, early-stop check |
| 6.1% | MarkIntervals | saturation bookkeeping over idx's live intervals |
| 4.5% | PlaceDecision | stack/in-place/align placement decision |
| 3.6% | RebuildPositions | rebuild inverse permutation |
| 1.9% | `push_back` | append to `cand_` (part of GatherCandidates) |

**GatherCandidates is ~2/3 of all recompute time** (≈69% with its `push_back`).
It re-scans the *entire* overlap set of every buffer on every recompute, keeping
only the earlier-positioned ones — with heavy lifetime overlap that is ~O(n) per
position, i.e. ~O(n²) per `RecomputeAll`. That is the single clear optimization
target (e.g. iterate only placed buffers, or index overlaps by position, or
maintain candidates incrementally). Everything else is minor.

Line-level detail *within* GatherCandidates needs real cycle sampling
(`perf annotate`), unavailable on the Spyre dev box (no perf; py-spy is
function-level and throttled; callgrind crashes under this box's valgrind/Python).
The standalone module builds torch-free anywhere perf exists — see
`benchmarks/standalone_packer/README.md`.

## Optimization: aggregate fast path for the candidate scan

`PlaceDecision`, for a buffer with **no in-place partner**, needs only two facts
about its overlapping earlier buffers: (1) did any get evicted (a `None`
dominates → this buffer is evicted too), and (2) their max top (the floor to
stack on). Both are per-interval aggregates. The sweep already tracks
`placed_at_` / `has_none_at_`; adding a per-interval running **max-top**
(`max_top_at_`) lets the common case read eviction + floor straight off the
buffer's intervals in `O(hi-lo)` and skip gathering the candidate list entirely.
Bit-exact (a buffer's intervals are exactly the times it overlaps earlier
buffers). Buffers *with* in-place partners still gather (they need the real list
for partner-slot logic). Verified against the Python oracle by the stress
differential suite (116 tests, thousands of seeds).

Measured (standalone micro-driver, `rotate` loop, loop-only after subtracting
~0.06 s startup):

| n | scan (old) | fast path | speedup |
|---|---|---|---|
| 256 | 0.65 s | 0.33 s | 1.9× |
| 512 | 1.03 s | 0.48 s | 2.1× |
| 1024 | 1.49 s | 0.75 s | 2.0× |

Full SA A/B (native, n=32/64/128): **2.835 s → 2.294 s (~19% faster)**; native
vs pure-Python rose 9.1× → **11.6×**.

Note: the saturation early-stop already bounds the scan (it only runs until the
layout saturates), so this is a ~2× constant-factor win on `RecomputeAll`, not
an asymptotic-class change. Fast-path coverage = fraction of buffers **without**
in-place partners: ~57% in this in-place-heavy synthetic workload (25% child
probability). Real graphs with sparser in-place reuse would see higher coverage
and a larger win — worth measuring on the captured graphs with `perf` on a
machine that has it.
