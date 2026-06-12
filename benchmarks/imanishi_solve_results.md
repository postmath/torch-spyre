# Profiling a full ImanishiXu solve

Where time goes in one full anneal on the `random_buffers` example workload
(seed 0, N=100, first-fit init, 30x100 exponential schedule = 3000 steps), and
how two placement optimizations moved it.

Reproduce with:

```bash
python benchmarks/profile_imanishi_solve.py
```

(The line-by-line pass needs `line_profiler`: `uv pip install line_profiler`.)

## Where the time is

The solve is essentially `swap` (~98% cumulative). Each annealing step copies
the plan, probes reinsertion positions with a forward sweep of `swap`s on the
copy, applies one accepted `rotate` to the live plan, then runs a cleanup
sweep. So the cost funnels through:

```
solve -> anneal -> swap -> _recompute_address -> _placement_decision
```

84% of `swap` calls are O(1) no-ops (non-overlapping adjacent pair). Of the
real work, `_recompute_address` (re-placing one buffer's address) dominates, and
within it `_placement_decision` is ~all of it.

## Optimization progression

| version | wall | `_placement_decision` self | hot inner-loop work |
|---|---:|---:|---|
| baseline | 25.4 s | 3.02 s | 30.2M `_in_place_pair` + 30.2M `_top` |
| + in-place partner precompute | 19.8 s | 0.72 s | 30.2M `_top` |
| + inline `_top` / precomputed `_sizes` | **15.7 s** | 0.72 s | none (arithmetic inlined) |

**−38% overall**, behaviour unchanged (`best_quality` identical at every step;
full + stress differential suites pass).

1. **In-place partner precompute.** `_placement_decision` probed *every*
   candidate with `_in_place_pair` to find an in-place partner, but partners are
   static and sparse. Precomputing each buffer's partner set (its declared
   parents plus the children that declare it) and probing only those removed
   ~30M calls that almost all returned "not a pair".
2. **Inline `_top` with precomputed sizes.** The remaining hot loop,
   `max(_top(p) for p in candidates)` run once per placed buffer, made ~30M
   method calls each doing a dataclass attribute lookup. Precomputing an
   immutable `_sizes` list and inlining `_top` as `addr[p] + sizes[p]` deleted
   those calls.

## Final breakdown (cProfile, 15.7 s)

| self | calls | function |
|---:|---:|---|
| 3.21 s | 506 k | `swap` |
| 1.89 s | 1.97 M | `max` (builtin) |
| 1.83 s | 1.9 M | `_recompute_address` |
| 1.56 s | 32.1 M | `_placement_decision` max scan (`addr[p]+sizes[p]`) |
| 0.84 s | 1.57 M | `Profile.segments` |
| 0.72 s | 1.9 M | `_placement_decision` |

`_placement_decision` line-by-line is now the irreducible max scan (68%) plus
`_align_up` (17%); the in-place probe is invisible. What remains is genuinely
load-bearing: the max-top scan (must read every candidate's top), the swap /
propagation bookkeeping, and the profile-splice maintenance
(`segments` / `splice` / `relabel` / `_coalesce_segments`, ~4-5 s combined).

Further levers (lower ROI, higher risk, not taken): special-casing
`_align_up` for `alignment == 1` (example-specific; real Spyre uses 128), or a
maintained `tops` array (saves the per-element add at the cost of a synced
invariant).
