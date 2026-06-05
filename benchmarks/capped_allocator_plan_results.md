# CappedAllocatorPlan: reference vs incremental

Performance comparison of the two `CappedAllocatorPlan` implementations in
`torch_spyre/_inductor/scratchpad/plan_solver.py`:

- **reference** (`ReferenceCappedAllocatorPlan`) — rebuilds the whole layout
  from scratch on every `swap`.
- **incremental** (`CappedAllocatorPlan`) — maintains a neighbour graph and
  re-places only the buffers a swap actually affects.

Reproduce with:

```bash
python benchmarks/profile_capped_allocator_plan.py
```

## Workload

- Localized lifetimes: each buffer lives for ~1–8 ticks over a horizon of `n`
  ticks, so overlap density stays bounded (like a real schedule rather than
  everything-alive-at-once).
- Capacity = 30% of total buffer size, to force eviction pressure.
- ~25% of buffers are in-place children of a nearby earlier buffer.
- 128-byte alignment.

## Sample run

Times: `m` = milliseconds, `us` = microseconds. `noop%` is the fraction of
random adjacent swaps that touch non-overlapping buffers and return in O(1).
`spdup` is reference-swap time divided by the corresponding incremental time.

```
     n |  bld ref bld fast |   swap ref |  rnd fast (noop%)  spdup | ovlp fast  spdup
-------------------------------------------------------------------------------------
    16 |    0.02m    0.05m |     20.0us |    11.6us     61%     2x |    31.5us     1x
    32 |    0.06m    0.17m |     59.6us |    24.0us     71%     2x |   116.9us     1x
    64 |    0.16m    0.42m |    165.1us |    11.8us     86%    14x |    81.2us     2x
   128 |    0.50m    1.23m |    513.4us |     8.0us     95%    64x |   105.5us     5x
   256 |    1.82m    4.16m |   1799.4us |    24.0us     96%    75x |   225.2us     8x
   512 |    7.45m   16.57m |   7530.6us |    14.7us     97%   514x |   333.8us    23x
  1024 |   28.82m   62.36m |  28623.7us |     4.9us     99%  5871x |   382.3us    75x
  2048 |  109.70m  256.88m | 110691.6us |    19.9us     99%  5571x |   468.4us   236x
```

(Single run on one machine; absolute numbers vary with hardware, but the
scaling trends are the point.)

## Interpretation

**Build.** The incremental plan is ~2.3x slower to build than the reference —
it pays a one-time cost to construct the neighbour graph on top of computing
addresses. Both are roughly O(n^2). This penalty is amortized away after a
handful of swaps.

**Swap.** This is what the neighbour graph buys:

- **Reference** rebuilds on every swap, so each swap is **O(n^2)** — 20 µs at
  n=16 growing to 111 ms at n=2048.
- **Incremental, realistic random swaps** (`rnd fast`): with localized
  lifetimes, adjacent permutation entries rarely overlap, so most swaps are
  O(1) no-ops (99% at n ≥ 1024). Average stays ~5–24 µs — up to **~5900x
  faster**.
- **Incremental, worst case** (`ovlp fast`): forcing *every* swap onto an
  overlapping pair (full propagation) still grows only ~linearly (31 → 468 µs),
  giving **236x at n=2048** with a widening gap.

**Bottom line.** Reference swap is O(n^2); incremental swap is roughly O(n)
even on the all-real-work path. For a local search performing thousands of
swaps, the ~2.3x build penalty is negligible and the per-swap speedup is one to
three orders of magnitude.

### Note on the residual O(n) per swap

The incremental swap currently still does O(n) work per call independent of how
much actually changes: it rebuilds the position array and, when an address
changes, marks every later-positioned overlapping buffer dirty via a linear
scan. The propagation itself is local. Replacing those linear scans with
`above_neighbors`-driven propagation would make the worst-case swap sub-linear
in `n` (proportional to the number of buffers actually affected).
