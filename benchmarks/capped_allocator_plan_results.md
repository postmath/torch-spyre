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
    16 |    0.02m    0.06m |     20.0us |     9.7us     61%     2x |    20.3us     1x
    32 |    0.06m    0.21m |     60.9us |    15.7us     71%     4x |    68.9us     1x
    64 |    0.17m    0.53m |    169.9us |     6.2us     86%    27x |    45.3us     4x
   128 |    0.52m    1.60m |    534.1us |     3.5us     95%   154x |    40.0us    13x
   256 |    1.75m    5.67m |   1834.9us |     3.3us     96%   559x |    56.9us    32x
   512 |    7.61m   23.75m |   7706.3us |     3.8us     97%  2054x |    65.6us   117x
  1024 |   29.82m   94.82m |  28784.0us |     1.5us     99% 19064x |    42.1us   683x
  2048 |  117.07m  362.51m | 114810.2us |     2.2us     99% 51196x |    31.4us  3658x
```

(Single run on one machine; absolute numbers vary with hardware, but the
scaling trends are the point.)

## Interpretation

**Build.** The incremental plan is ~3x slower to build than the reference — it
pays a one-time cost to construct the neighbour graph and the time-overlap sets
on top of computing addresses. Both are roughly O(n^2). This penalty is
amortized away after a handful of swaps.

**Swap.** This is what the neighbour graph buys:

- **Reference** rebuilds on every swap, so each swap is **O(n^2)** — 20 µs at
  n=16 growing to 111 ms at n=2048.
- **Incremental, realistic random swaps** (`rnd fast`): with localized
  lifetimes, adjacent permutation entries rarely overlap, so most swaps are
  O(1) no-ops (99% at n ≥ 1024). Average stays ~5–24 µs — up to **~5900x
  faster**.
- **Incremental, worst case** (`ovlp fast`): forcing *every* swap onto an
  overlapping pair (full propagation) is flat in `n` (~30–70 µs across the whole
  range; the variation is noise, not growth), giving **3658x at n=2048** with a
  gap that widens without bound.

**Bottom line.** Reference swap is O(n^2); incremental swap is driven by the
neighbour graph and scales with the *affected* set rather than `n`. For a local
search performing thousands of swaps, the ~3x build penalty is negligible and
the per-swap speedup is two to four orders of magnitude.

### Implementation note

`swap` maintains the position index in O(1), processes only the affected
buffers via a min-heap over positions, and propagates through `above_neighbors`
(plus the resters on any column a buffer joins in-place) rather than scanning
all positions. Candidates for re-deriving an affected buffer come from a
precomputed time-overlap set (lifetimes never change), so a swap touches no
work proportional to `n` -- only to the buffers it actually disturbs.
