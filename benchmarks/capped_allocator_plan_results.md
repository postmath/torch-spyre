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
    16 |    0.02m    0.06m |     20.0us |    11.2us     61%     2x |    22.8us     1x
    32 |    0.06m    0.18m |     62.2us |    18.2us     71%     3x |    81.4us     1x
    64 |    0.16m    0.43m |    168.8us |     8.7us     86%    19x |    59.5us     3x
   128 |    0.53m    1.21m |    519.6us |     5.8us     95%    89x |    73.5us     7x
   256 |    1.81m    4.08m |   1783.6us |     6.7us     96%   265x |   131.4us    14x
   512 |    7.42m   16.37m |   7497.7us |    10.5us     97%   712x |   218.8us    34x
  1024 |   29.91m   64.05m |  28795.0us |     4.3us     99%  6707x |   228.3us   126x
  2048 |  114.04m  244.48m | 110663.9us |    10.3us     99% 10791x |   255.0us   434x
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
  overlapping pair (full propagation) stays nearly flat at large `n`
  (228 → 255 µs from n=1024 to n=2048), giving **434x at n=2048** with a
  widening gap.

**Bottom line.** Reference swap is O(n^2); incremental swap is driven by the
neighbour graph and scales with the *affected* set rather than `n`. For a local
search performing thousands of swaps, the ~2.3x build penalty is negligible and
the per-swap speedup is one to four orders of magnitude.

### Implementation note

`swap` maintains the position index in O(1), processes only the affected
buffers via a min-heap over positions, and propagates through `above_neighbors`
(plus the resters on any column a buffer joins in-place) rather than scanning
all positions. The remaining mild growth in the worst-case column comes from
re-deriving each affected buffer's below-set, which still scans candidate
buffers; bounding that to the buffer's actual overlap set would remove it.
