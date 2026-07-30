# PermutationBasedLayoutSolver: reference vs incremental

Performance comparison of the two permutation-based layout solvers in
`torch_spyre/_inductor/scratchpad/plan_solver.py`:

- **reference** (`ReferencePermutationBasedLayoutSolver`) — rebuilds the whole
  layout from scratch on every `swap`.
- **incremental** (`PermutationBasedLayoutSolver`) — maintains order-based
  contact profiles and re-places only the buffers a swap actually affects.

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
    16 |    0.02m    0.07m |     20.6us |     8.6us     61%     2x |    16.6us     1x
    32 |    0.06m    0.20m |     67.4us |     7.5us     74%     9x |    30.7us     2x
    64 |    0.17m    0.51m |    177.8us |     3.9us     87%    46x |    26.6us     7x
   128 |    0.58m    1.54m |    579.6us |     2.1us     94%   271x |    24.5us    24x
   256 |    2.04m    5.57m |   2045.9us |     1.8us     97%  1136x |    29.0us    71x
   512 |    8.52m   23.53m |   8548.8us |     3.8us     98%  2252x |    34.2us   250x
  1024 |   32.84m   89.34m |  32207.5us |     1.9us     98% 16635x |    20.4us  1576x
  2048 |  123.05m  354.92m | 124638.4us |     1.7us     99% 75209x |    15.3us  8137x
```

(Single run on one machine; absolute numbers vary with hardware, but the
scaling trends are the point.)

## Interpretation

**Build.** The incremental plan is ~3x slower to build than the reference — it
pays a one-time cost to construct the contact profiles and the time-overlap
sets on top of computing addresses. Both are roughly O(n^2). This penalty is
amortized away after a handful of swaps.

**Swap.** This is what the contact profiles buy:

- **Reference** rebuilds on every swap, so each swap is **O(n^2)** — 21 µs at
  n=16 growing to 125 ms at n=2048.
- **Incremental, realistic random swaps** (`rnd fast`): with localized
  lifetimes, adjacent permutation entries rarely overlap, so most swaps are
  O(1) no-ops (99% at n ≥ 1024). Average stays ~2–9 µs — up to **~75,000x
  faster**.
- **Incremental, worst case** (`ovlp fast`): forcing *every* swap onto an
  overlapping pair (full propagation) is flat in `n` (~15–34 µs across the whole
  range; the variation is noise, not growth), giving **8137x at n=2048** with a
  gap that widens without bound.

**Bottom line.** Reference swap is O(n^2); incremental swap is driven by the
contact frontier and scales with the *affected* set rather than `n`. For a local
search performing thousands of swaps, the ~3x build penalty is negligible and
the per-swap speedup is two to five orders of magnitude.

### Implementation note

`swap` maintains the position index in O(1), processes only the affected
buffers via a min-heap over positions, and propagates along the order-based
contact profiles (`above_profile`) -- dirtying the buffers directly above one
whose address changed, plus, on an in-place status flip, the buffers resting on
the affected pair at their shared tick -- rather than scanning all positions.
Candidates for re-deriving an affected buffer come from a precomputed
time-overlap set (lifetimes never change), so a swap touches no work
proportional to `n` -- only to the buffers it actually disturbs.
