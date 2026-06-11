# copy() vs swap() vs full sweep

Cost comparison for the operations the Imanishi/Xu reinsertion sweep is built
from, to decide whether that sweep should keep a second copy of the plan (to
avoid re-traversing) -- worth it only if a copy is cheaper than the swaps it
would save.

Reproduce with:

```bash
python benchmarks/profile_copy_vs_swap.py
```

## Workload

- Half-open lifetimes `[start, end)` over a horizon of `n` ticks; lifetime
  length is `uniform(1, span_frac * n)`, so `span_frac` controls how many
  buffers are simultaneously live. Three densities: sparse (0.05), medium
  (0.2), dense (0.5).
- Capacity is effectively unbounded (10^9), so every buffer is placed.
- `ovlp%` is the measured fraction of adjacent permutation pairs that overlap
  in time (i.e. the fraction of sweep swaps that are *not* O(1) no-ops).
- `swap0` = a no-op swap (non-overlapping adjacent pair); `swapX` = an
  overlapping swap (full propagation); `sweep` = one full single-element sweep
  (`rotate(n//2, 0)` then `n-1` forward swaps).
- The `n=1000 dense` config is skipped by a conservative build-cost guard in
  the script. (The guard predates the contact-profile rewrite, which made
  `_build` far cheaper; the sweep itself would still be expensive at that size
  and density, which is unrealistic, so the guard is left in place.)

## Sample run

Times are mean per operation. `build` in ms; others as labeled. Single run on
one machine -- absolute numbers vary with hardware, but the orders of magnitude
are the point.

| n | density | ovlp% | build | copy | swap0 | swapX | sweep |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10 | sparse | 0% | 0.04 ms | 3.4 µs | 195 ns | — | 2.8 µs |
| 10 | medium | 11% | 0.03 ms | 3.2 µs | 182 ns | 7.31 µs | 17.5 µs |
| 10 | dense | 33% | 0.04 ms | 3.3 µs | 181 ns | 11.63 µs | 270.8 µs |
| 30 | sparse | 3% | 0.13 ms | 10.4 µs | 304 ns | 16.03 µs | 46.0 µs |
| 30 | medium | 28% | 0.17 ms | 9.1 µs | 193 ns | 9.99 µs | 72.7 µs |
| 30 | dense | 34% | 0.20 ms | 9.2 µs | 184 ns | 52.03 µs | 558.3 µs |
| 100 | sparse | 7% | 1.09 ms | 29.0 µs | 197 ns | 35.23 µs | 63.4 µs |
| 100 | medium | 22% | 1.41 ms | 30.0 µs | 197 ns | 14.65 µs | 1232.0 µs |
| 100 | dense | 42% | 1.99 ms | 30.0 µs | 197 ns | 15.98 µs | 12040.0 µs |
| 300 | sparse | 5% | 8.58 ms | 133.3 µs | 208 ns | 15.24 µs | 602.0 µs |
| 300 | medium | 18% | 12.28 ms | 135.5 µs | 213 ns | 25.87 µs | 15222.4 µs |
| 300 | dense | 41% | 17.12 ms | 138.3 µs | 214 ns | 22.59 µs | 276940.6 µs |
| 1000 | sparse | 5% | 100.26 ms | 566.3 µs | 334 ns | 162.04 µs | 9381.5 µs |
| 1000 | medium | 20% | 154.11 ms | 577.8 µs | 286 ns | 18211.27 µs | 599230.6 µs |
| 1000 | dense | — | *(skipped: build guard)* | | | | |

(At small `n` the `swapX`/`sweep` figures are noisy. `n=10 sparse` has no
`swapX`: with 0% overlap there is no overlapping adjacent pair to time.)

## Interpretation

- **`copy()` is a cheap, predictable O(n)** -- roughly 0.57 µs/buffer (~570 µs at
  n=1000), independent of density.
- **A no-op `swap` is flat O(1)** (~180-330 ns).
- **The full sweep is the dominant per-step cost, and it scales with overlap
  density.** It is the `n-1` swaps the probe must do regardless, and the
  *overlapping* ones carry the propagation: a single overlapping swap is 162 µs
  (sparse) to 18 ms (medium) at n=1000, and a full sweep is 9.4 ms (sparse) to
  0.6 s (medium).

### Decision: a second copy in the sweep is not worth it

The two-copy scheme would save the `rotate-to-0` re-traversal -- but those are
~`i` *no-op* swaps (the swept buffer rarely overlaps what it bubbles past),
≈ `(n/2) * ~300 ns` = ~150 µs at n=1000. It would *add* one copy ≈ 570 µs. So it
is net-negative (the added copy exceeds the cheap swaps saved), and either way
it is a small fraction of the sparse sweep and noise against the medium sweep.

### What actually costs, and what doesn't

- Holding the layout by composition and probing on a `copy()` (vs sweeping the
  live plan and restoring) adds one copy per step ≈ a few percent of a step.
  Negligible.
- The real cost is now **`_placement_decision`** (re-placing an address from its
  candidates), driven by the overlap density: each affected buffer rescans its
  earlier-overlapping candidates and probes in-place partners. It only bites in
  non-sparse regimes; localized lifetimes (the realistic workload) are sparse
  (~5% here), where a full sweep is ~9 ms at n=1000.

## History: the `_replace_buffer` cascade is gone

These figures are from the contact-profile solver: a swap transposes the
order-based profiles by O(segments) splices, then re-places addresses along the
contact frontier (order-above neighbours plus an in-place-status-transition
rule). The earlier solver propagated through a `_replace_buffer` cascade that
grew toward O(n · degree) and blew up super-linearly with density. Replacing it
collapsed the non-sparse cells by one to two orders of magnitude (same machine,
same workload):

| cell | metric | before (cascade) | after (profiles) | speedup |
|---:|:---|---:|---:|---:|
| n=300 dense | sweep | 12,025 ms | 277 ms | ~43× |
| n=1000 medium | sweep | 31,353 ms | 599 ms | ~52× |
| n=1000 medium | swapX | 884.7 ms | 18.2 ms | ~49× |
| n=100 dense | sweep | 153 ms | 12 ms | ~13× |
| n=1000 medium | build | 800.8 ms | 154 ms | ~5× |
| n=300 dense | build | 118 ms | 17 ms | ~7× |

`_build` also dropped because it now constructs contact profiles instead of the
old O(ticks · alive²) neighbour graph. The sweep still exceeds a single build in
dense regimes, but in the realistic sparse regime a full `n-1`-swap sweep is
~10× cheaper than one rebuild (9.4 ms vs 100 ms at n=1000) -- so sweeping rather
than rebuilding remains the right call.
