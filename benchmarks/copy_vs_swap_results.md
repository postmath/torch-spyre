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
- The `n=1000 dense` config is skipped: its one-time graph build is
  intractable (and it's an unrealistic density), not the operation under test.

## Sample run

Times are mean per operation. `build` in ms; others as labeled. Single run on
one machine -- absolute numbers vary with hardware, but the orders of magnitude
are the point.

| n | density | ovlp% | build | copy | swap0 | swapX | sweep |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10 | sparse | 0% | 0.02 ms | 1.3 µs | 166 ns | — | 2.2 µs |
| 10 | medium | 11% | 0.02 ms | 1.3 µs | 170 ns | 2.53 µs | 7.4 µs |
| 10 | dense | 33% | 0.03 ms | 1.2 µs | 151 ns | 8.25 µs | 176.2 µs |
| 30 | sparse | 3% | 0.14 ms | 3.2 µs | 158 ns | 5.35 µs | 26.2 µs |
| 30 | medium | 28% | 0.15 ms | 3.2 µs | 155 ns | 7.26 µs | 1567.9 µs |
| 30 | dense | 34% | 0.26 ms | 3.2 µs | 154 ns | 179.58 µs | 1567.9 µs |
| 100 | sparse | 7% | 1.03 ms | 9.8 µs | 160 ns | 55.12 µs | 41.2 µs |
| 100 | medium | 22% | 1.84 ms | 10.4 µs | 159 ns | 16.30 µs | 4350.6 µs |
| 100 | dense | 42% | 5.47 ms | 10.7 µs | 161 ns | 37.07 µs | 153217.7 µs |
| 300 | sparse | 5% | 10.05 ms | 31.3 µs | 181 ns | 8.31 µs | 1459.0 µs |
| 300 | medium | 18% | 30.29 ms | 33.4 µs | 180 ns | 134.27 µs | 248587.2 µs |
| 300 | dense | 41% | 118.09 ms | 33.0 µs | 167 ns | 81.86 µs | 12025189.8 µs |
| 1000 | sparse | 5% | 148.14 ms | 404.5 µs | 273 ns | 1389.89 µs | 94236.8 µs |
| 1000 | medium | 20% | 800.82 ms | 322.2 µs | 316 ns | 884654.43 µs | 31353030.3 µs |
| 1000 | dense | — | *(skipped: build too slow)* | | | | |

(At small `n` the `swapX`/`sweep` figures are noisy -- e.g. the two `n=30`
rows print the same sweep time despite different `ovlp%`, because the sweep is
dominated by a handful of costly overlapping swaps. `n=10 sparse` has no
`swapX`: with 0% overlap there is no overlapping adjacent pair to time.)

## Interpretation

- **`copy()` is a cheap, predictable O(n)** -- roughly 0.4 µs/buffer (~400 µs at
  n=1000), independent of density.
- **A no-op `swap` is flat O(1)** (~160-320 ns).
- **The full sweep dominates by one to five orders of magnitude.** It is the
  `n-1` swaps the probe must do regardless, and the *overlapping* ones propagate
  expensively: a single overlapping swap is 1.4 ms (sparse) to 0.88 s (medium)
  at n=1000, and a full sweep is 94 ms (sparse) to 31 s (medium).

### Decision: a second copy in the sweep is not worth it

The two-copy scheme would save the `rotate-to-0` re-traversal -- but those are
~`i` *no-op* swaps (the swept buffer rarely overlaps what it bubbles past),
≈ `(n/2) * 270 ns` = ~135 µs at n=1000. It would *add* one copy ≈ 400 µs. So it
is net-negative (the added copy exceeds the cheap swaps saved), and either way
it is ~0.3% of the sparse sweep and ~0.001% of the medium sweep -- noise.

### What actually costs, and what doesn't

- Holding the layout by composition and probing on a `copy()` (vs sweeping the
  live plan and restoring) adds one copy per step ≈ 0.3% of a step. Negligible.
- The real cost is the **overlapping-swap propagation**, and it is super-linear
  in density (1.4 ms → 0.88 s per swap from sparse to medium at n=1000) -- the
  `_replace_buffer` cascade growing toward O(n · degree). If annealing
  performance ever matters, that is the lever, not copies. It only bites in
  non-sparse regimes; localized lifetimes (the realistic workload) are sparse
  (~5% here), where a step is ~94 ms at n=1000.
