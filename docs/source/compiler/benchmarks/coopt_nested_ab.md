# Nested two-timescale SA vs the single-loop engine (A/B)

Nested variants (greedy/annealed inner x constant/linear/convex/adaptive length curve) vs crude and the incumbent `reheat`, over run length, 5 seeds, capacity `footprint//2`. `delta%` and the frontier are both vs `reheat`. Wall-clock is recorded because the nested inner loop skips the per-step rescore, so quality-vs-time is the honest efficiency axis.

## Headline finding

**The nested engine's win is wall-clock, not plan quality. Read both numbers together or this table will mislead you.** At each graph's longest run length the best nested config scores +0.02% against `reheat` -- a tie, within noise, *not* an improvement -- while taking 6.6x less time (3.3x-26.4x). It ties or beats on 9 of 11 graphs. Every "speedup" below is therefore the price of the same answer, never a better one; the largest are on the largest graphs, where the incumbent's per-step full rescore is most expensive.

That framing matters because of what the extra budget itself buys, which is almost nothing: `reheat` returns a bit-identical score at *every* run length on 9 of 11 graphs, so the search has already converged at the default budget. The time the nested engine saves is therefore time spent on steps that do not change the answer. A speedup on a budget nobody needs is not a reason to adopt it.

Where the win comes from:

- **Skipping the per-step full rescore.** The incumbent scores the whole state
  every step; the nested inner layout loop drives the packer's incremental
  quality and computes the full score once per outer (structural) move. Under the
  cost-model objective a full score is far more expensive than it was under the
  memory-only one, so this advantage is *larger* here than when the nested engine
  was first measured.
- **Warm-started, rotate-based inner loops.** Layout re-adapts to each structural
  change from the persisted permutation, using single-buffer reinsertions (fast
  mixing), not adjacent swaps.

Where the incumbent still wins: `flash_big` +0.16%; `flash_attention` +0.03%.

### At the default step budget the result reverses

The grid above starts at `steps_per_buffer=40`, the engine's default and the only point production runs at. There the nested engine is **behind on 4 of 11 graphs and ahead on none**, by +1.26% on average -- worst `flash_big` +6.45%, `flash_attention` +3.70%, `mlp` +3.05%. It is still 3.9x cheaper in solver time, but the absolute saving is fractions of a second per graph.

The nested engine needs budget to amortize: each outer structural move spends a whole inner layout loop, so a small total budget buys few structural evaluations. Its advantage therefore appears only well above the operating point, and the equal-steps framing of the table above is what makes it look unconditional. **This is the cell to read before making nested a default.**

Secondary findings:

- **Greedy vs annealed inner loop:** the annealed inner loop is worse by +0.20% on average, and 0.00% on the graphs where every config converges to the same score. Greedy-cold remains the robust choice.
- **The inner-length curve barely separates the greedy configs** (`constant` +0.02%, `convex` +0.23%, `linear` +0.52%, `adaptive` +1.15%), so the simplest `constant` inner length is a fine default. The "grow the inner loop over the run" hypothesis is still not strongly supported -- warm-start and rescore-skipping carry the win, not the length schedule.

_Caveats: capacity = footprint//2; y is the cost model's fixed-point prediction, not measured hardware time (the seconds here are solver compute); flash_big capped at spb 2560. Several graphs converge to an identical score across every config and run length, so their +0.00% is a genuine tie, not a rounding artefact._

![delta](../../_static/images/coopt/coopt_nested_delta.png)

![frontier](../../_static/images/coopt/coopt_nested_frontier.png)

## Best nested config vs incumbent, per (graph, run length)

| graph | n | spb | reheat score | reheat s | best nested | nested % | nested s | speedup |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| mlp | 3 | 40 | 17,982,871 | 0.01 | nest-greedy-adaptive | +3.05 | 0.00 | 2.59x |
| mlp | 3 | 160 | 17,982,871 | 0.04 | nest-greedy-constant | +0.50 | 0.01 | 3.84x |
| mlp | 3 | 640 | 17,982,871 | 0.26 | nest-greedy-constant | +0.00 | 0.10 | 2.58x |
| mlp | 3 | 2560 | 17,982,871 | 0.87 | nest-greedy-constant | +0.00 | 0.34 | 2.55x |
| mlp | 3 | 10240 | 17,982,871 | 1.53 | nest-greedy-linear | +0.00 | 0.46 | 3.33x |
| swiglu | 4 | 40 | 28,977,122 | 0.05 | nest-greedy-constant | +0.00 | 0.01 | 9.68x |
| swiglu | 4 | 160 | 28,977,122 | 0.12 | nest-greedy-constant | +0.00 | 0.07 | 1.77x |
| swiglu | 4 | 640 | 28,977,122 | 0.41 | nest-greedy-linear | +0.00 | 0.07 | 5.90x |
| swiglu | 4 | 2560 | 28,977,122 | 1.56 | nest-greedy-linear | +0.00 | 0.33 | 4.68x |
| swiglu | 4 | 10240 | 28,977,122 | 1.78 | nest-greedy-linear | +0.00 | 0.48 | 3.71x |
| softmax | 6 | 40 | 26,127,019 | 0.01 | nest-greedy-constant | +0.00 | 0.05 | 0.27x |
| softmax | 6 | 160 | 26,127,019 | 0.16 | nest-greedy-constant | +0.00 | 0.05 | 3.43x |
| softmax | 6 | 640 | 26,127,019 | 0.45 | nest-greedy-linear | +0.00 | 0.12 | 3.83x |
| softmax | 6 | 2560 | 26,127,019 | 1.43 | nest-greedy-linear | +0.00 | 0.29 | 4.87x |
| rms_norm | 7 | 40 | 2,132,658 | 0.02 | nest-greedy-constant | +0.00 | 0.01 | 2.06x |
| rms_norm | 7 | 160 | 2,132,658 | 0.12 | nest-greedy-constant | +0.00 | 0.04 | 3.32x |
| rms_norm | 7 | 640 | 2,132,658 | 0.50 | nest-greedy-linear | +0.00 | 0.12 | 4.04x |
| rms_norm | 7 | 2560 | 2,132,658 | 1.41 | nest-greedy-linear | +0.00 | 0.21 | 6.59x |
| sdpa | 9 | 40 | 11,669,380 | 0.10 | nest-greedy-constant | +0.00 | 0.04 | 2.34x |
| sdpa | 9 | 160 | 11,669,380 | 0.38 | nest-greedy-constant | +0.00 | 0.11 | 3.38x |
| sdpa | 9 | 640 | 11,669,380 | 1.15 | nest-greedy-linear | +0.00 | 0.20 | 5.70x |
| sdpa | 9 | 2560 | 11,669,380 | 3.04 | nest-greedy-linear | +0.00 | 0.51 | 5.93x |
| simple_attn | 9 | 40 | 9,452,682 | 0.10 | nest-greedy-constant | +0.62 | 0.03 | 3.95x |
| simple_attn | 9 | 160 | 9,452,682 | 0.37 | nest-greedy-constant | +0.00 | 0.12 | 3.05x |
| simple_attn | 9 | 640 | 9,452,682 | 1.14 | nest-greedy-linear | +0.00 | 0.23 | 4.95x |
| simple_attn | 9 | 2560 | 9,452,682 | 2.67 | nest-greedy-linear | +0.00 | 0.42 | 6.42x |
| block_x2 | 26 | 40 | 44,029,034 | 0.70 | nest-greedy-constant | +0.00 | 0.19 | 3.74x |
| block_x2 | 26 | 160 | 44,029,034 | 2.57 | nest-greedy-linear | +0.00 | 0.19 | 13.49x |
| block_x2 | 26 | 640 | 44,029,034 | 9.27 | nest-greedy-linear | +0.00 | 0.63 | 14.70x |
| block_x3 | 39 | 40 | 65,637,010 | 1.93 | nest-greedy-linear | +0.00 | 0.14 | 13.77x |
| block_x3 | 39 | 160 | 65,637,010 | 6.06 | nest-greedy-convex | +0.00 | 0.46 | 13.08x |
| block_x3 | 39 | 640 | 65,637,010 | 15.95 | nest-greedy-linear | +0.00 | 0.60 | 26.37x |
| flash_attention | 44 | 40 | 133,358,813 | 1.83 | nest-greedy-constant | +3.70 | 0.28 | 6.61x |
| flash_attention | 44 | 160 | 133,135,273 | 8.61 | nest-greedy-constant | +0.50 | 0.86 | 10.01x |
| flash_attention | 44 | 640 | 133,135,273 | 17.65 | nest-greedy-constant | +0.03 | 1.82 | 9.70x |
| block_x4 | 52 | 40 | 87,244,985 | 2.62 | nest-greedy-constant | +0.00 | 0.47 | 5.55x |
| block_x4 | 52 | 160 | 87,244,985 | 10.38 | nest-greedy-linear | +0.00 | 0.31 | 33.92x |
| block_x4 | 52 | 640 | 87,244,985 | 18.70 | nest-greedy-linear | +0.00 | 0.93 | 20.08x |
| flash_big | 80 | 40 | 492,836,182 | 7.58 | nest-greedy-constant | +6.45 | 0.42 | 17.95x |
| flash_big | 80 | 160 | 492,052,415 | 21.09 | nest-greedy-constant | +0.29 | 1.80 | 11.71x |
| flash_big | 80 | 640 | 491,876,189 | 19.87 | nest-greedy-constant | +0.16 | 2.08 | 9.55x |
