# Best-first reinsertion sweep vs random rotation (co-optimizer A/B)

The co-optimizer's `reorder` move rotates a random buffer to a random position; the layout-only annealer sweeps every reinsertion position and takes them best-first. Arms are compared at **equal wall-clock**: each arm's `steps_per_buffer` grid is derived from a calibration pass so its solve times land on the incumbent's. CPU time (`process_time`) is the cost axis, so pool contention cannot flatter an arm.

## Calibration: cost per step

| graph | n | random us/step | sweep-q us/step | sweep-s us/step | sweep-q-unbi us/step | sweep probes | sweep evals | reorder accept |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| mlp | 3 | 28.4 (x1.00) | 31.1 (x1.10) | 31.1 (x1.09) | 30.5 (x1.07) | 1.7 | 1.00 | 1.00 |
| swiglu | 4 | 30.2 (x1.00) | 34.9 (x1.16) | 34.8 (x1.15) | 33.4 (x1.10) | 1.5 | 1.00 | 1.00 |
| softmax | 6 | 23.3 (x1.00) | 30.9 (x1.32) | 33.0 (x1.41) | 28.1 (x1.20) | 5.0 | 1.00 | 1.00 |
| rms_norm | 7 | 19.9 (x1.00) | 26.4 (x1.33) | 32.1 (x1.62) | 26.6 (x1.34) | 6.0 | 1.02 | 1.00 |
| sdpa | 9 | 29.8 (x1.00) | 37.9 (x1.27) | 44.6 (x1.50) | 36.4 (x1.22) | 8.0 | 1.00 | 1.00 |
| simple_attn | 9 | 37.5 (x1.00) | 45.8 (x1.22) | 53.4 (x1.43) | 44.1 (x1.18) | 7.9 | 1.00 | 1.00 |
| block_x2 | 26 | 94.0 (x1.00) | 107.0 (x1.14) | 148.0 (x1.58) | 104.7 (x1.11) | 25.0 | 1.00 | 1.00 |
| block_x3 | 39 | 97.4 (x1.00) | 107.6 (x1.11) | 161.9 (x1.66) | 105.1 (x1.08) | 38.0 | 1.00 | 1.00 |
| flash_attention | 44 | 88.4 (x1.00) | 100.3 (x1.13) | 156.5 (x1.77) | 99.5 (x1.12) | 42.8 | 1.00 | 1.00 |
| block_x4 | 52 | 96.5 (x1.00) | 108.1 (x1.12) | 178.6 (x1.85) | 104.9 (x1.09) | 51.0 | 1.00 | 1.00 |
| flash_big | 80 | 93.1 (x1.00) | 105.9 (x1.14) | 196.6 (x2.11) | 104.5 (x1.12) | 79.0 | 1.00 | 1.00 |

![frontier](../../_static/images/coopt/coopt_reorder_frontier_cap4.png)

![isotime](../../_static/images/coopt/coopt_reorder_isotime_cap4.png)

## Iso-time comparison

| graph | n | level | cpu s | random score | sweep-q % | sweep-s % | sweep-q-unbi % |
|---|--:|--:|--:|--:|--:|--:|--:|
| mlp | 3 | 40 | 0.01 | 16,492,790 | +0.00 | -- | +0.00 |
| mlp | 3 | 160 | 0.02 | 16,492,790 | +0.00 | +0.00 | +0.00 |
| mlp | 3 | 640 | 0.06 | 16,492,790 | +0.00 | +0.00 | +0.00 |
| swiglu | 4 | 40 | 0.01 | 26,990,346 | +0.00 | +0.00 | +0.00 |
| swiglu | 4 | 160 | 0.02 | 26,990,346 | +0.00 | +0.00 | +0.00 |
| swiglu | 4 | 640 | 0.09 | 26,990,346 | +0.00 | +0.00 | +0.00 |
| softmax | 6 | 40 | 0.01 | 22,719,147 | +0.00 | +0.00 | +0.00 |
| softmax | 6 | 160 | 0.03 | 22,719,147 | +0.00 | +0.00 | +0.00 |
| softmax | 6 | 640 | 0.11 | 22,719,147 | -- | +0.00 | +0.00 |
| rms_norm | 7 | 40 | 0.01 | 2,132,658 | +0.00 | -- | +0.00 |
| rms_norm | 7 | 160 | 0.04 | 2,132,658 | +0.00 | +0.00 | +0.00 |
| rms_norm | 7 | 640 | 0.11 | 2,132,658 | +0.00 | +0.00 | +0.00 |
| sdpa | 9 | 40 | 0.02 | 9,786,109 | +0.00 | +0.00 | +0.00 |
| sdpa | 9 | 160 | 0.06 | 9,786,109 | +0.00 | +0.00 | +0.00 |
| sdpa | 9 | 640 | 0.19 | 9,786,109 | +0.00 | +0.00 | +0.00 |
| simple_attn | 9 | 40 | 0.02 | 8,938,472 | +0.00 | +0.00 | +0.00 |
| simple_attn | 9 | 160 | 0.07 | 8,938,472 | +0.00 | +0.00 | +0.00 |
| simple_attn | 9 | 640 | 0.24 | 8,938,472 | +0.00 | +0.00 | +0.00 |
| block_x2 | 26 | 40 | 0.14 | 39,915,356 | +0.00 | +0.00 | +0.00 |
| block_x2 | 26 | 160 | 0.50 | 39,915,356 | +0.00 | +0.00 | +0.00 |
| block_x2 | 26 | 640 | 1.72 | 39,915,356 | +0.00 | +0.00 | +0.00 |
| block_x3 | 39 | 40 | 0.29 | 59,466,493 | +0.00 | +0.00 | +0.00 |
| block_x3 | 39 | 160 | 1.08 | 59,466,493 | +0.00 | +0.00 | +0.00 |
| block_x3 | 39 | 640 | 2.50 | 59,466,493 | +0.00 | +0.00 | +0.00 |
| flash_attention | 44 | 40 | 0.31 | 123,561,124 | +0.00 | +0.00 | +0.00 |
| flash_attention | 44 | 160 | 1.23 | 123,561,124 | +0.00 | +0.00 | +0.00 |
| flash_attention | 44 | 640 | 2.61 | 123,561,124 | +0.00 | +0.00 | +0.00 |
| block_x4 | 52 | 40 | 0.49 | 79,017,629 | +0.00 | +0.00 | +0.00 |
| block_x4 | 52 | 160 | 1.87 | 79,017,629 | +0.00 | +0.00 | +0.00 |
| block_x4 | 52 | 640 | 3.49 | 79,017,629 | +0.00 | +0.00 | +0.00 |
| flash_big | 80 | 40 | 1.10 | 463,802,281 | +0.00 | +0.00 | +0.00 |
| flash_big | 80 | 160 | 4.35 | 463,802,281 | +0.00 | +0.00 | +0.00 |
| flash_big | 80 | 640 | 5.12 | 463,802,281 | +0.00 | +0.00 | +0.00 |

## Aggregate

| arm | cells | mean % | median % | better | worse | tied |
|---|--:|--:|--:|--:|--:|--:|
| sweep-q | 32 | +0.00 | +0.00 | 0 | 0 | 32 |
| sweep-s | 31 | +0.00 | +0.00 | 0 | 0 | 31 |
| sweep-q-unbi | 33 | +0.00 | +0.00 | 0 | 0 | 33 |

_Negative % = the sweep arm reaches a better (lower) score than the incumbent at the same CPU time. Capacity = footprint//4; scores are the SA fixed-point objective. Cells where an arm's measured time range does not cover the target are left blank rather than extrapolated._
