# Best-first reinsertion sweep vs random rotation (co-optimizer A/B)

The co-optimizer's `reorder` move rotates a random buffer to a random position; the layout-only annealer sweeps every reinsertion position and takes them best-first. Arms are compared at **equal wall-clock**: each arm's `steps_per_buffer` grid is derived from a calibration pass so its solve times land on the incumbent's. CPU time (`process_time`) is the cost axis, so pool contention cannot flatter an arm.

## Calibration: cost per step

| graph | n | random us/step | sweep-q us/step | sweep-s us/step | sweep-q-unbi us/step | sweep probes | sweep evals | reorder accept |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| mlp | 3 | 28.4 (x1.00) | 31.3 (x1.10) | 31.2 (x1.10) | 30.3 (x1.07) | 1.7 | 1.00 | 1.00 |
| swiglu | 4 | 30.6 (x1.00) | 34.7 (x1.14) | 34.9 (x1.14) | 33.6 (x1.10) | 1.5 | 1.00 | 1.00 |
| softmax | 6 | 23.0 (x1.00) | 28.9 (x1.25) | 33.1 (x1.44) | 27.6 (x1.20) | 5.0 | 1.00 | 1.00 |
| rms_norm | 7 | 15.0 (x1.00) | 22.5 (x1.50) | 26.9 (x1.79) | 21.4 (x1.42) | 6.0 | 1.00 | 1.00 |
| sdpa | 9 | 30.6 (x1.00) | 37.3 (x1.22) | 45.1 (x1.47) | 36.8 (x1.20) | 8.0 | 1.00 | 1.00 |
| simple_attn | 9 | 39.4 (x1.00) | 46.7 (x1.18) | 52.7 (x1.34) | 44.9 (x1.14) | 7.9 | 1.00 | 1.00 |
| block_x2 | 26 | 92.7 (x1.00) | 106.0 (x1.14) | 146.2 (x1.58) | 104.2 (x1.12) | 25.0 | 1.00 | 1.00 |
| block_x3 | 39 | 96.0 (x1.00) | 107.2 (x1.12) | 161.7 (x1.68) | 105.2 (x1.10) | 38.0 | 1.00 | 1.00 |
| flash_attention | 44 | 88.5 (x1.00) | 99.3 (x1.12) | 154.8 (x1.75) | 98.9 (x1.12) | 42.8 | 1.00 | 1.00 |
| block_x4 | 52 | 95.9 (x1.00) | 106.5 (x1.11) | 173.8 (x1.81) | 104.8 (x1.09) | 51.0 | 1.00 | 1.00 |
| flash_big | 80 | 92.5 (x1.00) | 104.8 (x1.13) | 196.5 (x2.12) | 103.9 (x1.12) | 79.0 | 1.00 | 1.00 |

![frontier](../../_static/images/coopt/coopt_reorder_frontier.png)

![isotime](../../_static/images/coopt/coopt_reorder_isotime.png)

## Iso-time comparison

| graph | n | level | cpu s | random score | sweep-q % | sweep-s % | sweep-q-unbi % |
|---|--:|--:|--:|--:|--:|--:|--:|
| mlp | 3 | 40 | 0.01 | 16,492,790 | +0.00 | +0.00 | +0.00 |
| mlp | 3 | 160 | 0.02 | 16,492,790 | +0.00 | +0.00 | +0.00 |
| mlp | 3 | 640 | 0.06 | 16,492,790 | +0.00 | +0.00 | +0.00 |
| swiglu | 4 | 40 | 0.01 | 26,990,346 | +0.00 | +0.00 | +0.00 |
| swiglu | 4 | 160 | 0.02 | 26,990,346 | +0.00 | +0.00 | +0.00 |
| swiglu | 4 | 640 | 0.09 | 26,990,346 | +0.00 | +0.00 | +0.00 |
| softmax | 6 | 40 | 0.01 | 22,719,147 | +0.00 | +0.00 | +0.00 |
| softmax | 6 | 160 | 0.03 | 22,719,147 | +0.00 | +0.00 | +0.00 |
| softmax | 6 | 640 | 0.09 | 22,719,147 | +0.00 | +0.00 | +0.00 |
| rms_norm | 7 | 40 | 0.01 | 2,132,658 | +0.00 | +0.00 | +0.00 |
| rms_norm | 7 | 160 | 0.03 | 2,132,658 | +0.00 | +0.00 | +0.00 |
| rms_norm | 7 | 640 | 0.09 | 2,132,658 | +0.00 | +0.00 | +0.00 |
| sdpa | 9 | 40 | 0.02 | 9,786,109 | +0.00 | +0.00 | +0.00 |
| sdpa | 9 | 160 | 0.06 | 9,786,109 | +0.00 | +0.00 | +0.00 |
| sdpa | 9 | 640 | 0.20 | 9,786,109 | +0.00 | +0.00 | +0.00 |
| simple_attn | 9 | 40 | 0.02 | 8,938,472 | +0.00 | +0.00 | +0.00 |
| simple_attn | 9 | 160 | 0.07 | 8,938,472 | +0.00 | +0.00 | +0.00 |
| simple_attn | 9 | 640 | 0.24 | 8,938,472 | +0.00 | +0.00 | +0.00 |
| block_x2 | 26 | 40 | 0.14 | 39,915,356 | +0.00 | +0.00 | +0.00 |
| block_x2 | 26 | 160 | 0.51 | 39,915,356 | +0.00 | +0.00 | +0.00 |
| block_x2 | 26 | 640 | 1.81 | 39,915,356 | +0.00 | +0.00 | +0.00 |
| block_x3 | 39 | 40 | 0.29 | 59,466,493 | -- | +0.00 | +0.00 |
| block_x3 | 39 | 160 | 1.09 | 59,466,493 | +0.00 | +0.00 | +0.00 |
| block_x3 | 39 | 640 | 2.55 | 59,466,493 | +0.00 | +0.00 | +0.00 |
| flash_attention | 44 | 40 | 0.32 | 123,561,124 | +0.00 | +0.00 | +0.00 |
| flash_attention | 44 | 160 | 1.27 | 123,561,124 | +0.00 | +0.00 | +0.00 |
| flash_attention | 44 | 640 | 2.70 | 123,561,124 | +0.00 | +0.00 | +0.00 |
| block_x4 | 52 | 40 | 0.50 | 79,017,629 | +0.00 | +0.00 | +0.00 |
| block_x4 | 52 | 160 | 1.93 | 79,017,629 | +0.00 | +0.00 | +0.00 |
| block_x4 | 52 | 640 | 3.40 | 79,017,629 | +0.00 | +0.00 | +0.00 |
| flash_big | 80 | 40 | 1.11 | 463,802,281 | +0.00 | +0.00 | +0.00 |
| flash_big | 80 | 160 | 4.37 | 463,802,281 | +0.00 | +0.00 | +0.00 |
| flash_big | 80 | 640 | 5.20 | 463,802,281 | +0.00 | +0.00 | +0.00 |

## Aggregate

| arm | cells | mean % | median % | better | worse | tied |
|---|--:|--:|--:|--:|--:|--:|
| sweep-q | 32 | +0.00 | +0.00 | 0 | 0 | 32 |
| sweep-s | 33 | +0.00 | +0.00 | 0 | 0 | 33 |
| sweep-q-unbi | 33 | +0.00 | +0.00 | 0 | 0 | 33 |

_Negative % = the sweep arm reaches a better (lower) score than the incumbent at the same CPU time. Capacity = footprint//2; scores are the SA fixed-point objective. Cells where an arm's measured time range does not cover the target are left blank rather than extrapolated._
