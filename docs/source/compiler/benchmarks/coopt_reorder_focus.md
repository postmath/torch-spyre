# Sweep vs random reorder: high-power check on the graphs that move

20 seeds per cell (the main A/B used 5), capacity `footprint//2`. Graphs in scope (11): `block_x2`, `block_x3`, `block_x4`, `flash_attention`, `flash_big`, `mlp`, `rms_norm`, `sdpa`, `simple_attn`, `softmax`, `swiglu` -- derived from the main sweep as the ones whose arms differ at all, not fixed in advance, because which graphs discriminate is a property of the objective and changed when the objective did. `delta%` is the arm's mean score minus the incumbent's, as a percent of the incumbent's, with a 95% percentile-bootstrap CI. Negative = the sweep is better.

| graph | level | arm | cpu s | mean score | delta % | 95% CI |
|---|--:|---|--:|--:|--:|---|
| block_x2 | 40 | random | 0.14 | 39,915,356 | -- | -- |
| block_x2 | 40 | sweep-q | 0.13 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 40 | sweep-s | 0.13 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 40 | sweep-q-unbi | 0.13 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 160 | random | 0.50 | 39,915,356 | -- | -- |
| block_x2 | 160 | sweep-q | 0.49 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 160 | sweep-s | 0.49 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 160 | sweep-q-unbi | 0.50 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 640 | random | 1.69 | 39,915,356 | -- | -- |
| block_x2 | 640 | sweep-q | 1.87 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 640 | sweep-s | 1.88 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 640 | sweep-q-unbi | 1.89 | 39,915,356 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 40 | random | 0.29 | 59,466,493 | -- | -- |
| block_x3 | 40 | sweep-q | 0.29 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 40 | sweep-s | 0.29 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 40 | sweep-q-unbi | 0.29 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 160 | random | 1.12 | 59,466,493 | -- | -- |
| block_x3 | 160 | sweep-q | 1.13 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 160 | sweep-s | 1.09 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 160 | sweep-q-unbi | 1.10 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 640 | random | 2.66 | 59,466,493 | -- | -- |
| block_x3 | 640 | sweep-q | 2.96 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 640 | sweep-s | 4.28 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 640 | sweep-q-unbi | 2.81 | 59,466,493 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 40 | random | 0.49 | 79,017,629 | -- | -- |
| block_x4 | 40 | sweep-q | 0.49 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 40 | sweep-s | 0.50 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 40 | sweep-q-unbi | 0.51 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 160 | random | 1.91 | 79,017,629 | -- | -- |
| block_x4 | 160 | sweep-q | 1.92 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 160 | sweep-s | 1.92 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 160 | sweep-q-unbi | 1.94 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 640 | random | 3.48 | 79,017,629 | -- | -- |
| block_x4 | 640 | sweep-q | 3.83 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 640 | sweep-s | 6.26 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 640 | sweep-q-unbi | 3.80 | 79,017,629 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 40 | random | 0.33 | 123,561,124 | -- | -- |
| flash_attention | 40 | sweep-q | 0.33 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 40 | sweep-s | 0.33 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 40 | sweep-q-unbi | 0.33 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 160 | random | 1.28 | 123,561,124 | -- | -- |
| flash_attention | 160 | sweep-q | 1.29 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 160 | sweep-s | 1.27 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 160 | sweep-q-unbi | 1.28 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 640 | random | 2.70 | 123,561,124 | -- | -- |
| flash_attention | 640 | sweep-q | 3.04 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 640 | sweep-s | 4.73 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 640 | sweep-q-unbi | 3.03 | 123,561,124 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 40 | random | 1.14 | 463,802,281 | -- | -- |
| flash_big | 40 | sweep-q | 1.11 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 40 | sweep-s | 1.13 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 40 | sweep-q-unbi | 1.15 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 160 | random | 4.47 | 463,802,281 | -- | -- |
| flash_big | 160 | sweep-q | 4.44 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 160 | sweep-s | 4.42 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 160 | sweep-q-unbi | 4.38 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 640 | random | 5.19 | 463,802,281 | -- | -- |
| flash_big | 640 | sweep-q | 5.90 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 640 | sweep-s | 11.11 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| flash_big | 640 | sweep-q-unbi | 5.85 | 463,802,281 | +0.00 | [+0.00, +0.00] ns |
| mlp | 40 | random | 0.01 | 16,492,790 | -- | -- |
| mlp | 40 | sweep-q | 0.01 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 40 | sweep-s | 0.01 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 40 | sweep-q-unbi | 0.01 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 160 | random | 0.02 | 16,492,790 | -- | -- |
| mlp | 160 | sweep-q | 0.02 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 160 | sweep-s | 0.02 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 160 | sweep-q-unbi | 0.02 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 640 | random | 0.06 | 16,492,790 | -- | -- |
| mlp | 640 | sweep-q | 0.06 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 640 | sweep-s | 0.06 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| mlp | 640 | sweep-q-unbi | 0.06 | 16,492,790 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 40 | random | 0.01 | 2,132,658 | -- | -- |
| rms_norm | 40 | sweep-q | 0.01 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 40 | sweep-s | 0.01 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 40 | sweep-q-unbi | 0.01 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 160 | random | 0.03 | 2,132,658 | -- | -- |
| rms_norm | 160 | sweep-q | 0.03 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 160 | sweep-s | 0.03 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 160 | sweep-q-unbi | 0.03 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 640 | random | 0.09 | 2,132,658 | -- | -- |
| rms_norm | 640 | sweep-q | 0.09 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 640 | sweep-s | 0.09 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 640 | sweep-q-unbi | 0.09 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 40 | random | 0.02 | 9,786,109 | -- | -- |
| sdpa | 40 | sweep-q | 0.02 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 40 | sweep-s | 0.02 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 40 | sweep-q-unbi | 0.02 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 160 | random | 0.07 | 9,786,109 | -- | -- |
| sdpa | 160 | sweep-q | 0.06 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 160 | sweep-s | 0.06 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 160 | sweep-q-unbi | 0.06 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 640 | random | 0.20 | 9,786,109 | -- | -- |
| sdpa | 640 | sweep-q | 0.21 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 640 | sweep-s | 0.20 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 640 | sweep-q-unbi | 0.20 | 9,786,109 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 40 | random | 0.02 | 8,938,472 | -- | -- |
| simple_attn | 40 | sweep-q | 0.02 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 40 | sweep-s | 0.02 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 40 | sweep-q-unbi | 0.02 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 160 | random | 0.07 | 8,938,472 | -- | -- |
| simple_attn | 160 | sweep-q | 0.07 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 160 | sweep-s | 0.07 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 160 | sweep-q-unbi | 0.07 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 640 | random | 0.24 | 8,938,472 | -- | -- |
| simple_attn | 640 | sweep-q | 0.25 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 640 | sweep-s | 0.25 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 640 | sweep-q-unbi | 0.25 | 8,938,472 | +0.00 | [+0.00, +0.00] ns |
| softmax | 40 | random | 0.01 | 22,719,147 | -- | -- |
| softmax | 40 | sweep-q | 0.01 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 40 | sweep-s | 0.01 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 40 | sweep-q-unbi | 0.01 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 160 | random | 0.03 | 22,719,147 | -- | -- |
| softmax | 160 | sweep-q | 0.03 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 160 | sweep-s | 0.03 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 160 | sweep-q-unbi | 0.03 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 640 | random | 0.10 | 22,719,147 | -- | -- |
| softmax | 640 | sweep-q | 0.10 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 640 | sweep-s | 0.10 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| softmax | 640 | sweep-q-unbi | 0.10 | 22,719,147 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 40 | random | 0.01 | 26,990,346 | -- | -- |
| swiglu | 40 | sweep-q | 0.01 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 40 | sweep-s | 0.01 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 40 | sweep-q-unbi | 0.01 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 160 | random | 0.02 | 26,990,346 | -- | -- |
| swiglu | 160 | sweep-q | 0.02 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 160 | sweep-s | 0.02 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 160 | sweep-q-unbi | 0.02 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 640 | random | 0.08 | 26,990,346 | -- | -- |
| swiglu | 640 | sweep-q | 0.08 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 640 | sweep-s | 0.08 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 640 | sweep-q-unbi | 0.08 | 26,990,346 | +0.00 | [+0.00, +0.00] ns |

## Verdict per arm

| arm | cells | sig better | sig worse | not significant | mean delta % |
|---|--:|--:|--:|--:|--:|
| sweep-q | 33 | 0 | 0 | 33 | +0.00 |
| sweep-s | 33 | 0 | 0 | 33 | +0.00 |
| sweep-q-unbi | 33 | 0 | 0 | 33 | +0.00 |
