# Sweep vs random reorder: high-power check on the graphs that move

20 seeds per cell (the main A/B used 5), capacity `footprint//2`. Graphs in scope (11): `block_x2`, `block_x3`, `block_x4`, `flash_attention`, `flash_big`, `mlp`, `rms_norm`, `sdpa`, `simple_attn`, `softmax`, `swiglu` -- derived from the main sweep as the ones whose arms differ at all, not fixed in advance, because which graphs discriminate is a property of the objective and changed when the objective did. `delta%` is the arm's mean score minus the incumbent's, as a percent of the incumbent's, with a 95% percentile-bootstrap CI. Negative = the sweep is better.

| graph | level | arm | cpu s | mean score | delta % | 95% CI |
|---|--:|---|--:|--:|--:|---|
| block_x2 | 40 | random | 0.24 | 44,029,034 | -- | -- |
| block_x2 | 40 | sweep-q | 0.21 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 40 | sweep-s | 0.21 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 40 | sweep-q-unbi | 0.21 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 160 | random | 0.81 | 44,029,034 | -- | -- |
| block_x2 | 160 | sweep-q | 0.80 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 160 | sweep-s | 0.79 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 160 | sweep-q-unbi | 0.79 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 640 | random | 2.71 | 44,029,034 | -- | -- |
| block_x2 | 640 | sweep-q | 2.73 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 640 | sweep-s | 2.83 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x2 | 640 | sweep-q-unbi | 2.70 | 44,029,034 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 40 | random | 0.45 | 65,637,010 | -- | -- |
| block_x3 | 40 | sweep-q | 0.46 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 40 | sweep-s | 0.47 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 40 | sweep-q-unbi | 0.46 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 160 | random | 1.79 | 65,637,010 | -- | -- |
| block_x3 | 160 | sweep-q | 1.78 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 160 | sweep-s | 1.81 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 160 | sweep-q-unbi | 1.81 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 640 | random | 4.20 | 65,637,010 | -- | -- |
| block_x3 | 640 | sweep-q | 4.34 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 640 | sweep-s | 4.40 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x3 | 640 | sweep-q-unbi | 4.23 | 65,637,010 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 40 | random | 0.81 | 87,244,985 | -- | -- |
| block_x4 | 40 | sweep-q | 0.81 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 40 | sweep-s | 0.82 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 40 | sweep-q-unbi | 0.81 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 160 | random | 3.20 | 87,244,985 | -- | -- |
| block_x4 | 160 | sweep-q | 3.18 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 160 | sweep-s | 3.18 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 160 | sweep-q-unbi | 3.16 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 640 | random | 5.74 | 87,244,985 | -- | -- |
| block_x4 | 640 | sweep-q | 5.86 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 640 | sweep-s | 6.09 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| block_x4 | 640 | sweep-q-unbi | 5.90 | 87,244,985 | +0.00 | [+0.00, +0.00] ns |
| flash_attention | 40 | random | 0.58 | 133,331,110 | -- | -- |
| flash_attention | 40 | sweep-q | 0.56 | 133,255,915 | -0.06 | [-0.15, +0.03] ns |
| flash_attention | 40 | sweep-s | 0.57 | 133,392,153 | +0.05 | [-0.05, +0.13] ns |
| flash_attention | 40 | sweep-q-unbi | 0.57 | 133,304,797 | -0.02 | [-0.11, +0.07] ns |
| flash_attention | 160 | random | 2.21 | 133,121,155 | -- | -- |
| flash_attention | 160 | sweep-q | 2.20 | 133,151,330 | +0.02 | [-0.01, +0.06] ns |
| flash_attention | 160 | sweep-s | 2.20 | 133,135,273 | +0.01 | [-0.01, +0.03] ns |
| flash_attention | 160 | sweep-q-unbi | 2.32 | 133,159,516 | +0.03 | [-0.01, +0.07] ns |
| flash_attention | 640 | random | 4.64 | 133,128,214 | -- | -- |
| flash_attention | 640 | sweep-q | 4.70 | 133,114,096 | -0.01 | [-0.03, +0.01] ns |
| flash_attention | 640 | sweep-s | 5.00 | 133,114,096 | -0.01 | [-0.03, +0.01] ns |
| flash_attention | 640 | sweep-q-unbi | 4.79 | 133,128,214 | +0.00 | [-0.02, +0.02] ns |
| flash_big | 40 | random | 1.96 | 493,232,877 | -- | -- |
| flash_big | 40 | sweep-q | 2.00 | 493,049,526 | -0.04 | [-0.16, +0.08] ns |
| flash_big | 40 | sweep-s | 1.98 | 492,785,910 | -0.09 | [-0.21, +0.02] ns |
| flash_big | 40 | sweep-q-unbi | 1.97 | 492,614,716 | -0.13 | [-0.23, -0.02] sig |
| flash_big | 160 | random | 7.78 | 491,683,856 | -- | -- |
| flash_big | 160 | sweep-q | 7.92 | 491,753,320 | +0.01 | [-0.03, +0.06] ns |
| flash_big | 160 | sweep-s | 7.95 | 492,127,301 | +0.09 | [+0.04, +0.14] sig |
| flash_big | 160 | sweep-q-unbi | 7.67 | 492,124,190 | +0.09 | [+0.04, +0.14] sig |
| flash_big | 640 | random | 9.10 | 491,829,852 | -- | -- |
| flash_big | 640 | sweep-q | 9.22 | 491,893,076 | +0.01 | [-0.04, +0.07] ns |
| flash_big | 640 | sweep-s | 9.65 | 491,893,076 | +0.01 | [-0.04, +0.07] ns |
| flash_big | 640 | sweep-q-unbi | 9.20 | 492,021,524 | +0.04 | [-0.02, +0.10] ns |
| mlp | 40 | random | 0.01 | 18,120,788 | -- | -- |
| mlp | 40 | sweep-q | 0.01 | 18,181,568 | +0.34 | [-0.53, +1.27] ns |
| mlp | 40 | sweep-s | 0.01 | 18,181,568 | +0.34 | [-0.53, +1.27] ns |
| mlp | 40 | sweep-q-unbi | 0.01 | 18,160,409 | +0.22 | [-0.53, +1.03] ns |
| mlp | 160 | random | 0.02 | 18,012,378 | -- | -- |
| mlp | 160 | sweep-q | 0.02 | 17,996,508 | -0.09 | [-0.38, +0.11] ns |
| mlp | 160 | sweep-s | 0.02 | 17,989,690 | -0.13 | [-0.42, +0.08] ns |
| mlp | 160 | sweep-q-unbi | 0.02 | 17,998,380 | -0.08 | [-0.42, +0.18] ns |
| mlp | 640 | random | 0.08 | 17,982,871 | -- | -- |
| mlp | 640 | sweep-q | 0.08 | 17,982,871 | +0.00 | [+0.00, +0.00] ns |
| mlp | 640 | sweep-s | 0.08 | 17,982,871 | +0.00 | [+0.00, +0.00] ns |
| mlp | 640 | sweep-q-unbi | 0.08 | 17,982,871 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 40 | random | 0.02 | 2,132,658 | -- | -- |
| rms_norm | 40 | sweep-q | 0.02 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 40 | sweep-s | 0.02 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 40 | sweep-q-unbi | 0.02 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 160 | random | 0.05 | 2,132,658 | -- | -- |
| rms_norm | 160 | sweep-q | 0.05 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 160 | sweep-s | 0.05 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 160 | sweep-q-unbi | 0.05 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 640 | random | 0.15 | 2,132,658 | -- | -- |
| rms_norm | 640 | sweep-q | 0.15 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 640 | sweep-s | 0.16 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| rms_norm | 640 | sweep-q-unbi | 0.16 | 2,132,658 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 40 | random | 0.03 | 11,669,380 | -- | -- |
| sdpa | 40 | sweep-q | 0.03 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 40 | sweep-s | 0.03 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 40 | sweep-q-unbi | 0.03 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 160 | random | 0.11 | 11,669,380 | -- | -- |
| sdpa | 160 | sweep-q | 0.11 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 160 | sweep-s | 0.11 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 160 | sweep-q-unbi | 0.11 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 640 | random | 0.37 | 11,669,380 | -- | -- |
| sdpa | 640 | sweep-q | 0.37 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 640 | sweep-s | 0.38 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| sdpa | 640 | sweep-q-unbi | 0.38 | 11,669,380 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 40 | random | 0.03 | 9,452,682 | -- | -- |
| simple_attn | 40 | sweep-q | 0.03 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 40 | sweep-s | 0.03 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 40 | sweep-q-unbi | 0.03 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 160 | random | 0.11 | 9,452,682 | -- | -- |
| simple_attn | 160 | sweep-q | 0.11 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 160 | sweep-s | 0.11 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 160 | sweep-q-unbi | 0.11 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 640 | random | 0.37 | 9,452,682 | -- | -- |
| simple_attn | 640 | sweep-q | 0.38 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 640 | sweep-s | 0.37 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| simple_attn | 640 | sweep-q-unbi | 0.37 | 9,452,682 | +0.00 | [+0.00, +0.00] ns |
| softmax | 40 | random | 0.01 | 26,127,019 | -- | -- |
| softmax | 40 | sweep-q | 0.01 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 40 | sweep-s | 0.01 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 40 | sweep-q-unbi | 0.01 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 160 | random | 0.04 | 26,127,019 | -- | -- |
| softmax | 160 | sweep-q | 0.04 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 160 | sweep-s | 0.04 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 160 | sweep-q-unbi | 0.04 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 640 | random | 0.14 | 26,127,019 | -- | -- |
| softmax | 640 | sweep-q | 0.14 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 640 | sweep-s | 0.14 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| softmax | 640 | sweep-q-unbi | 0.14 | 26,127,019 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 40 | random | 0.01 | 28,977,122 | -- | -- |
| swiglu | 40 | sweep-q | 0.01 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 40 | sweep-s | 0.01 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 40 | sweep-q-unbi | 0.01 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 160 | random | 0.03 | 28,977,122 | -- | -- |
| swiglu | 160 | sweep-q | 0.03 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 160 | sweep-s | 0.03 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 160 | sweep-q-unbi | 0.03 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 640 | random | 0.13 | 28,977,122 | -- | -- |
| swiglu | 640 | sweep-q | 0.13 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 640 | sweep-s | 0.13 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |
| swiglu | 640 | sweep-q-unbi | 0.13 | 28,977,122 | +0.00 | [+0.00, +0.00] ns |

## Verdict per arm

| arm | cells | sig better | sig worse | not significant | mean delta % |
|---|--:|--:|--:|--:|--:|
| sweep-q | 33 | 0 | 0 | 33 | +0.01 |
| sweep-s | 33 | 0 | 1 | 32 | +0.01 |
| sweep-q-unbi | 33 | 1 | 1 | 31 | +0.00 |
