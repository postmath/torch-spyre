# Sweep vs random reorder: high-power check on the two graphs that move

20 seeds per cell (the main A/B used 5). Only `flash_attention` and `flash_big` show any arm-to-arm difference at all; the other nine captures tie exactly at every budget. `delta%` is the arm's mean score minus the incumbent's, as a percent of the incumbent's, with a 95% percentile-bootstrap CI. Negative = the sweep is better.

| graph | level | arm | cpu s | mean score | delta % | 95% CI |
|---|--:|---|--:|--:|--:|---|
| sdpa | 160 | random | 0.19 | 5,088,000 | -- | -- |
| sdpa | 160 | sweep-q | 0.19 | 4,928,000 | -3.14 | [-12.58, +6.29] ns |
| sdpa | 160 | sweep-s | 0.21 | 5,280,000 | +3.77 | [-5.66, +13.21] ns |
| sdpa | 160 | sweep-q-unbi | 0.18 | 4,960,000 | -2.52 | [-11.95, +6.29] ns |
| sdpa | 640 | random | 0.72 | 4,800,000 | -- | -- |
| sdpa | 640 | sweep-q | 0.72 | 4,160,000 | -13.33 | [-20.00, -7.33] sig |
| sdpa | 640 | sweep-s | 0.74 | 4,160,000 | -13.33 | [-20.00, -6.67] sig |
| sdpa | 640 | sweep-q-unbi | 0.71 | 4,192,000 | -12.67 | [-19.33, -6.67] sig |
| sdpa | 2560 | random | 2.84 | 4,064,000 | -- | -- |
| sdpa | 2560 | sweep-q | 2.87 | 3,712,000 | -8.66 | [-14.96, -2.36] sig |
| sdpa | 2560 | sweep-s | 2.76 | 3,648,000 | -10.24 | [-16.54, -3.15] sig |
| sdpa | 2560 | sweep-q-unbi | 2.81 | 3,808,000 | -6.30 | [-13.39, +0.79] ns |
| sdpa | 10240 | random | 11.23 | 3,488,000 | -- | -- |
| sdpa | 10240 | sweep-q | 11.60 | 3,296,000 | -5.50 | [-10.09, -0.92] sig |
| sdpa | 10240 | sweep-s | 11.06 | 3,520,000 | +0.92 | [-5.50, +7.34] ns |
| sdpa | 10240 | sweep-q-unbi | 11.17 | 3,360,000 | -3.67 | [-9.17, +1.83] ns |
| flash_attention | 160 | random | 0.30 | 51,298,000 | -- | -- |
| flash_attention | 160 | sweep-q | 0.30 | 51,280,000 | -0.04 | [-1.20, +1.17] ns |
| flash_attention | 160 | sweep-s | 0.30 | 51,172,000 | -0.25 | [-1.34, +0.83] ns |
| flash_attention | 160 | sweep-q-unbi | 0.32 | 51,214,000 | -0.16 | [-1.56, +1.29] ns |
| flash_attention | 640 | random | 1.16 | 50,474,000 | -- | -- |
| flash_attention | 640 | sweep-q | 1.18 | 49,992,000 | -0.95 | [-2.57, +0.68] ns |
| flash_attention | 640 | sweep-s | 1.19 | 50,350,000 | -0.25 | [-1.79, +1.35] ns |
| flash_attention | 640 | sweep-q-unbi | 1.22 | 50,914,000 | +0.87 | [-0.77, +2.50] ns |
| flash_attention | 2560 | random | 4.82 | 49,032,000 | -- | -- |
| flash_attention | 2560 | sweep-q | 4.73 | 49,263,000 | +0.47 | [-1.01, +1.97] ns |
| flash_attention | 2560 | sweep-s | 4.77 | 49,142,000 | +0.22 | [-1.33, +1.84] ns |
| flash_attention | 2560 | sweep-q-unbi | 4.79 | 49,248,000 | +0.44 | [-1.25, +2.13] ns |
| flash_attention | 10240 | random | 18.59 | 48,010,000 | -- | -- |
| flash_attention | 10240 | sweep-q | 19.03 | 47,904,000 | -0.22 | [-1.77, +1.35] ns |
| flash_attention | 10240 | sweep-s | 19.06 | 48,122,000 | +0.23 | [-1.33, +1.80] ns |
| flash_attention | 10240 | sweep-q-unbi | 19.24 | 47,670,000 | -0.71 | [-2.23, +0.85] ns |
| flash_big | 160 | random | 1.22 | 193,744,000 | -- | -- |
| flash_big | 160 | sweep-q | 1.16 | 188,602,000 | -2.65 | [-4.56, -0.81] sig |
| flash_big | 160 | sweep-s | 1.16 | 191,830,000 | -0.99 | [-2.46, +0.55] ns |
| flash_big | 160 | sweep-q-unbi | 1.15 | 191,432,000 | -1.19 | [-2.64, +0.35] ns |
| flash_big | 640 | random | 4.68 | 188,272,000 | -- | -- |
| flash_big | 640 | sweep-q | 4.60 | 186,150,000 | -1.13 | [-2.71, +0.47] ns |
| flash_big | 640 | sweep-s | 4.57 | 188,188,000 | -0.04 | [-1.52, +1.41] ns |
| flash_big | 640 | sweep-q-unbi | 4.53 | 188,728,000 | +0.24 | [-1.18, +1.65] ns |
| flash_big | 2560 | random | 18.44 | 186,142,000 | -- | -- |
| flash_big | 2560 | sweep-q | 18.54 | 182,666,000 | -1.87 | [-3.16, -0.53] sig |
| flash_big | 2560 | sweep-s | 18.07 | 182,448,000 | -1.98 | [-3.22, -0.68] sig |
| flash_big | 2560 | sweep-q-unbi | 18.38 | 186,470,000 | +0.18 | [-1.08, +1.42] ns |
| flash_big | 10240 | random | 74.09 | 181,262,000 | -- | -- |
| flash_big | 10240 | sweep-q | 73.70 | 180,254,000 | -0.56 | [-1.62, +0.47] ns |
| flash_big | 10240 | sweep-s | 72.80 | 180,018,000 | -0.69 | [-1.73, +0.34] ns |
| flash_big | 10240 | sweep-q-unbi | 72.06 | 181,874,000 | +0.34 | [-0.82, +1.48] ns |

## Verdict per arm

| arm | cells | sig better | sig worse | not significant | mean delta % |
|---|--:|--:|--:|--:|--:|
| sweep-q | 12 | 5 | 0 | 7 | -3.13 |
| sweep-s | 12 | 3 | 0 | 9 | -1.88 |
| sweep-q-unbi | 12 | 1 | 0 | 11 | -2.10 |
