# Sweep vs random reorder: high-power check on the two graphs that move

20 seeds per cell (the main A/B used 5). Only `flash_attention` and `flash_big` show any arm-to-arm difference at all; the other nine captures tie exactly at every budget. `delta%` is the arm's mean score minus the incumbent's, as a percent of the incumbent's, with a 95% percentile-bootstrap CI. Negative = the sweep is better.

| graph | level | arm | cpu s | mean score | delta % | 95% CI |
|---|--:|---|--:|--:|--:|---|
| flash_attention | 160 | random | 0.32 | 40,686,000 | -- | -- |
| flash_attention | 160 | sweep-q | 0.32 | 41,136,000 | +1.11 | [-1.21, +3.53] ns |
| flash_attention | 160 | sweep-s | 0.32 | 41,236,000 | +1.35 | [-0.87, +3.78] ns |
| flash_attention | 160 | sweep-q-unbi | 0.31 | 40,862,000 | +0.43 | [-2.17, +3.14] ns |
| flash_attention | 640 | random | 1.25 | 39,674,000 | -- | -- |
| flash_attention | 640 | sweep-q | 1.27 | 38,739,000 | -2.36 | [-5.60, +0.98] ns |
| flash_attention | 640 | sweep-s | 1.25 | 39,763,000 | +0.22 | [-3.24, +3.65] ns |
| flash_attention | 640 | sweep-q-unbi | 1.26 | 39,515,000 | -0.40 | [-3.63, +2.94] ns |
| flash_attention | 2560 | random | 5.00 | 37,193,000 | -- | -- |
| flash_attention | 2560 | sweep-q | 5.05 | 36,438,000 | -2.03 | [-4.60, +0.58] ns |
| flash_attention | 2560 | sweep-s | 4.96 | 36,836,000 | -0.96 | [-3.65, +1.75] ns |
| flash_attention | 2560 | sweep-q-unbi | 4.94 | 37,110,000 | -0.22 | [-2.97, +2.59] ns |
| flash_attention | 10240 | random | 19.73 | 35,795,000 | -- | -- |
| flash_attention | 10240 | sweep-q | 19.99 | 35,559,000 | -0.66 | [-2.17, +0.86] ns |
| flash_attention | 10240 | sweep-s | 19.79 | 35,562,000 | -0.65 | [-2.16, +0.86] ns |
| flash_attention | 10240 | sweep-q-unbi | 19.02 | 36,144,000 | +0.97 | [-0.83, +2.74] ns |
| flash_big | 160 | random | 1.23 | 152,310,000 | -- | -- |
| flash_big | 160 | sweep-q | 1.20 | 151,254,000 | -0.69 | [-2.58, +1.18] ns |
| flash_big | 160 | sweep-s | 1.16 | 151,570,000 | -0.49 | [-2.16, +1.11] ns |
| flash_big | 160 | sweep-q-unbi | 1.19 | 154,142,000 | +1.20 | [-0.93, +3.43] ns |
| flash_big | 640 | random | 4.84 | 150,558,000 | -- | -- |
| flash_big | 640 | sweep-q | 4.66 | 146,770,000 | -2.52 | [-4.18, -0.93] sig |
| flash_big | 640 | sweep-s | 4.59 | 148,544,000 | -1.34 | [-2.74, +0.00] ns |
| flash_big | 640 | sweep-q-unbi | 4.70 | 149,218,000 | -0.89 | [-2.23, +0.40] ns |
| flash_big | 2560 | random | 19.13 | 147,192,000 | -- | -- |
| flash_big | 2560 | sweep-q | 18.51 | 146,036,000 | -0.79 | [-1.76, +0.16] ns |
| flash_big | 2560 | sweep-s | 18.60 | 144,408,000 | -1.89 | [-3.58, -0.42] sig |
| flash_big | 2560 | sweep-q-unbi | 18.78 | 146,496,000 | -0.47 | [-1.93, +0.73] ns |
| flash_big | 10240 | random | 76.80 | 143,156,000 | -- | -- |
| flash_big | 10240 | sweep-q | 74.16 | 141,628,000 | -1.07 | [-2.51, +0.41] ns |
| flash_big | 10240 | sweep-s | 74.77 | 142,516,000 | -0.45 | [-1.90, +1.01] ns |
| flash_big | 10240 | sweep-q-unbi | 74.29 | 143,698,000 | +0.38 | [-0.94, +1.69] ns |

## Verdict per arm

| arm | cells | sig better | sig worse | not significant | mean delta % |
|---|--:|--:|--:|--:|--:|
| sweep-q | 8 | 1 | 0 | 7 | -1.13 |
| sweep-s | 8 | 1 | 0 | 7 | -0.52 |
| sweep-q-unbi | 8 | 0 | 0 | 8 | +0.13 |
