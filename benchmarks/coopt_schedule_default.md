# Should `crude` be the co-optimizer's default schedule?

`crude` minus `reheating`, as a percent of `reheating`. **Negative means crude is better.** Seeds 30-49 (out-of-sample with respect to every earlier sweep in this series), steps-per-buffer [160, 640, 2560]. Schedule choice does not change per-step work, so equal steps are equal time.

Cells where both schedules reach the same score under every seed are counted as ties, not wins.

## Headline: per capacity x reorder move

| capacity | reorder move | cells | tied | crude better | reheat better | mean % | pooled 95% CI |
|---|---|--:|--:|--:|--:|--:|---|
| footprint//2 | sweep_quality | 33 | 27 | 4 | 2 | +1.58 | [+0.45, +2.74] |
| footprint//2 | random | 33 | 27 | 0 | 6 | +4.54 | [+3.21, +5.88] |
| footprint//4 | sweep_quality | 33 | 23 | 5 | 5 | -2.54 | [-4.15, -0.92] |
| footprint//4 | random | 33 | 24 | 5 | 4 | -5.70 | [-7.65, -3.75] |

## Per graph (non-tied cells only)

| graph | n | capacity | move | spb | reheating | crude | delta % |
|---|--:|---|---|--:|--:|--:|--:|
| mlp | 7 | //4 | sweep_quality | 160 | 10,240,000 | 10,368,000 | +1.25 |
| sdpa | 25 | //4 | sweep_quality | 160 | 5,024,000 | 4,672,000 | -7.01 |
| sdpa | 25 | //4 | sweep_quality | 640 | 4,320,000 | 3,904,000 | -9.63 |
| sdpa | 25 | //4 | sweep_quality | 2560 | 4,000,000 | 3,520,000 | -12.00 |
| sdpa | 25 | //4 | random | 160 | 5,504,000 | 4,480,000 | -18.60 |
| sdpa | 25 | //4 | random | 640 | 4,608,000 | 3,584,000 | -22.22 |
| sdpa | 25 | //4 | random | 2560 | 4,032,000 | 3,456,000 | -14.29 |
| flash_attention | 43 | //2 | sweep_quality | 160 | 40,941,000 | 40,654,000 | -0.70 |
| flash_attention | 43 | //2 | sweep_quality | 640 | 38,885,000 | 41,237,000 | +6.05 |
| flash_attention | 43 | //2 | sweep_quality | 2560 | 36,297,000 | 38,406,000 | +5.81 |
| flash_attention | 43 | //2 | random | 160 | 40,838,000 | 41,042,000 | +0.50 |
| flash_attention | 43 | //2 | random | 640 | 38,075,000 | 41,239,000 | +8.31 |
| flash_attention | 43 | //2 | random | 2560 | 36,806,000 | 40,220,000 | +9.28 |
| flash_attention | 43 | //4 | sweep_quality | 160 | 51,172,000 | 51,368,000 | +0.38 |
| flash_attention | 43 | //4 | sweep_quality | 640 | 49,602,000 | 50,349,000 | +1.51 |
| flash_attention | 43 | //4 | sweep_quality | 2560 | 48,918,000 | 49,923,000 | +2.05 |
| flash_attention | 43 | //4 | random | 160 | 50,698,000 | 51,384,000 | +1.35 |
| flash_attention | 43 | //4 | random | 640 | 50,585,000 | 50,869,000 | +0.56 |
| flash_attention | 43 | //4 | random | 2560 | 49,024,000 | 50,472,000 | +2.95 |
| flash_big | 79 | //2 | sweep_quality | 160 | 150,958,000 | 149,960,000 | -0.66 |
| flash_big | 79 | //2 | sweep_quality | 640 | 147,676,000 | 146,376,000 | -0.88 |
| flash_big | 79 | //2 | sweep_quality | 2560 | 143,710,000 | 143,544,000 | -0.12 |
| flash_big | 79 | //2 | random | 160 | 151,378,000 | 161,308,000 | +6.56 |
| flash_big | 79 | //2 | random | 640 | 150,166,000 | 150,642,000 | +0.32 |
| flash_big | 79 | //2 | random | 2560 | 146,370,000 | 149,662,000 | +2.25 |
| flash_big | 79 | //4 | sweep_quality | 160 | 189,320,000 | 185,614,000 | -1.96 |
| flash_big | 79 | //4 | sweep_quality | 640 | 186,320,000 | 187,356,000 | +0.56 |
| flash_big | 79 | //4 | sweep_quality | 2560 | 183,282,000 | 182,286,000 | -0.54 |
| flash_big | 79 | //4 | random | 160 | 190,950,000 | 189,412,000 | -0.81 |
| flash_big | 79 | //4 | random | 640 | 189,740,000 | 189,904,000 | +0.09 |
| flash_big | 79 | //4 | random | 2560 | 186,894,000 | 186,232,000 | -0.35 |

## Verdict: MIXED -- see per-cell table

A global default flip is justified only if crude wins (CI strictly below zero) in **every** capacity x move combination. If it wins under the sweep but not under `random`, the honest change is a conditional default.

