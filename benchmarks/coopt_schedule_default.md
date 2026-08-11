# Should `crude` be the co-optimizer's default schedule?

`crude` minus `reheating`, as a percent of `reheating`. **Negative means crude is better.** Seeds 70-89 (out-of-sample with respect to every earlier sweep in this series), steps-per-buffer [40, 160, 640]. Schedule choice does not change per-step work, so equal steps are equal time.

Cells where both schedules reach the same score under every seed are counted as ties, not wins.

## Headline: per capacity x reorder move

| capacity | reorder move | cells | tied | crude better | reheat better | mean % | pooled 95% CI |
|---|---|--:|--:|--:|--:|--:|---|
| footprint//2 | sweep_quality | 33 | 25 | 6 | 2 | +0.03 | [-0.12, +0.18] |
| footprint//2 | random | 33 | 25 | 5 | 3 | +0.14 | [-0.03, +0.33] |
| footprint//4 | sweep_quality | 33 | 25 | 6 | 2 | +0.02 | [-0.13, +0.18] |
| footprint//4 | random | 33 | 25 | 5 | 3 | +0.12 | [-0.05, +0.31] |

## Per graph (non-tied cells only)

| graph | n | capacity | move | spb | reheating | crude | delta % |
|---|--:|---|---|--:|--:|--:|--:|
| mlp | 3 | //2 | sweep_quality | 40 | 18,197,438 | 18,304,365 | +0.59 |
| mlp | 3 | //2 | sweep_quality | 160 | 18,021,429 | 18,068,678 | +0.26 |
| mlp | 3 | //2 | random | 40 | 18,133,736 | 18,315,390 | +1.00 |
| mlp | 3 | //2 | random | 160 | 18,043,757 | 18,129,925 | +0.48 |
| mlp | 3 | //4 | sweep_quality | 40 | 18,197,798 | 18,321,986 | +0.68 |
| mlp | 3 | //4 | sweep_quality | 160 | 18,014,611 | 18,037,299 | +0.13 |
| mlp | 3 | //4 | random | 40 | 18,140,916 | 18,332,939 | +1.06 |
| mlp | 3 | //4 | random | 160 | 18,028,248 | 18,107,236 | +0.44 |
| simple_attn | 9 | //2 | random | 40 | 9,452,682 | 9,481,949 | +0.31 |
| simple_attn | 9 | //4 | random | 40 | 9,452,682 | 9,467,316 | +0.15 |
| flash_attention | 44 | //2 | sweep_quality | 40 | 133,324,521 | 133,288,810 | -0.03 |
| flash_attention | 44 | //2 | sweep_quality | 160 | 133,135,273 | 133,121,155 | -0.01 |
| flash_attention | 44 | //2 | sweep_quality | 640 | 133,121,155 | 133,107,037 | -0.01 |
| flash_attention | 44 | //2 | random | 40 | 133,325,918 | 133,311,144 | -0.01 |
| flash_attention | 44 | //2 | random | 160 | 133,151,932 | 133,121,155 | -0.02 |
| flash_attention | 44 | //4 | sweep_quality | 40 | 133,324,521 | 133,288,810 | -0.03 |
| flash_attention | 44 | //4 | sweep_quality | 160 | 133,135,273 | 133,121,155 | -0.01 |
| flash_attention | 44 | //4 | sweep_quality | 640 | 133,121,155 | 133,107,037 | -0.01 |
| flash_attention | 44 | //4 | random | 40 | 133,325,918 | 133,311,144 | -0.01 |
| flash_attention | 44 | //4 | random | 160 | 133,151,932 | 133,121,155 | -0.02 |
| flash_big | 80 | //2 | sweep_quality | 40 | 492,966,030 | 491,864,857 | -0.22 |
| flash_big | 80 | //2 | sweep_quality | 160 | 491,862,292 | 490,974,165 | -0.18 |
| flash_big | 80 | //2 | sweep_quality | 640 | 491,774,411 | 490,960,047 | -0.17 |
| flash_big | 80 | //2 | random | 40 | 492,944,040 | 491,693,329 | -0.25 |
| flash_big | 80 | //2 | random | 160 | 492,003,232 | 490,931,811 | -0.22 |
| flash_big | 80 | //2 | random | 640 | 491,794,469 | 490,889,456 | -0.18 |
| flash_big | 80 | //4 | sweep_quality | 40 | 492,966,030 | 491,864,857 | -0.22 |
| flash_big | 80 | //4 | sweep_quality | 160 | 491,862,292 | 490,974,165 | -0.18 |
| flash_big | 80 | //4 | sweep_quality | 640 | 491,774,411 | 490,960,047 | -0.17 |
| flash_big | 80 | //4 | random | 40 | 492,944,040 | 491,693,329 | -0.25 |
| flash_big | 80 | //4 | random | 160 | 492,003,232 | 490,931,811 | -0.22 |
| flash_big | 80 | //4 | random | 640 | 491,794,469 | 490,889,456 | -0.18 |

## Verdict: MIXED -- see per-cell table

A global default flip is justified only if crude wins (CI strictly below zero) in **every** capacity x move combination. If it wins under the sweep but not under `random`, the honest change is a conditional default.

