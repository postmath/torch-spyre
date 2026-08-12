# Should `crude` be the co-optimizer's default schedule?

`crude` minus `reheating`, as a percent of `reheating`. **Negative means crude is better.** Seeds 70-89 (out-of-sample with respect to every earlier sweep in this series).

**Compared at matched wall-clock, not matched steps.** This sweep used to assume the schedules cost the same per step, which held under the memory-only objective. It does not hold under the cost objective: the two propose different move types -- crude ~50% reorder, reheating ~46% recolor -- and a recolor rewrites a region's divisions and dirties far more bundles than a reorder does. Reheating costs ~1.7x per step, so equal steps handed crude 1.7x less machine and called the difference a schedule effect. Each arm's spb is now derived from a calibration pass so both land on the same wall-clock targets, which the per-cell cpu column lets you check. Incumbent targets: steps-per-buffer [40, 160, 640] for `reheating`.

Cells where both schedules reach the same score under every seed are counted as ties, not wins.

## Headline: per capacity x reorder move

| capacity | reorder move | cells | tied | crude better | reheat better | mean % | pooled 95% CI |
|---|---|--:|--:|--:|--:|--:|---|
| footprint//2 | sweep_quality | 33 | 25 | 7 | 1 | -0.05 | [-0.20, +0.11] |
| footprint//2 | random | 33 | 26 | 6 | 1 | -0.03 | [-0.20, +0.17] |
| footprint//4 | sweep_quality | 33 | 25 | 7 | 1 | -0.02 | [-0.17, +0.13] |
| footprint//4 | random | 33 | 26 | 6 | 1 | -0.04 | [-0.20, +0.16] |

**Match quality.** Achieved crude/reheating CPU ratio across cells: mean 0.87 (min 0.53, max 1.07); 1.00 is a perfect match. 2 of 132 cells sit on the engine's `min_steps=200` floor, where both arms run the same steps whatever spb was assigned, so the cheaper arm just uses less time. Those cells are unmatched by construction and tilt the aggregate toward crude.

## Per graph (non-tied cells only)

| graph | n | capacity | move | spb rh/cr | cpu s rh/cr | reheating | crude | delta % |
|---|--:|---|---|--:|--:|--:|--:|--:|
| mlp | 3 | //2 | sweep_quality | 40/56 | 0.01/0.01 | 18,197,438 | 18,304,365 | +0.59 |
| mlp | 3 | //2 | sweep_quality | 160/224 | 0.02/0.02 | 18,021,429 | 17,998,741 | -0.13 |
| mlp | 3 | //2 | random | 40/68 | 0.01/0.01 | 18,133,736 | 18,322,208 | +1.04 |
| mlp | 3 | //2 | random | 160/273 | 0.02/0.02 | 18,043,757 | 17,982,871 | -0.34 |
| mlp | 3 | //4 | sweep_quality | 40/60 | 0.01/0.01 | 18,197,798 | 18,321,986 | +0.68 |
| mlp | 3 | //4 | sweep_quality | 160/241 | 0.02/0.02 | 18,014,611 | 18,005,559 | -0.05 |
| mlp | 3 | //4 | random | 40/69 | 0.01/0.01 | 18,140,916 | 18,304,601 | +0.90 |
| mlp | 3 | //4 | random | 160/275 | 0.02/0.02 | 18,028,248 | 17,982,871 | -0.25 |
| flash_attention | 44 | //2 | sweep_quality | 40/65 | 0.57/0.57 | 133,324,521 | 133,208,405 | -0.09 |
| flash_attention | 44 | //2 | sweep_quality | 160/261 | 2.22/2.22 | 133,135,273 | 133,107,037 | -0.02 |
| flash_attention | 44 | //2 | sweep_quality | 640/1046 | 4.75/2.88 | 133,121,155 | 133,107,037 | -0.01 |
| flash_attention | 44 | //2 | random | 40/73 | 0.57/0.55 | 133,325,918 | 133,174,445 | -0.11 |
| flash_attention | 44 | //2 | random | 160/291 | 2.21/2.17 | 133,151,932 | 133,114,096 | -0.03 |
| flash_attention | 44 | //4 | sweep_quality | 40/65 | 0.57/0.56 | 133,324,521 | 133,208,405 | -0.09 |
| flash_attention | 44 | //4 | sweep_quality | 160/260 | 2.24/2.21 | 133,135,273 | 133,107,037 | -0.02 |
| flash_attention | 44 | //4 | sweep_quality | 640/1041 | 4.73/2.88 | 133,121,155 | 133,107,037 | -0.01 |
| flash_attention | 44 | //4 | random | 40/74 | 0.56/0.55 | 133,325,918 | 133,163,509 | -0.12 |
| flash_attention | 44 | //4 | random | 160/296 | 2.27/2.22 | 133,151,932 | 133,107,037 | -0.03 |
| flash_big | 80 | //2 | sweep_quality | 40/67 | 1.97/1.98 | 492,966,030 | 491,211,715 | -0.36 |
| flash_big | 80 | //2 | sweep_quality | 160/267 | 7.85/5.53 | 491,862,292 | 490,960,047 | -0.18 |
| flash_big | 80 | //2 | sweep_quality | 640/1068 | 9.13/5.51 | 491,774,411 | 490,960,047 | -0.17 |
| flash_big | 80 | //2 | random | 40/74 | 1.98/1.92 | 492,944,040 | 491,212,890 | -0.35 |
| flash_big | 80 | //2 | random | 160/297 | 7.82/4.89 | 492,003,232 | 490,889,456 | -0.23 |
| flash_big | 80 | //2 | random | 640/1188 | 9.22/4.87 | 491,794,469 | 490,889,456 | -0.18 |
| flash_big | 80 | //4 | sweep_quality | 40/67 | 2.01/2.00 | 492,966,030 | 491,211,715 | -0.36 |
| flash_big | 80 | //4 | sweep_quality | 160/266 | 7.92/5.58 | 491,862,292 | 490,960,047 | -0.18 |
| flash_big | 80 | //4 | sweep_quality | 640/1065 | 9.29/5.53 | 491,774,411 | 490,960,047 | -0.17 |
| flash_big | 80 | //4 | random | 40/75 | 1.98/1.99 | 492,944,040 | 491,271,814 | -0.34 |
| flash_big | 80 | //4 | random | 160/301 | 7.87/4.88 | 492,003,232 | 490,889,456 | -0.23 |
| flash_big | 80 | //4 | random | 640/1205 | 9.24/4.89 | 491,794,469 | 490,889,456 | -0.18 |

## Verdict: MIXED -- see per-cell table

A global default flip is justified only if crude wins (CI strictly below zero) in **every** capacity x move combination. If it wins under the sweep but not under `random`, the honest change is a conditional default.
