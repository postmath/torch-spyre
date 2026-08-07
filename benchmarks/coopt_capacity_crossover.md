# Where does `crude` overtake `reheating`?

`crude` minus `reheating` as a percent of `reheating`; **negative means crude is better**. Capacity swept as a fraction of the seed footprint (0.5 is the old `//2`, 0.25 the old `//4`), steps-per-buffer 640, seeds 50-69 (out-of-sample against every earlier run). Both reorder moves swept, so the threshold does not depend on the current default move.

![crossover](results/coopt_capacity_crossover.png)


## reorder_move = `sweep_quality`

| capacity / footprint | spill fraction | non-tied cells | delta % | 95% CI |
|--:|--:|--:|--:|---|
| 0.80 | 0.22 | 1 | -0.70 **sig** | [-0.96, -0.44] |
| 0.60 | 0.33 | 2 | -1.57 **sig** | [-2.91, -0.19] |
| 0.50 | 0.33 | 2 | +0.63 | [-1.43, +2.65] |
| 0.42 | 0.34 | 2 | +0.29 | [-1.12, +1.80] |
| 0.35 | 0.34 | 2 | +2.22 **sig** | [+0.29, +4.33] |
| 0.30 | 0.34 | 3 | +0.34 | [-3.85, +3.40] |
| 0.25 | 0.34 | 3 | -3.83 **sig** | [-6.17, -1.54] |
| 0.20 | 0.34 | 2 | -1.23 | [-4.90, +2.50] |
| 0.15 | 0.40 | 3 | -1.81 **sig** | [-2.54, -1.14] |
| 0.10 | 0.47 | 2 | -3.94 **sig** | [-4.77, -3.11] |

Zero crossing (linear interpolation): capacity ratio **0.296**, spill fraction **0.34**.


## reorder_move = `random`

| capacity / footprint | spill fraction | non-tied cells | delta % | 95% CI |
|--:|--:|--:|--:|---|
| 0.80 | 0.22 | 2 | -0.43 **sig** | [-0.62, -0.25] |
| 0.60 | 0.32 | 2 | +1.07 | [-1.20, +3.46] |
| 0.50 | 0.33 | 3 | +3.98 **sig** | [+2.30, +5.82] |
| 0.42 | 0.33 | 2 | +0.06 | [-1.40, +1.58] |
| 0.35 | 0.33 | 2 | +3.28 **sig** | [+1.33, +5.34] |
| 0.30 | 0.34 | 3 | -3.86 | [-9.80, +1.45] |
| 0.25 | 0.34 | 3 | -8.95 **sig** | [-12.92, -5.23] |
| 0.20 | 0.35 | 2 | -10.13 **sig** | [-13.78, -6.30] |
| 0.15 | 0.39 | 3 | -1.24 **sig** | [-1.88, -0.60] |
| 0.10 | 0.47 | 3 | -2.42 **sig** | [-3.22, -1.64] |

Zero crossing (linear interpolation): capacity ratio **0.327**, spill fraction **0.34**.

