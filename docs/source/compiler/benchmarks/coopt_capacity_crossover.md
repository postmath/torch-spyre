# Where does `crude` overtake `reheating`?

`crude` minus `reheating` as a percent of `reheating`; **negative means crude is better**. Capacity swept as a fraction of the seed footprint (0.5 is the old `//2`, 0.25 the old `//4`), steps-per-buffer 640, seeds 70-89 (out-of-sample against every earlier run). Both reorder moves swept, so the threshold does not depend on the current default move.

![crossover](../../_static/images/coopt/coopt_capacity_crossover.png)

## reorder_move = `sweep_quality`

| capacity / footprint | spill fraction | non-tied cells | delta % | 95% CI |
|--:|--:|--:|--:|---|

No sign change over the swept range.

## reorder_move = `random`

| capacity / footprint | spill fraction | non-tied cells | delta % | 95% CI |
|--:|--:|--:|--:|---|

No sign change over the swept range.
