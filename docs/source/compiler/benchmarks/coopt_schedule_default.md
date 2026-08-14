# Should `crude` be the co-optimizer's default schedule?

`crude` minus `reheating`, as a percent of `reheating`. **Negative means crude is better.** Seeds 70-89 (out-of-sample with respect to every earlier sweep in this series).

**Compared at matched wall-clock, not matched steps.** This sweep used to assume the schedules cost the same per step, which held under the memory-only objective. It does not hold under the cost objective: the two propose different move types -- crude ~50% reorder, reheating ~46% recolor -- and a recolor rewrites a region's divisions and dirties far more bundles than a reorder does. Reheating costs ~1.7x per step, so equal steps handed crude 1.7x less machine and called the difference a schedule effect. Each arm's spb is now derived from a calibration pass so both land on the same wall-clock targets, which the per-cell cpu column lets you check. Incumbent targets: steps-per-buffer [40, 160, 640] for `reheating`.

Cells where both schedules reach the same score under every seed are counted as ties, not wins.

## Headline: per capacity x reorder move

| capacity | reorder move | cells | tied | crude better | reheat better | mean % | pooled 95% CI |
|---|---|--:|--:|--:|--:|--:|---|

**Match quality.** Achieved crude/reheating CPU ratio across cells: mean 0.89 (min 0.53, max 1.18); 1.00 is a perfect match. 3 of 132 cells sit on the engine's `min_steps=200` floor, where both arms run the same steps whatever spb was assigned, so the cheaper arm just uses less time. Those cells are unmatched by construction and tilt the aggregate toward crude.

## Per graph (non-tied cells only)

| graph | n | capacity | move | spb rh/cr | cpu s rh/cr | reheating | crude | delta % |
|---|--:|---|---|--:|--:|--:|--:|--:|

## Verdict: PROMOTE crude

A global default flip is justified only if crude wins (CI strictly below zero) in **every** capacity x move combination. If it wins under the sweep but not under `random`, the honest change is a conditional default.
