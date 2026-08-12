# Crude's proposal weights on the score/CPU frontier

`reorder_weight` / `flip_weight` / `recolor_weight`, swept for the first time. Every arm is calibrated to the incumbent's wall-clock at each of steps-per-buffer [40, 160], 10 seeds, capacity `footprint//2`. Score is % against the incumbent weights (0.5, 0.3, 0.2) (negative = better); CPU is relative to the same. An arm only beats the default if it is at or left of 1.00x CPU *and* at or below 0% score.

## Per arm, pooled

| arm | (reorder, flip, recolor) | proposal mix r/f/c | CPU vs incumbent | score % | 95% CI |
|---|---|---|--:|--:|---|
| `incumbent` | (0.5, 0.3, 0.2) | 50/29/21% | 1.00x | +0.000 | [-0.113, +0.114] |
| `reorder-heavy` | (0.8, 0.1, 0.1) | 79/10/10% | 0.92x | +0.079 | [-0.038, +0.198] |
| `reorder-only-ish` | (0.94, 0.03, 0.03) | 95/3/3% | 0.86x | +0.523 | [+0.309, +0.770] (worse) |
| `balanced` | (0.33, 0.33, 0.33) | 31/34/34% | 1.02x | -0.099 | [-0.198, -0.005] |
| `flip-heavy` | (0.2, 0.6, 0.2) | 22/58/20% | 1.05x | -0.012 | [-0.113, +0.086] |
| `recolor-heavy` | (0.2, 0.2, 0.6) | 22/18/60% | 1.04x | -0.124 | [-0.229, -0.027] |
| `structure-heavy` | (0.1, 0.45, 0.45) | 9/47/44% | 1.06x | -0.092 | [-0.202, +0.014] |
| `as-reheating` | (0.05, 0.49, 0.46) | 7/47/46% | 1.07x | -0.102 | [-0.202, -0.011] |

![frontier](../../_static/images/coopt/coopt_move_weights_frontier.png)

## Verdict

**No arm dominates the incumbent weights.** Nothing swept here is both significantly better on score and no more expensive, so the guessed defaults survive their first contact with evidence -- worth stating plainly rather than leaving as an absence, since this mix was the largest unmeasured surface in the engine.

The frontier, cheapest first: `reorder-only-ish` (0.86x, +0.523%), `reorder-heavy` (0.92x, +0.079%), `incumbent` (1.00x, +0.000%), `balanced` (1.02x, -0.099%), `recolor-heavy` (1.04x, -0.124%). `incumbent` is on it, sitting between the arms that buy score with CPU and the arms that buy CPU with score. Everything here is a ~0.1% score effect against a cost model that has never been checked on hardware, so the frontier's shape is more informative than any single point on it.

**Was the schedule difference ever about cooling?** `as-reheating` puts the reheating schedule's observed mix (~5/49/46) inside the crude schedule, changing the proposal mix and nothing else. It lands at -0.102% score for 1.07x CPU -- the same shape as reheating itself against crude: a small score gain bought with CPU. So the schedule knob is mostly a mix knob, and these three weights largely subsume it. That also means the two are not independent: retuning them re-opens the schedule decision.

