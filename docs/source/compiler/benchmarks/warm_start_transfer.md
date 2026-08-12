# Does the layout permutation survive a work-division change?

The joint annealer keeps one layout permutation `pi` and re-uses it across work-division
changes. That warm start is the whole reason the joint loop can beat the nested baseline
(enumerate divisions × solve layout for each). It also creates a specific failure mode: a
division flip is judged by Metropolis against a `pi` tuned for the *old* division, so good flips
can be rejected and the search collapses to "layout search around the seed division" — exactly
the lower bound the joint loop is supposed to beat.

`warm_start_transfer.py` measures whether the warm start carries reusable information. At the
layout solver's interface a division change is (a) a resize of a correlated subset of buffers and
(b) an eligibility toggle for K-split eviction. The harness models resize-only, which is the
conservative case: eligibility toggling only removes buffers and relieves capacity pressure.

```{note}
This page is written from the console transcripts in `data/scratchpad_wst_*.out`, not generated
from structured results. It is the one page here that cannot be rebuilt with `--report`, and its
figures were not re-derived — they are quoted. The experiment also predates the cost objective
entirely: it scores on the layout `quality()` objective, on synthetic buffer universes rather
than captured graphs. Treat it as evidence about *layout transfer*, not about anything the cost
model decides.
```

## Part A — how fast does a warm layout recover?

n=40, capacity 60% of baseline peak, budget L=1200, reference 7200 steps, ε=2%. `b*` is the
burst length needed to reach within ε of a cold full-budget solve.

| division change | warm score at b=0, vs optimum | `b*` warm | `b*` cold | `b*`/L |
|--:|--:|--:|--:|--:|
| 0.10 | 0.889 | 60 | 251 | 0.05 |
| 0.25 | 0.972 | 113 | 144 | 0.09 |
| 0.50 | 0.951 | 970 | 354 | 0.81 |

At small and medium changes the ordering alone recovers most of the optimum before any
refinement, and reaches parity in 5–9% of the budget — several times faster than a cold start.
**At the largest change it inverts**: 970 warm against 354 cold. A stale ordering is worse than
no ordering once the division moves far enough, so the warm start is not unconditionally good.

## Part B — does a warm score rank divisions correctly?

The decision-relevant question is not the score but the *ranking*: does Metropolis on the warm
state prefer the divisions that would win if each were solved properly? Measured as
sign-agreement across division comparisons, averaged over 5 instances (seeds 100–104):

| burst | 0 | 1n | 2n | 3n | 5n | 8n |
|---|--:|--:|--:|--:|--:|--:|
| unweighted | 63% | 85% | 87% | 91% | 93% | 95% |
| stakes-weighted | 69% | 92% | 93% | 96% | 98% | 99% |
| Spearman vs true optimum | +0.29 | +0.80 | +0.84 | +0.91 | +0.95 | +0.96 |

A burst of roughly 1n–3n steps after a division flip takes decision fidelity from coin-flip
territory to 85–96%. Stakes-weighting sits consistently above unweighted, meaning the residual
misrankings are low-stakes near-ties where a wrong Metropolis call costs little.

**The single-instance run overstated this badly.** One instance reported b=0 sign-agreement of
91% and Spearman +0.94; across five instances the same quantities are 63% and +0.29, with b=0
ranging 53–82% between instances. The burst columns are both higher and much tighter
(3n: 88–95%), so the stale layout is the noisy regime and a small burst raises *and* stabilizes
fidelity. Any future version of this experiment should run multiple instances by default.

## Scaling with graph size

Smallest absolute burst reaching 90% stakes-weighted agreement:

| n | 10 | 20 | 40 | 80 |
|---|--:|--:|--:|--:|
| `b*` | 5 | 5 | 20 | 20 |
| `b*`/n | 0.50 | 0.25 | 0.50 | 0.25 |

`b*` grows with n but far more slowly, and neither `b*`/n nor `b*`/√n is flat across four
points measured on two instances each. The useful conclusion is the weak one: the burst does not
need to scale linearly with the graph, so a fraction-of-n rule is not obviously right.

## What this settles, and what it does not

The collapse-to-baseline risk is **low, conditional on a burst**. The ordering transfers well
enough that good flips are not systematically rejected, provided a flip carries a small layout
burst with it — which is why the engine's compound flip+burst move exists.

Two loose ends worth naming:

* `burst_fraction` currently defaults to **0.5**, i.e. a burst of 0.5n, which sits *below* the
  1n–3n range where these transcripts show fidelity crossing 85–91%. That default has never been
  swept, and this experiment is the closest thing to evidence about it. Whether 0.5n is
  adequate under the cost objective — where scoring, not layout, dominates per-step cost — is
  untested.
* The large-magnitude inversion (warm 970 vs cold 354 at change 0.50) suggests region-recolor,
  which moves many divisions at once, may want a larger burst than an atomic flip, or may want
  to discard the warm layout entirely. Also untested.
