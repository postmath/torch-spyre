# Convergence: which option gets there soonest

Steps until an arm's best-seen score is within a tolerance of the best score *any* arm reached on that graph, at the default `steps_per_buffer=40`, 5 seeds, capacity `footprint//2`. Lower is sooner. `final gap` is where the arm ends up, which is what every other benchmark in this series reports.

## Per arm

| arm | median steps to within 1% | to within 0.1% | mean final gap % | first to arrive (graphs) |
|---|--:|--:|--:|--:|
| `incumbent` | 82 (10/11) | 80 (8/11) | +0.13 | 2 |
| `reheating` | 40 (10/11) | 34 (8/11) | +0.09 | 7 |
| `reorder=random` | 73 (9/11) | 66 (7/11) | +0.12 | 1 |
| `nested` | 118 (8/11) | 118 (8/11) | +0.45 | 0 |

## Per graph: steps to within 1% of the best any arm reached

| graph | n | `incumbent` | `reheating` | `reorder=random` | `nested` |
|---|--:|--:|--:|--:|--:|
| mlp | 3 | never | never | never | never |
| swiglu | 4 | 80 | 29 | 46 | 49 |
| softmax | 6 | 0 | 0 | 0 | 0 |
| rms_norm | 7 | 0 | 0 | 0 | 0 |
| sdpa | 9 | 133 | 39 | 73 | 128 |
| simple_attn | 9 | 69 | 24 | never | 108 |
| block_x2 | 26 | 84 | 56 | 100 | 355 |
| block_x3 | 39 | 81 | 56 | 124 | 388 |
| flash_attention | 44 | 616 | 224 | 539 | never |
| block_x4 | 52 | 205 | 41 | 66 | 561 |
| flash_big | 80 | 1280 | 1254 | 1216 | never |

![curves](../../_static/images/coopt/coopt_convergence_curves.png)

## Reading this

A tie in `final gap` with a difference in `steps to within 1%` is an arm that reaches the same place faster -- invisible to every endpoint sweep in this series, and the state `schedule` turned out to be in. A tie in both is a genuine non-difference. `never` means the arm did not reach the bar inside the default budget on some graph, which is the case a mean over the arrivals would quietly drop.
