# Convergence: which option gets there soonest

Steps until an arm's best-seen score is within a tolerance of the best score *any* arm reached on that graph, at the default `steps_per_buffer=40`, 5 seeds, capacity `footprint//2`. Lower is sooner. `final gap` is where the arm ends up, which is what every other benchmark in this series reports.

## Per arm

| arm | median steps to within 1% | to within 0.1% | mean final gap % | first to arrive (graphs) |
|---|--:|--:|--:|--:|
| `incumbent` | 18 (11/11) | 21 (11/11) | +0.00 | 6 |
| `reheating` | 10 (11/11) | 10 (11/11) | +0.00 | 5 |
| `reorder=random` | 18 (11/11) | 20 (11/11) | +0.00 | 0 |
| `nested` | 23 (11/11) | 28 (11/11) | +0.00 | 0 |

## Per graph: steps to within 1% of the best any arm reached

| graph | n | `incumbent` | `reheating` | `reorder=random` | `nested` |
|---|--:|--:|--:|--:|--:|
| mlp | 3 | 18 | 11 | 18 | 20 |
| swiglu | 4 | 11 | 7 | 14 | 14 |
| softmax | 6 | 0 | 0 | 0 | 0 |
| rms_norm | 7 | 0 | 0 | 0 | 0 |
| sdpa | 9 | 21 | 10 | 21 | 34 |
| simple_attn | 9 | 18 | 10 | 14 | 23 |
| block_x2 | 26 | 28 | 20 | 28 | 38 |
| block_x3 | 39 | 31 | 37 | 43 | 136 |
| flash_attention | 44 | 0 | 0 | 0 | 0 |
| block_x4 | 52 | 41 | 49 | 49 | 269 |
| flash_big | 80 | 64 | 64 | 64 | 504 |

![curves](../../_static/images/coopt/coopt_convergence_curves.png)

## Reading this

A tie in `final gap` with a difference in `steps to within 1%` is an arm that reaches the same place faster -- invisible to every endpoint sweep in this series, and the state `schedule` turned out to be in. A tie in both is a genuine non-difference. `never` means the arm did not reach the bar inside the default budget on some graph, which is the case a mean over the arrivals would quietly drop.
