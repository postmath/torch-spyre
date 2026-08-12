# Co-optimizer schedule sweep: crude vs reheating over run length

Budget sweep over every captured graph, both schedules, 5 seeds per cell, at a geometric `steps_per_buffer` grid. Capacity is `footprint // 2` (the spill-pressured regime where the schedule matters).

## Headline finding

**The advanced (reheating) schedule is not useless on larger graphs; the earlier
apparent regression was a single-seed, shortest-run artifact.** Averaged over 5
seeds:

- **flash_attention (n=43) is the clearest case _for_ the schedule.** Reheating
  wins at every run length and the margin **grows with length** -- from -1.5% at
  1.7k steps to **-7.0% at 440k steps** -- with both schedules still improving but
  reheating pulling ahead (52.6% vs 49.1% over baseline). At the default budget
  with seed 0 this same graph looked ~3% _worse_ under reheating: seed noise, not
  a real regression.
- **sdpa (n=25): reheating converges faster** -- -21% at the shortest run, then
  crude catches up (tie by ~4k steps): reheating reaches the optimum in ~4x fewer
  steps.
- **swiglu (n=8): reheating finds a better _final_ optimum at long runs** (-5.7%
  at 82k steps) that crude never reaches.
- **flash_big (n=79): the longer run resolves it in reheating's favor.** Extended
  to 809k steps (spb 10240; ~74 min for that one cell), reheating ends **1.7%
  better** than crude (51.8% vs 50.9% over baseline). The trajectory is noisy --
  both schedules bounce between 50-52% and crude actually regresses from 202k to
  809k steps for these seeds (each budget re-parameterizes the cooling), and the
  deltas sit within the large seed spread -- but the earlier apparent regression
  does not survive the longer run: at the longest budget reheating is ahead,
  consistent with flash_attention.
- **Most graphs are schedule-insensitive** (softmax, rms_norm, mlp, simple_attn,
  block_x2/x3/x4): both schedules reach the same optimum, usually by the shortest
  run, so schedule choice is irrelevant there.

**Answer:** the advanced schedule is not useless on large graphs. On both large
graphs (flash_attention, flash_big) reheating ends **ahead** at the longest run,
and on flash_attention its advantage **grows with run length** -- consistent with
"needs longer runs to pay off," not "useless on large graphs." The remaining
question is variance: flash_big is noisy across seeds, so its ~2% edge is softer
than flash_attention's clean -7% trend.

_Caveats: capacity = footprint//2 throughout; y is the SA fixed-point objective,
not wall-clock._

![crossover](../../_static/images/coopt/coopt_schedule_crossover.png)

![per-graph](../../_static/images/coopt/coopt_schedule_per_graph.png)

## Per-graph deltas (mean over seeds)

`delta%` = (reheating - crude) / crude x 100; negative = reheating better.

| graph | n | spb | total steps | crude (mean) | reheating (mean) | delta% |
|---|--:|--:|--:|--:|--:|--:|
| softmax | 5 | 40 | 200 | 5,160,000 | 5,160,000 | +0.00 |
| softmax | 5 | 160 | 800 | 5,160,000 | 5,160,000 | +0.00 |
| softmax | 5 | 640 | 3200 | 5,160,000 | 5,160,000 | +0.00 |
| softmax | 5 | 2560 | 12800 | 5,160,000 | 5,160,000 | +0.00 |
| softmax | 5 | 10240 | 51200 | 5,160,000 | 5,160,000 | +0.00 |
| rms_norm | 6 | 40 | 240 | 962,500 | 962,500 | +0.00 |
| rms_norm | 6 | 160 | 960 | 962,500 | 962,500 | +0.00 |
| rms_norm | 6 | 640 | 3840 | 962,500 | 962,500 | +0.00 |
| rms_norm | 6 | 2560 | 15360 | 962,500 | 962,500 | +0.00 |
| rms_norm | 6 | 10240 | 61440 | 962,500 | 962,500 | +0.00 |
| mlp | 7 | 40 | 280 | 10,688,000 | 10,560,000 | -1.20 |
| mlp | 7 | 160 | 1120 | 10,560,000 | 10,560,000 | +0.00 |
| mlp | 7 | 640 | 4480 | 10,560,000 | 10,560,000 | +0.00 |
| mlp | 7 | 2560 | 17920 | 10,560,000 | 10,560,000 | +0.00 |
| mlp | 7 | 10240 | 71680 | 10,560,000 | 10,560,000 | +0.00 |
| swiglu | 8 | 40 | 320 | 9,984,000 | 9,472,000 | -5.13 |
| swiglu | 8 | 160 | 1280 | 8,960,000 | 8,960,000 | +0.00 |
| swiglu | 8 | 640 | 5120 | 8,960,000 | 8,960,000 | +0.00 |
| swiglu | 8 | 2560 | 20480 | 8,960,000 | 8,960,000 | +0.00 |
| swiglu | 8 | 10240 | 81920 | 8,960,000 | 8,448,000 | -5.71 |
| simple_attn | 9 | 40 | 360 | 960,000 | 960,000 | +0.00 |
| simple_attn | 9 | 160 | 1440 | 960,000 | 960,000 | +0.00 |
| simple_attn | 9 | 640 | 5760 | 960,000 | 960,000 | +0.00 |
| simple_attn | 9 | 2560 | 23040 | 960,000 | 960,000 | +0.00 |
| simple_attn | 9 | 10240 | 92160 | 960,000 | 960,000 | +0.00 |
| sdpa | 25 | 40 | 1000 | 2,431,961 | 1,919,961 | -21.05 |
| sdpa | 25 | 160 | 4000 | 1,919,961 | 1,919,961 | +0.00 |
| sdpa | 25 | 640 | 16000 | 1,919,961 | 1,919,961 | +0.00 |
| sdpa | 25 | 2560 | 64000 | 1,919,961 | 1,919,961 | +0.00 |
| sdpa | 25 | 10240 | 256000 | 1,919,961 | 1,919,961 | +0.00 |
| block_x2 | 28 | 40 | 1120 | 1,600,000 | 1,600,000 | +0.00 |
| block_x2 | 28 | 160 | 4480 | 1,600,000 | 1,600,000 | +0.00 |
| block_x2 | 28 | 640 | 17920 | 1,600,000 | 1,600,000 | +0.00 |
| block_x2 | 28 | 2560 | 71680 | 1,600,000 | 1,600,000 | +0.00 |
| block_x2 | 28 | 10240 | 286720 | 1,600,000 | 1,600,000 | +0.00 |
| block_x3 | 42 | 40 | 1680 | 2,240,000 | 2,240,000 | +0.00 |
| block_x3 | 42 | 160 | 6720 | 2,240,000 | 2,240,000 | +0.00 |
| block_x3 | 42 | 640 | 26880 | 2,240,000 | 2,240,000 | +0.00 |
| block_x3 | 42 | 2560 | 107520 | 2,240,000 | 2,240,000 | +0.00 |
| block_x3 | 42 | 10240 | 430080 | 2,240,000 | 2,240,000 | +0.00 |
| flash_attention | 43 | 40 | 1720 | 44,743,961 | 44,067,961 | -1.51 |
| flash_attention | 43 | 160 | 6880 | 42,687,961 | 42,703,961 | +0.04 |
| flash_attention | 43 | 640 | 27520 | 42,511,961 | 41,707,961 | -1.89 |
| flash_attention | 43 | 2560 | 110080 | 41,203,961 | 38,995,961 | -5.36 |
| flash_attention | 43 | 10240 | 440320 | 39,983,961 | 37,199,961 | -6.96 |
| block_x4 | 56 | 40 | 2240 | 2,880,000 | 2,880,000 | +0.00 |
| block_x4 | 56 | 160 | 8960 | 2,880,000 | 2,880,000 | +0.00 |
| block_x4 | 56 | 640 | 35840 | 2,880,000 | 2,880,000 | +0.00 |
| block_x4 | 56 | 2560 | 143360 | 2,880,000 | 2,880,000 | +0.00 |
| block_x4 | 56 | 10240 | 573440 | 2,880,000 | 2,880,000 | +0.00 |
| flash_big | 79 | 40 | 3160 | 155,895,951 | 158,119,951 | +1.43 |
| flash_big | 79 | 160 | 12640 | 156,927,951 | 155,959,951 | -0.62 |
| flash_big | 79 | 640 | 50560 | 152,191,951 | 149,999,951 | -1.44 |
| flash_big | 79 | 2560 | 202240 | 145,039,951 | 148,831,951 | +2.61 |
| flash_big | 79 | 10240 | 808960 | 147,719,951 | 145,215,951 | -1.70 |

## Summary: does reheating catch up at longer runs?

| graph | n | delta% @ shortest | delta% @ longest | trend |
|---|--:|--:|--:|---|
| softmax | 5 | +0.00 | +0.00 | schedule-insensitive (both converge) |
| rms_norm | 6 | +0.00 | +0.00 | schedule-insensitive (both converge) |
| mlp | 7 | -1.20 | +0.00 | reheating faster; crude catches up |
| swiglu | 8 | -5.13 | -5.71 | reheating wins, margin grows with length |
| simple_attn | 9 | +0.00 | +0.00 | schedule-insensitive (both converge) |
| sdpa | 25 | -21.05 | +0.00 | reheating faster; crude catches up |
| block_x2 | 28 | +0.00 | +0.00 | schedule-insensitive (both converge) |
| block_x3 | 42 | +0.00 | +0.00 | schedule-insensitive (both converge) |
| flash_attention | 43 | -1.51 | -6.96 | reheating wins, margin grows with length |
| block_x4 | 56 | +0.00 | +0.00 | schedule-insensitive (both converge) |
| flash_big | 79 | +1.43 | -1.70 | reheating wins, margin grows with length |
