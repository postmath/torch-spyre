# Reheating cycle-count x run-length product sweep

Reheating schedule with `cycles` in [1, 2, 4, 8, 16] x `steps_per_buffer` in [40, 160, 640], 5 seeds/cell, capacity `footprint//2`. `cycle length = total_steps / cycles`, so `cycles=1` is a single long cool (no reheating) and larger values are more, shorter reheats. Heatmap cells are reheating-vs-crude % (blue/negative = reheating better).

## Headline finding

**The cycle count matters on 3 of 11 graphs (`mlp`, `flash_attention`, `flash_big`), and nowhere else.** The default `cycles=4` is off the best available count in 5 (graph, run length) cells -- worst `mlp` at spb 40 (+0.35% off the best), `flash_big` at spb 40 (+0.10% off the best), `flash_attention` at spb 40 (+0.04% off the best).

Cycle-insensitive here means every count in [1, 2, 4, 8, 16] lands within 0.005% at
every run length -- a tie, not a narrow win. 8 of 11 graphs
are in that state.

_Caveats: capacity = footprint//2; y is the cost model's fixed-point prediction,
not measured hardware time. A tie means the search reaches the same place by
every schedule shape, which is a statement about this corpus as much as about the
schedule._

![heatmap](results/coopt_cycle_heatmap.png)

![lines](results/coopt_cycle_lines.png)

## Best cycle count per (graph, run length)

| graph | n | spb | total steps | best cycles | reheat(best) vs crude % | cycles=4 vs crude % |
|---|--:|--:|--:|--:|--:|--:|
| mlp | 3 | 40 | 200 | 8 | -1.05 | -0.70 |
| mlp | 3 | 160 | 480 | 1 | -0.69 | -0.69 |
| mlp | 3 | 640 | 1920 | 1 | +0.00 | +0.00 |
| swiglu | 4 | 40 | 200 | 1 | +0.00 | +0.00 |
| swiglu | 4 | 160 | 640 | 1 | +0.00 | +0.00 |
| swiglu | 4 | 640 | 2560 | 2 | +0.00 | +0.00 |
| softmax | 6 | 40 | 240 | 1 | +0.00 | +0.00 |
| softmax | 6 | 160 | 960 | 1 | +0.00 | +0.00 |
| softmax | 6 | 640 | 3840 | 2 | +0.00 | +0.00 |
| rms_norm | 7 | 40 | 280 | 1 | +0.00 | +0.00 |
| rms_norm | 7 | 160 | 1120 | 1 | +0.00 | +0.00 |
| rms_norm | 7 | 640 | 4480 | 1 | +0.00 | +0.00 |
| sdpa | 9 | 40 | 360 | 1 | +0.00 | +0.00 |
| sdpa | 9 | 160 | 1440 | 2 | +0.00 | +0.00 |
| sdpa | 9 | 640 | 5760 | 2 | +0.00 | +0.00 |
| simple_attn | 9 | 40 | 360 | 2 | +0.00 | +0.00 |
| simple_attn | 9 | 160 | 1440 | 1 | +0.00 | +0.00 |
| simple_attn | 9 | 640 | 5760 | 1 | +0.00 | +0.00 |
| block_x2 | 26 | 40 | 1040 | 1 | +0.00 | +0.00 |
| block_x2 | 26 | 160 | 4160 | 2 | +0.00 | +0.00 |
| block_x2 | 26 | 640 | 16640 | 1 | +0.00 | +0.00 |
| block_x3 | 39 | 40 | 1560 | 1 | +0.00 | +0.00 |
| block_x3 | 39 | 160 | 6240 | 1 | +0.00 | +0.00 |
| block_x3 | 39 | 640 | 24960 | 2 | +0.00 | +0.00 |
| flash_attention | 44 | 40 | 1760 | 2 | -0.03 | +0.01 |
| flash_attention | 44 | 160 | 7040 | 2 | +0.00 | +0.02 |
| flash_attention | 44 | 640 | 28160 | 2 | +0.00 | +0.00 |
| block_x4 | 52 | 40 | 2080 | 1 | +0.00 | +0.00 |
| block_x4 | 52 | 160 | 8320 | 4 | +0.00 | +0.00 |
| block_x4 | 52 | 640 | 33280 | 1 | +0.00 | +0.00 |
| flash_big | 80 | 40 | 3200 | 16 | +0.17 | +0.28 |
| flash_big | 80 | 160 | 12800 | 4 | +0.19 | +0.19 |
| flash_big | 80 | 640 | 51200 | 8 | +0.14 | +0.14 |
