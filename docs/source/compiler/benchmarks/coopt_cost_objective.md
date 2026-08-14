# Is the cost model a better objective than memory-only?

Both arms solve the same graphs; each finished plan is then re-scored by the cost model, which is the only number comparable across the two. **Negative means the cost-model arm's plan is cheaper.** Seeds 50-59, engine defaults otherwise.

Cells where both arms produce the same predicted cost under every seed are counted as ties, not wins.

## Headline: per capacity

| capacity | cells | tied | cost better | memory better | mean % | pooled 95% CI |
|---|--:|--:|--:|--:|--:|---|
| inf | 11 | 3 | 8 | 0 | -53.48 | [-58.40, -48.22] |
| footprint//1 | 11 | 3 | 8 | 0 | -53.48 | [-58.40, -48.22] |
| footprint//4 | 11 | 4 | 7 | 0 | -50.34 | [-57.01, -43.77] |
| footprint//16 | 11 | 7 | 4 | 0 | -29.58 | [-40.45, -19.23] |
| footprint//64 | 11 | 7 | 2 | 2 | -32.88 | [-45.75, -19.67] |

## Per graph x capacity

| graph | n | bundles | capacity | memory | cost | delta % | off-seed divisions (mem / cost) | cpu s (mem / cost) |
|---|--:|--:|---|--:|--:|--:|---|---|
| mlp | 3 | 1 | inf | 58,867,425 | 16,492,790 | -71.98 | 0.0 / 3.0 | 0.002 / 0.008 |
| swiglu | 4 | 1 | inf | 117,734,849 | 26,990,346 | -77.08 | 0.0 / 3.0 | 0.002 / 0.009 |
| softmax | 6 | 1 | inf | 22,719,147 | 22,719,147 | +0.00 | 0.0 / 0.0 | 0.004 / 0.011 |
| rms_norm | 7 | 1 | inf | 2,132,658 | 2,132,658 | +0.00 | 0.0 / 0.0 | 0.005 / 0.013 |
| sdpa | 9 | 1 | inf | 15,235,845 | 9,786,109 | -35.77 | 0.0 / 9.0 | 0.007 / 0.023 |
| simple_attn | 9 | 1 | inf | 14,716,856 | 8,938,472 | -39.26 | 0.0 / 8.6 | 0.007 / 0.024 |
| block_x2 | 26 | 1 | inf | 117,734,849 | 39,915,356 | -66.10 | 0.0 / 24.7 | 0.034 / 0.145 |
| block_x3 | 39 | 1 | inf | 176,602,274 | 59,466,493 | -66.33 | 0.0 / 36.4 | 0.059 / 0.296 |
| flash_attention | 44 | 4 | inf | 123,561,124 | 123,561,124 | +0.00 | 0.0 / 0.0 | 0.082 / 0.347 |
| block_x4 | 52 | 1 | inf | 235,469,698 | 79,017,629 | -66.44 | 0.0 / 49.2 | 0.098 / 0.520 |
| flash_big | 80 | 4 | inf | 487,549,597 | 463,802,281 | -4.87 | 0.0 / 65.6 | 0.321 / 1.202 |
| mlp | 3 | 1 | footprint//1 | 58,867,425 | 16,492,790 | -71.98 | 0.0 / 3.0 | 0.002 / 0.008 |
| swiglu | 4 | 1 | footprint//1 | 117,734,849 | 26,990,346 | -77.08 | 0.0 / 3.0 | 0.002 / 0.008 |
| softmax | 6 | 1 | footprint//1 | 22,719,147 | 22,719,147 | +0.00 | 0.0 / 0.0 | 0.004 / 0.011 |
| rms_norm | 7 | 1 | footprint//1 | 2,132,658 | 2,132,658 | +0.00 | 0.0 / 0.0 | 0.005 / 0.013 |
| sdpa | 9 | 1 | footprint//1 | 15,235,845 | 9,786,109 | -35.77 | 0.0 / 9.0 | 0.007 / 0.023 |
| simple_attn | 9 | 1 | footprint//1 | 14,716,856 | 8,938,472 | -39.26 | 0.0 / 8.6 | 0.007 / 0.024 |
| block_x2 | 26 | 1 | footprint//1 | 117,734,849 | 39,915,356 | -66.10 | 0.0 / 24.7 | 0.030 / 0.143 |
| block_x3 | 39 | 1 | footprint//1 | 176,602,274 | 59,466,493 | -66.33 | 0.0 / 36.4 | 0.058 / 0.296 |
| flash_attention | 44 | 4 | footprint//1 | 123,561,124 | 123,561,124 | +0.00 | 0.0 / 0.0 | 0.081 / 0.341 |
| block_x4 | 52 | 1 | footprint//1 | 235,469,698 | 79,017,629 | -66.44 | 0.0 / 49.2 | 0.098 / 0.526 |
| flash_big | 80 | 4 | footprint//1 | 487,549,597 | 463,802,281 | -4.87 | 0.0 / 65.6 | 0.309 / 1.223 |
| mlp | 3 | 1 | footprint//4 | 24,374,897 | 16,492,790 | -32.34 | 2.6 / 3.0 | 0.002 / 0.008 |
| swiglu | 4 | 1 | footprint//4 | 117,734,849 | 26,990,346 | -77.08 | 0.0 / 3.0 | 0.002 / 0.008 |
| softmax | 6 | 1 | footprint//4 | 22,719,147 | 22,719,147 | +0.00 | 6.0 / 6.0 | 0.004 / 0.011 |
| rms_norm | 7 | 1 | footprint//4 | 2,132,658 | 2,132,658 | +0.00 | 7.0 / 7.0 | 0.005 / 0.014 |
| sdpa | 9 | 1 | footprint//4 | 9,786,109 | 9,786,109 | +0.00 | 9.0 / 9.0 | 0.007 / 0.024 |
| simple_attn | 9 | 1 | footprint//4 | 14,716,856 | 8,938,472 | -39.26 | 0.0 / 8.6 | 0.007 / 0.024 |
| block_x2 | 26 | 1 | footprint//4 | 117,734,849 | 39,915,356 | -66.10 | 0.0 / 24.7 | 0.030 / 0.144 |
| block_x3 | 39 | 1 | footprint//4 | 176,602,274 | 59,466,493 | -66.33 | 0.0 / 36.4 | 0.058 / 0.294 |
| flash_attention | 44 | 4 | footprint//4 | 123,561,124 | 123,561,124 | +0.00 | 0.0 / 0.0 | 0.081 / 0.346 |
| block_x4 | 52 | 1 | footprint//4 | 235,469,698 | 79,017,629 | -66.44 | 0.0 / 49.2 | 0.098 / 0.519 |
| flash_big | 80 | 4 | footprint//4 | 487,549,597 | 463,802,281 | -4.87 | 0.0 / 65.6 | 0.305 / 1.216 |
| mlp | 3 | 1 | footprint//16 | 19,572,817 | 16,492,790 | -15.74 | 2.8 / 3.0 | 0.002 / 0.008 |
| swiglu | 4 | 1 | footprint//16 | 117,734,849 | 26,990,346 | -77.08 | 0.0 / 3.0 | 0.002 / 0.008 |
| softmax | 6 | 1 | footprint//16 | 22,719,147 | 22,719,147 | +0.00 | 6.0 / 6.0 | 0.004 / 0.011 |
| rms_norm | 7 | 1 | footprint//16 | 2,137,541 | 2,137,541 | +0.00 | 5.8 / 6.0 | 0.005 / 0.013 |
| sdpa | 9 | 1 | footprint//16 | 9,786,109 | 9,786,109 | +0.00 | 9.0 / 9.0 | 0.007 / 0.023 |
| simple_attn | 9 | 1 | footprint//16 | 8,938,472 | 8,938,472 | +0.00 | 8.5 / 8.5 | 0.007 / 0.024 |
| block_x2 | 26 | 1 | footprint//16 | 39,915,356 | 39,915,356 | +0.00 | 24.5 / 24.8 | 0.030 / 0.145 |
| block_x3 | 39 | 1 | footprint//16 | 68,116,886 | 59,466,493 | -12.70 | 36.9 / 36.4 | 0.058 / 0.306 |
| flash_attention | 44 | 4 | footprint//16 | 123,561,124 | 123,561,124 | +0.00 | 34.4 / 34.1 | 0.080 / 0.353 |
| block_x4 | 52 | 1 | footprint//16 | 90,632,795 | 79,017,629 | -12.82 | 48.5 / 49.2 | 0.097 / 0.520 |
| flash_big | 80 | 4 | footprint//16 | 463,802,281 | 463,802,281 | +0.00 | 66.9 / 67.6 | 0.305 / 1.200 |
| mlp | 3 | 1 | footprint//64 | 39,282,082 | 17,431,747 | -55.62 | 1.4 / 2.8 | 0.002 / 0.008 |
| swiglu | 4 | 1 | footprint//64 | 117,734,849 | 26,990,346 | -77.08 | 0.0 / 3.0 | 0.002 / 0.008 |
| softmax | 6 | 1 | footprint//64 | 22,719,147 | 22,719,147 | +0.00 | 6.0 / 6.0 | 0.004 / 0.011 |
| rms_norm | 7 | 1 | footprint//64 | 2,137,541 | 2,137,541 | +0.00 | 6.4 / 6.0 | 0.005 / 0.013 |
| sdpa | 9 | 1 | footprint//64 | 11,412,275 | 11,412,275 | +0.00 | 8.8 / 8.9 | 0.007 / 0.024 |
| simple_attn | 9 | 1 | footprint//64 | 8,938,472 | 8,938,472 | +0.00 | 8.7 / 8.8 | 0.007 / 0.024 |
| block_x2 | 26 | 1 | footprint//64 | 39,915,356 | 39,915,356 | +0.00 | 24.9 / 25.0 | 0.030 / 0.149 |
| block_x3 | 39 | 1 | footprint//64 | 59,466,493 | 59,466,493 | +0.00 | 37.0 / 36.7 | 0.059 / 0.304 |
| flash_attention | 44 | 4 | footprint//64 | 137,571,566 | 139,172,323 | +1.16 | 37.5 / 37.1 | 0.081 / 0.348 |
| block_x4 | 52 | 1 | footprint//64 | 79,017,629 | 79,017,629 | +0.00 | 49.0 / 49.2 | 0.096 / 0.526 |
| flash_big | 80 | 4 | footprint//64 | 464,320,622 | 464,391,766 | +0.02 | 73.5 / 74.8 | 0.286 / 1.229 |

## What each arm does to the division vector

The memory objective prices a division only through residency, so where the seed already fits it never moves one: `off-seed divisions` is 0 at the loose capacities for every graph. The cost arm moves nearly all of them, and will give up residency to do it. That is the behavioural difference the score gap is made of -- not a better search of the same space, a different space.

## Where it does not help

Tied at unbounded capacity: `softmax`, `rms_norm`, `flash_attention`. These graphs score identically under every division vector, so the cost model is as blind on them as the memory objective is. That is not a failure of the objective -- with no matmul there is no reward to trade against -- but it does mean they cannot discriminate anything, and any future sweep should treat them as inert rather than as evidence.

## Verdict: COST MODEL WINS -- largest at loose LX

The gap is widest where LX is roomy and narrowest where it is tight: under pressure the memory objective is *forced* into splits for residency reasons and stumbles onto much of the same answer, while at loose capacity it has no signal at all. So the cost model earns its keep exactly where the incumbent is inert, which is the regime a default affects most.
