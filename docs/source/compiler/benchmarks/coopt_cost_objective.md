# Is the cost model a better objective than memory-only?

Both arms solve the same graphs; each finished plan is then re-scored by the cost model, which is the only number comparable across the two. **Negative means the cost-model arm's plan is cheaper.** Seeds 50-59, engine defaults otherwise.

Cells where both arms produce the same predicted cost under every seed are counted as ties, not wins.

## Headline: per capacity

| capacity | cells | tied | cost better | memory better | mean % | pooled 95% CI |
|---|--:|--:|--:|--:|--:|---|
| inf | 11 | 2 | 9 | 0 | -57.12 | [-60.62, -53.52] |
| footprint//1 | 11 | 2 | 9 | 0 | -57.12 | [-60.62, -53.51] |
| footprint//4 | 11 | 2 | 9 | 0 | -48.99 | [-53.77, -44.08] |
| footprint//16 | 11 | 2 | 9 | 0 | -22.12 | [-27.86, -16.94] |
| footprint//64 | 11 | 3 | 8 | 0 | -18.38 | [-26.38, -11.15] |

## Per graph x capacity

| graph | n | bundles | capacity | memory | cost | delta % | off-seed divisions (mem / cost) | cpu s (mem / cost) |
|---|--:|--:|---|--:|--:|--:|---|---|
| mlp | 3 | 1 | inf | 68,729,046 | 18,100,178 | -73.66 | 0.0 / 3.0 | 0.002 / 0.009 |
| swiglu | 4 | 1 | inf | 132,309,636 | 28,977,122 | -78.10 | 0.0 / 4.0 | 0.002 / 0.010 |
| softmax | 6 | 1 | inf | 26,127,019 | 26,127,019 | +0.00 | 0.0 / 0.0 | 0.005 / 0.012 |
| rms_norm | 7 | 1 | inf | 2,132,658 | 2,132,658 | +0.00 | 0.0 / 0.0 | 0.006 / 0.015 |
| sdpa | 9 | 1 | inf | 20,520,344 | 11,669,380 | -43.13 | 0.0 / 8.8 | 0.009 / 0.031 |
| simple_attn | 9 | 1 | inf | 19,104,566 | 9,452,682 | -50.52 | 0.0 / 8.8 | 0.008 / 0.030 |
| block_x2 | 26 | 1 | inf | 139,685,031 | 44,029,034 | -68.48 | 0.0 / 25.0 | 0.037 / 0.209 |
| block_x3 | 39 | 1 | inf | 209,308,014 | 65,637,010 | -68.64 | 0.0 / 37.4 | 0.067 / 0.449 |
| flash_attention | 44 | 4 | inf | 189,379,974 | 133,195,622 | -29.67 | 0.0 / 40.9 | 0.094 / 0.537 |
| block_x4 | 52 | 1 | inf | 278,930,998 | 87,244,985 | -68.72 | 0.0 / 50.3 | 0.106 / 0.785 |
| flash_big | 80 | 4 | inf | 738,001,447 | 493,136,679 | -33.18 | 0.0 / 76.5 | 0.380 / 1.920 |
| mlp | 3 | 1 | footprint//1 | 68,729,046 | 18,100,178 | -73.66 | 0.0 / 3.0 | 0.002 / 0.009 |
| swiglu | 4 | 1 | footprint//1 | 132,309,636 | 28,977,122 | -78.10 | 0.0 / 4.0 | 0.002 / 0.010 |
| softmax | 6 | 1 | footprint//1 | 26,127,019 | 26,127,019 | +0.00 | 0.0 / 0.0 | 0.005 / 0.012 |
| rms_norm | 7 | 1 | footprint//1 | 2,132,658 | 2,132,658 | +0.00 | 0.0 / 0.0 | 0.006 / 0.015 |
| sdpa | 9 | 1 | footprint//1 | 20,520,344 | 11,669,380 | -43.13 | 0.0 / 8.8 | 0.008 / 0.030 |
| simple_attn | 9 | 1 | footprint//1 | 19,104,566 | 9,452,682 | -50.52 | 0.0 / 8.8 | 0.008 / 0.031 |
| block_x2 | 26 | 1 | footprint//1 | 139,685,031 | 44,029,034 | -68.48 | 0.0 / 25.0 | 0.034 / 0.206 |
| block_x3 | 39 | 1 | footprint//1 | 209,308,014 | 65,637,010 | -68.64 | 0.0 / 37.4 | 0.065 / 0.445 |
| flash_attention | 44 | 4 | footprint//1 | 189,379,974 | 133,195,622 | -29.67 | 0.0 / 40.9 | 0.095 / 0.539 |
| block_x4 | 52 | 1 | footprint//1 | 278,930,998 | 87,244,985 | -68.72 | 0.0 / 50.3 | 0.107 / 0.767 |
| flash_big | 80 | 4 | footprint//1 | 738,001,447 | 493,136,679 | -33.18 | 0.0 / 76.5 | 0.367 / 1.866 |
| mlp | 3 | 1 | footprint//4 | 26,556,023 | 18,082,797 | -31.91 | 2.9 / 3.0 | 0.002 / 0.009 |
| swiglu | 4 | 1 | footprint//4 | 132,309,636 | 28,977,122 | -78.10 | 0.0 / 4.0 | 0.002 / 0.010 |
| softmax | 6 | 1 | footprint//4 | 26,127,019 | 26,127,019 | +0.00 | 6.0 / 6.0 | 0.005 / 0.012 |
| rms_norm | 7 | 1 | footprint//4 | 2,132,658 | 2,132,658 | +0.00 | 7.0 / 7.0 | 0.006 / 0.016 |
| sdpa | 9 | 1 | footprint//4 | 13,213,201 | 11,669,380 | -11.68 | 9.0 / 8.8 | 0.009 / 0.031 |
| simple_attn | 9 | 1 | footprint//4 | 19,104,566 | 9,452,682 | -50.52 | 0.0 / 8.8 | 0.009 / 0.031 |
| block_x2 | 26 | 1 | footprint//4 | 139,685,031 | 44,029,034 | -68.48 | 0.0 / 25.0 | 0.033 / 0.205 |
| block_x3 | 39 | 1 | footprint//4 | 209,308,014 | 65,637,010 | -68.64 | 0.0 / 37.4 | 0.065 / 0.440 |
| flash_attention | 44 | 4 | footprint//4 | 189,379,974 | 133,195,622 | -29.67 | 0.0 / 40.9 | 0.094 / 0.536 |
| block_x4 | 52 | 1 | footprint//4 | 278,930,998 | 87,244,985 | -68.72 | 0.0 / 50.3 | 0.107 / 0.775 |
| flash_big | 80 | 4 | footprint//4 | 738,001,447 | 493,136,679 | -33.18 | 0.0 / 76.5 | 0.364 / 1.896 |
| mlp | 3 | 1 | footprint//16 | 23,231,266 | 18,069,160 | -22.22 | 2.9 / 3.0 | 0.002 / 0.009 |
| swiglu | 4 | 1 | footprint//16 | 132,309,636 | 28,977,122 | -78.10 | 0.0 / 4.0 | 0.002 / 0.010 |
| softmax | 6 | 1 | footprint//16 | 26,127,019 | 26,127,019 | +0.00 | 6.0 / 6.0 | 0.005 / 0.012 |
| rms_norm | 7 | 1 | footprint//16 | 2,137,541 | 2,137,541 | +0.00 | 6.2 / 6.4 | 0.006 / 0.015 |
| sdpa | 9 | 1 | footprint//16 | 11,915,482 | 11,669,380 | -2.07 | 9.0 / 8.9 | 0.009 / 0.032 |
| simple_attn | 9 | 1 | footprint//16 | 9,956,114 | 9,452,682 | -5.06 | 8.7 / 8.8 | 0.008 / 0.031 |
| block_x2 | 26 | 1 | footprint//16 | 49,811,058 | 44,029,034 | -11.61 | 24.6 / 25.0 | 0.033 / 0.208 |
| block_x3 | 39 | 1 | footprint//16 | 82,772,232 | 65,637,010 | -20.70 | 36.4 / 37.1 | 0.065 / 0.448 |
| flash_attention | 44 | 4 | footprint//16 | 162,437,084 | 133,373,872 | -17.89 | 35.8 / 40.9 | 0.094 / 0.547 |
| block_x4 | 52 | 1 | footprint//16 | 111,134,854 | 87,244,985 | -21.50 | 48.3 / 50.5 | 0.105 / 0.775 |
| flash_big | 80 | 4 | footprint//16 | 615,455,714 | 492,636,698 | -19.96 | 63.6 / 76.6 | 0.363 / 1.860 |
| mlp | 3 | 1 | footprint//64 | 29,473,870 | 18,610,410 | -36.86 | 2.4 / 3.0 | 0.002 / 0.009 |
| swiglu | 4 | 1 | footprint//64 | 132,309,636 | 28,977,122 | -78.10 | 0.0 / 4.0 | 0.002 / 0.010 |
| softmax | 6 | 1 | footprint//64 | 26,127,019 | 26,127,019 | +0.00 | 6.0 / 6.0 | 0.005 / 0.011 |
| rms_norm | 7 | 1 | footprint//64 | 2,137,541 | 2,137,541 | +0.00 | 6.0 / 5.6 | 0.006 / 0.016 |
| sdpa | 9 | 1 | footprint//64 | 11,720,801 | 11,669,380 | -0.44 | 8.7 / 8.6 | 0.008 / 0.031 |
| simple_attn | 9 | 1 | footprint//64 | 9,452,682 | 9,452,682 | +0.00 | 8.8 / 8.8 | 0.008 / 0.030 |
| block_x2 | 26 | 1 | footprint//64 | 44,965,584 | 44,029,034 | -2.08 | 24.6 / 24.8 | 0.034 / 0.208 |
| block_x3 | 39 | 1 | footprint//64 | 69,548,863 | 65,637,010 | -5.62 | 36.9 / 37.0 | 0.065 / 0.454 |
| flash_attention | 44 | 4 | footprint//64 | 151,383,426 | 142,136,584 | -6.11 | 40.0 / 40.4 | 0.091 / 0.534 |
| block_x4 | 52 | 1 | footprint//64 | 93,663,866 | 87,244,985 | -6.85 | 49.2 / 50.2 | 0.107 / 0.787 |
| flash_big | 80 | 4 | footprint//64 | 561,042,873 | 499,600,613 | -10.95 | 71.3 / 76.2 | 0.345 / 1.848 |

## What each arm does to the division vector

The memory objective prices a division only through residency, so where the seed already fits it never moves one: `off-seed divisions` is 0 at the loose capacities for every graph. The cost arm moves nearly all of them, and will give up residency to do it. That is the behavioural difference the score gap is made of -- not a better search of the same space, a different space.

## Where it does not help

Tied at unbounded capacity: `softmax`, `rms_norm`. These graphs score identically under every division vector, so the cost model is as blind on them as the memory objective is. That is not a failure of the objective -- with no matmul there is no reward to trade against -- but it does mean they cannot discriminate anything, and any future sweep should treat them as inert rather than as evidence.

## Verdict: COST MODEL WINS -- largest at loose LX

The gap is widest where LX is roomy and narrowest where it is tight: under pressure the memory objective is *forced* into splits for residency reasons and stumbles onto much of the same answer, while at loose capacity it has no signal at all. So the cost model earns its keep exactly where the incumbent is inert, which is the regime a default affects most.

