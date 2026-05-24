# 2026-05-24 Consensus Hit Mode

- created_at: `2026-05-24T15:13:31`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- champion: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\consensusmode_rank1_s0006p20m0004b080.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\consensusmode_rank2_s0005p25m0006b100.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\consensusmode_rank3_s001p15m00b050.csv']`

## Idea

- Start over with an unsupervised hit-mode view: candidate predictions are treated as a small coordinate cloud.
- Instead of predicting the target directly, choose a coordinate that sits in the densest weighted local consensus around strong candidates.
- This is deliberately different from residual fitting and post-hoc curvature thresholds; it tests whether candidate agreement is a usable proxy for 1cm hit probability.

## Champion OOF Proxy

| name | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm |
| --- | --- | --- | --- | --- | --- |
| champion_oof_proxy | 0.011362 | 0.006874 | 0.023886 | 0.039534 | 0.669900 |

## Consensus Leaderboard

| config_id | sigma | power | min_margin | max_route_fraction | blend | use_top_k | route_fraction | top_route | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | vs_champion_mean_delta | vs_champion_median_delta | vs_champion_p95_delta | vs_champion_max_delta | delta_hit_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.006000 | 2.000000 | 0.004000 | 0.220000 | 0.800000 | 7 | 0.020600 | fixed_a060:188,fixed_a090:16,temporal55:2 | 0.011362 | 0.006873 | 0.023886 | 0.039492 | 0.670200 | 0.000024 | 0.000000 | 0.000000 | 0.003391 | 0.000300 |
| 3 | 0.005000 | 2.500000 | 0.006000 | 0.160000 | 1.000000 | 7 | 0.018700 | fixed_a060:170,fixed_a090:16,temporal55:1 | 0.011362 | 0.006875 | 0.023886 | 0.039475 | 0.670200 | 0.000027 | 0.000000 | 0.000000 | 0.003945 | 0.000300 |
| 6 | 0.010000 | 1.500000 | 0.000000 | 0.400000 | 0.500000 | 11 | 0.400000 | fixed_a060:1825,temporal55:937,selector_soft:730 | 0.011362 | 0.006872 | 0.023941 | 0.039563 | 0.670200 | 0.000120 | 0.000000 | 0.000549 | 0.002746 | 0.000300 |
| 1 | 0.004000 | 3.000000 | 0.010000 | 0.080000 | 1.000000 | 5 | 0.004700 | fixed_a060:42,fixed_a090:4,temporal55:1 | 0.011362 | 0.006873 | 0.023886 | 0.039534 | 0.670000 | 0.000005 | 0.000000 | 0.000000 | 0.003945 | 0.000100 |
| 2 | 0.004500 | 3.000000 | 0.008000 | 0.120000 | 1.000000 | 5 | 0.008100 | fixed_a060:76,fixed_a090:4,temporal55:1 | 0.011362 | 0.006873 | 0.023886 | 0.039534 | 0.670000 | 0.000012 | 0.000000 | 0.000000 | 0.003945 | 0.000100 |
| 5 | 0.007500 | 2.000000 | 0.002000 | 0.300000 | 0.650000 | 9 | 0.026800 | fixed_a090:159,fixed_a060:107,temporal55:2 | 0.011362 | 0.006876 | 0.023886 | 0.039562 | 0.670000 | 0.000016 | 0.000000 | 0.000000 | 0.003570 | 0.000100 |

## Outputs

| rank | submission | config_id | cv_like_delta_hit | train_route_fraction | test_route_fraction | test_top_route | vs_champion_mean_delta | vs_champion_median_delta | vs_champion_p95_delta | vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | consensusmode_rank1_s0006p20m0004b080.csv | 4 | 0.000300 | 0.020600 | 0.019900 | fixed_a060:181,fixed_a090:14,selector_soft:4 | 0.000023 | 0.000000 | 0.000000 | 0.003318 |
| 2 | consensusmode_rank2_s0005p25m0006b100.csv | 3 | 0.000300 | 0.018700 | 0.018600 | fixed_a060:171,fixed_a090:12,selector_soft:3 | 0.000027 | 0.000000 | 0.000000 | 0.003976 |
| 3 | consensusmode_rank3_s001p15m00b050.csv | 6 | 0.000300 | 0.400000 | 0.400000 | temporal55:1995,selector_soft:1180,fixed_a060:387 | 0.000129 | 0.000000 | 0.000523 | 0.002074 |

## Candidate Diagnostics

| candidate | hit_rate | rescue_rate_vs_champion | harm_rate_vs_champion | mean_distance |
| --- | --- | --- | --- | --- |
| fixed_a060 | 0.670300 | 0.002900 | 0.002500 | 0.011365 |
| fixed_a120 | 0.670100 | 0.003400 | 0.003200 | 0.011362 |
| gate_t50 | 0.670000 | 0.000600 | 0.000500 | 0.011361 |
| cochamp_t52_t54 | 0.670000 | 0.000200 | 0.000100 | 0.011363 |
| champion | 0.669900 | 0.000000 | 0.000000 | 0.011362 |
| gate_t54 | 0.669700 | 0.000200 | 0.000400 | 0.011364 |
| fixed_a090 | 0.669100 | 0.002100 | 0.002900 | 0.011362 |
| temporal55 | 0.668800 | 0.014100 | 0.015200 | 0.011383 |
| selector_soft | 0.656700 | 0.010600 | 0.023800 | 0.011626 |
| phys_a275 | 0.600300 | 0.032800 | 0.102400 | 0.012933 |
| phys_v098_a275 | 0.598400 | 0.032700 | 0.104200 | 0.012859 |
| phys_a150 | 0.596200 | 0.028800 | 0.102500 | 0.012818 |
| phys_v102_a275 | 0.593800 | 0.030100 | 0.106200 | 0.013143 |
| phys_a400 | 0.591600 | 0.033400 | 0.111700 | 0.013222 |
| phys_a000 | 0.578800 | 0.026000 | 0.117100 | 0.012941 |
| poly_w3_d1 | 0.512400 | 0.034500 | 0.192000 | 0.015763 |
| wdiff_w5_d050 | 0.511600 | 0.032600 | 0.190900 | 0.015396 |
| wdiff_w7_d060 | 0.464600 | 0.029800 | 0.235100 | 0.016792 |
| wdiff_w5_d075 | 0.460800 | 0.032000 | 0.241100 | 0.017326 |
| poly_w4_d2 | 0.456900 | 0.032000 | 0.245000 | 0.019086 |
| poly_w5_d2 | 0.435500 | 0.030700 | 0.265100 | 0.020139 |
| poly_w4_d1 | 0.435400 | 0.031600 | 0.266100 | 0.019011 |
| poly_w3_d2 | 0.423700 | 0.030100 | 0.276300 | 0.020174 |
| wdiff_w11_d070 | 0.399800 | 0.026400 | 0.296500 | 0.018742 |
| poly_w5_d1 | 0.363100 | 0.027600 | 0.334400 | 0.022362 |
| poly_w7_d2 | 0.345500 | 0.023300 | 0.347700 | 0.024217 |
| poly_w7_d1 | 0.245400 | 0.017700 | 0.442200 | 0.029387 |
| poly_w11_d2 | 0.212300 | 0.013900 | 0.471500 | 0.033061 |
| poly_w11_d3 | 0.206900 | 0.014800 | 0.477800 | 0.034896 |
| poly_w11_d1 | 0.127300 | 0.009200 | 0.551800 | 0.042703 |

## Notes

- This is a high-variance public probe because config ranking still uses train OOF diagnostics.
- If the top route is mostly near-champion variants, submit risk is moderate; if it routes to low-hit physics/poly candidates, treat it as exploratory only.
