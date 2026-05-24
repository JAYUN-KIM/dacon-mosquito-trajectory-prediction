# 2026-05-24 Regime Miss Policy

- created_at: `2026-05-24T15:09:25`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- champion: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regimemiss_rank1_c64_min55_net0003_p165.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regimemiss_rank2_c16_min120_net0006_p125.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regimemiss_rank3_c24_min100_net0006_p125.csv']`

## Idea

- Start over from miss-regime analysis instead of creating another global coordinate perturbation.
- Cluster train/test by motion regime, then switch only clusters where train OOF says an alternate candidate has net hit gain.
- CV policy selection is used to reduce pure in-sample oracle overfit.

## Champion OOF Proxy

| name | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm |
| --- | --- | --- | --- | --- | --- |
| champion_oof_proxy | 0.011362 | 0.006874 | 0.023886 | 0.039534 | 0.669900 |

## Regime Policy CV

| config_id | n_clusters | min_rows | min_net | penalty | max_route_fraction | mean_r_hit | mean_baseline_r_hit | mean_delta_hit | min_delta_hit | mean_distance | mean_route_fraction | mean_policy_clusters | top_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 64 | 55 | 0.003000 | 1.650000 | 0.350000 | 0.664667 | 0.664000 | 0.000667 | -0.001000 | 0.011419 | 0.354500 | 14.000000 | temporal55 |
| 1 | 16 | 120 | 0.006000 | 1.250000 | 0.180000 | 0.664500 | 0.664000 | 0.000500 | -0.000500 | 0.011399 | 0.159000 | 4.666667 | temporal55 |
| 2 | 24 | 100 | 0.006000 | 1.250000 | 0.220000 | 0.664167 | 0.664000 | 0.000167 | -0.002000 | 0.011410 | 0.223500 | 7.000000 | temporal55 |
| 6 | 40 | 80 | 0.002000 | 1.200000 | 0.280000 | 0.663833 | 0.664000 | -0.000167 | -0.000500 | 0.011408 | 0.295167 | 6.333333 | fixed_a120 |
| 3 | 32 | 80 | 0.005000 | 1.350000 | 0.250000 | 0.663500 | 0.664000 | -0.000500 | -0.001000 | 0.011410 | 0.241833 | 9.000000 | temporal55 |
| 4 | 48 | 70 | 0.004000 | 1.500000 | 0.300000 | 0.663333 | 0.664000 | -0.000667 | -0.003500 | 0.011431 | 0.276333 | 15.000000 | temporal55 |

## Outputs

| rank | submission | config_id | cv_mean_delta_hit | cv_min_delta_hit | cv_mean_route_fraction | test_route_fraction | policy_clusters | test_top_candidate | vs_champion_mean_delta | vs_champion_median_delta | vs_champion_p95_delta | vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | regimemiss_rank1_c64_min55_net0003_p165.csv | 5 | 0.000667 | -0.001000 | 0.354500 | 0.352200 | 14 | temporal55 | 0.000079 | 0.000000 | 0.000485 | 0.003007 |
| 2 | regimemiss_rank2_c16_min120_net0006_p125.csv | 1 | 0.000500 | -0.000500 | 0.159000 | 0.187100 | 5 | temporal55 | 0.000078 | 0.000000 | 0.000516 | 0.007951 |
| 3 | regimemiss_rank3_c24_min100_net0006_p125.csv | 2 | 0.000167 | -0.002000 | 0.223500 | 0.228300 | 7 | temporal55 | 0.000097 | 0.000000 | 0.000598 | 0.007951 |

## Candidate Oracle Diagnostics

| candidate | hit_rate | rescue_rate_vs_champion | harm_rate_vs_champion | mean_distance |
| --- | --- | --- | --- | --- |
| poly_w3_d1 | 0.512400 | 0.034500 | 0.192000 | 0.015763 |
| phys_a400 | 0.591600 | 0.033400 | 0.111700 | 0.013222 |
| phys_a275 | 0.600300 | 0.032800 | 0.102400 | 0.012933 |
| phys_v098_a275 | 0.598400 | 0.032700 | 0.104200 | 0.012859 |
| wdiff_w5_d050 | 0.511600 | 0.032600 | 0.190900 | 0.015396 |
| wdiff_w5_d075 | 0.460800 | 0.032000 | 0.241100 | 0.017326 |
| poly_w4_d2 | 0.456900 | 0.032000 | 0.245000 | 0.019086 |
| poly_w4_d1 | 0.435400 | 0.031600 | 0.266100 | 0.019011 |
| poly_w5_d2 | 0.435500 | 0.030700 | 0.265100 | 0.020139 |
| phys_v102_a275 | 0.593800 | 0.030100 | 0.106200 | 0.013143 |
| poly_w3_d2 | 0.423700 | 0.030100 | 0.276300 | 0.020174 |
| wdiff_w7_d060 | 0.464600 | 0.029800 | 0.235100 | 0.016792 |
| phys_a150 | 0.596200 | 0.028800 | 0.102500 | 0.012818 |
| poly_w5_d1 | 0.363100 | 0.027600 | 0.334400 | 0.022362 |
| wdiff_w11_d070 | 0.399800 | 0.026400 | 0.296500 | 0.018742 |
| phys_a000 | 0.578800 | 0.026000 | 0.117100 | 0.012941 |
| poly_w7_d2 | 0.345500 | 0.023300 | 0.347700 | 0.024217 |
| poly_w7_d1 | 0.245400 | 0.017700 | 0.442200 | 0.029387 |
| poly_w11_d3 | 0.206900 | 0.014800 | 0.477800 | 0.034896 |
| temporal55 | 0.668800 | 0.014100 | 0.015200 | 0.011383 |
| poly_w11_d2 | 0.212300 | 0.013900 | 0.471500 | 0.033061 |
| selector_soft | 0.656700 | 0.010600 | 0.023800 | 0.011626 |
| poly_w11_d1 | 0.127300 | 0.009200 | 0.551800 | 0.042703 |
| fixed_a120 | 0.670100 | 0.003400 | 0.003200 | 0.011362 |
| fixed_a060 | 0.670300 | 0.002900 | 0.002500 | 0.011365 |
| fixed_a090 | 0.669100 | 0.002100 | 0.002900 | 0.011362 |
| gate_t50 | 0.670000 | 0.000600 | 0.000500 | 0.011361 |
| cochamp_t52_t54 | 0.670000 | 0.000200 | 0.000100 | 0.011363 |
| gate_t54 | 0.669700 | 0.000200 | 0.000400 | 0.011364 |
| champion | 0.669900 | 0.000000 | 0.000000 | 0.011362 |

## Policy Details

| config_id | seed | cluster | candidate |
| --- | --- | --- | --- |
| 1 | 42 | 3 | temporal55 |
| 1 | 42 | 9 | temporal55 |
| 1 | 42 | 10 | fixed_a060 |
| 1 | 42 | 11 | temporal55 |
| 1 | 42 | 14 | fixed_a120 |
| 1 | 777 | 1 | temporal55 |
| 1 | 777 | 2 | temporal55 |
| 1 | 777 | 14 | temporal55 |
| 1 | 777 | 15 | selector_soft |
| 1 | 777 | 3 | fixed_a120 |
| 1 | 2026 | 0 | temporal55 |
| 1 | 2026 | 3 | fixed_a120 |
| 1 | 2026 | 5 | temporal55 |
| 1 | 2026 | 8 | temporal55 |
| 2 | 42 | 10 | temporal55 |
| 2 | 42 | 2 | temporal55 |
| 2 | 42 | 14 | temporal55 |
| 2 | 42 | 16 | fixed_a120 |
| 2 | 42 | 15 | fixed_a060 |
| 2 | 42 | 19 | selector_soft |
| 2 | 777 | 14 | temporal55 |
| 2 | 777 | 22 | temporal55 |
| 2 | 777 | 6 | temporal55 |
| 2 | 777 | 2 | temporal55 |
| 2 | 777 | 21 | selector_soft |
| 2 | 777 | 18 | fixed_a120 |
| 2 | 777 | 10 | temporal55 |
| 2 | 2026 | 4 | temporal55 |
| 2 | 2026 | 1 | temporal55 |
| 2 | 2026 | 6 | temporal55 |
| 2 | 2026 | 18 | temporal55 |
| 2 | 2026 | 22 | fixed_a060 |
| 2 | 2026 | 20 | fixed_a120 |
| 2 | 2026 | 17 | fixed_a060 |
| 2 | 2026 | 3 | gate_t50 |
| 3 | 42 | 0 | gate_t50 |
| 3 | 42 | 2 | temporal55 |
| 3 | 42 | 5 | temporal55 |
| 3 | 42 | 9 | gate_t50 |
| 3 | 42 | 12 | temporal55 |
| 3 | 42 | 21 | fixed_a060 |
| 3 | 42 | 22 | fixed_a060 |
| 3 | 42 | 25 | temporal55 |
| 3 | 42 | 26 | gate_t50 |
| 3 | 777 | 1 | selector_soft |
| 3 | 777 | 8 | fixed_a060 |
| 3 | 777 | 10 | temporal55 |
| 3 | 777 | 17 | temporal55 |
| 3 | 777 | 21 | fixed_a120 |
| 3 | 777 | 22 | selector_soft |
| 3 | 777 | 23 | fixed_a120 |
| 3 | 777 | 28 | temporal55 |
| 3 | 777 | 30 | temporal55 |
| 3 | 777 | 31 | temporal55 |
| 3 | 2026 | 8 | temporal55 |
| 3 | 2026 | 11 | temporal55 |
| 3 | 2026 | 1 | fixed_a120 |
| 3 | 2026 | 9 | fixed_a120 |
| 3 | 2026 | 21 | temporal55 |
| 3 | 2026 | 10 | temporal55 |
| 3 | 2026 | 26 | fixed_a060 |
| 3 | 2026 | 15 | fixed_a060 |
| 4 | 42 | 1 | fixed_a120 |
| 4 | 42 | 6 | temporal55 |
| 4 | 42 | 7 | selector_soft |
| 4 | 42 | 8 | gate_t50 |
| 4 | 42 | 10 | fixed_a060 |
| 4 | 42 | 12 | fixed_a120 |
| 4 | 42 | 14 | temporal55 |
| 4 | 42 | 18 | fixed_a120 |
| 4 | 42 | 23 | fixed_a060 |
| 4 | 42 | 25 | fixed_a120 |
| 4 | 42 | 28 | temporal55 |
| 4 | 42 | 33 | gate_t50 |
| 4 | 42 | 37 | temporal55 |
| 4 | 42 | 38 | fixed_a060 |
| 4 | 42 | 39 | temporal55 |
| 4 | 42 | 45 | temporal55 |
| 4 | 42 | 46 | temporal55 |
| 4 | 777 | 1 | temporal55 |
| 4 | 777 | 11 | fixed_a060 |
| 4 | 777 | 27 | fixed_a060 |
| 4 | 777 | 42 | fixed_a060 |
| 4 | 777 | 39 | temporal55 |
| 4 | 777 | 25 | fixed_a120 |
| 4 | 777 | 28 | fixed_a120 |
| 4 | 777 | 16 | temporal55 |
| 4 | 777 | 44 | fixed_a120 |
| 4 | 777 | 26 | temporal55 |
| 4 | 777 | 8 | fixed_a090 |
| 4 | 777 | 45 | temporal55 |
| 4 | 777 | 21 | temporal55 |
| 4 | 777 | 20 | fixed_a060 |
| 4 | 777 | 7 | fixed_a060 |
| 4 | 2026 | 0 | selector_soft |
| 4 | 2026 | 7 | fixed_a060 |
| 4 | 2026 | 10 | temporal55 |
| 4 | 2026 | 16 | selector_soft |
| 4 | 2026 | 17 | fixed_a060 |
| 4 | 2026 | 19 | gate_t50 |
| 4 | 2026 | 20 | temporal55 |
| 4 | 2026 | 24 | temporal55 |
| 4 | 2026 | 25 | fixed_a120 |
| 4 | 2026 | 35 | fixed_a060 |
| 4 | 2026 | 36 | phys_a275 |
| 4 | 2026 | 37 | temporal55 |
| 4 | 2026 | 42 | fixed_a120 |
| 5 | 42 | 22 | fixed_a120 |
| 5 | 42 | 40 | gate_t54 |
| 5 | 42 | 25 | temporal55 |
| 5 | 42 | 35 | gate_t50 |
| 5 | 42 | 0 | fixed_a090 |
| 5 | 42 | 16 | fixed_a060 |
| 5 | 42 | 7 | fixed_a060 |
| 5 | 42 | 4 | fixed_a120 |
| 5 | 42 | 47 | temporal55 |
| 5 | 42 | 2 | temporal55 |
| 5 | 42 | 51 | temporal55 |
| 5 | 42 | 38 | temporal55 |
| 5 | 42 | 48 | fixed_a120 |
