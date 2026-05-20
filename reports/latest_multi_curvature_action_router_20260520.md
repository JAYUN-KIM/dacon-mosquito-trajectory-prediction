# 2026-05-20 Multi-Curvature Action Router

- created_at: `2026-05-20T22:22:12`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- current_best: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- public_feedback: `multicurv_action_rank2_currentblend25actiontop3p2.csv = 0.69020`
- action_count: `29`
- idea: instead of a single curvature gate, learn hit probability for multiple curvature config/alpha actions.
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\multicurv_action_rank1_currentblend35actiontop3p2.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\multicurv_action_rank2_currentblend25actiontop3p2.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\multicurv_action_rank3_currentblend35actiontop5p4.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\multicurv_action_rank4_currentblend50actiontop3p2.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\multicurv_action_rank5_currentblend35actionhard.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\multicurv_action_rank6_actiontop3p2.csv']`

## CV Leaderboard

| strategy | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- |
| action_hard | 0.668400 | 0.011934 | 0.652500 | 0.011385 | 0.006870 |
| action_top3_p2 | 0.665300 | 0.011256 | 0.651000 | 0.011431 | 0.006906 |
| action_top5_p2 | 0.665100 | 0.011871 | 0.649500 | 0.011419 | 0.006897 |
| action_top5_p4 | 0.665100 | 0.011871 | 0.649500 | 0.011419 | 0.006897 |

## Action OOF Table

| idx | action | alpha | label_count | oof_hit_rate | oof_mean_distance |
| --- | --- | --- | --- | --- | --- |
| 8 | w1_tm0p25_s0p25_d0p98_a120 | 0.120000 | 49 | 0.670700 | 0.011363 |
| 7 | w1_tm0p25_s0p25_d0p98_a105 | 0.105000 | 1 | 0.670400 | 0.011363 |
| 19 | w2_tm0p25_s0p5_d0p98_a105 | 0.105000 | 2 | 0.670200 | 0.011363 |
| 12 | w1_tm0p25_s0p5_d1p00_a120 | 0.120000 | 1550 | 0.670200 | 0.011370 |
| 4 | w1_tm0p25_s0p5_d0p98_a120 | 0.120000 | 929 | 0.670100 | 0.011362 |
| 6 | w1_tm0p25_s0p25_d0p98_a095 | 0.095000 | 0 | 0.670100 | 0.011363 |
| 15 | w1_tm0p25_s0p0_d0p98_a105 | 0.105000 | 9 | 0.670100 | 0.011368 |
| 11 | w1_tm0p25_s0p5_d1p00_a105 | 0.105000 | 5 | 0.670000 | 0.011368 |
| 3 | w1_tm0p25_s0p5_d0p98_a105 | 0.105000 | 2 | 0.669900 | 0.011361 |
| 5 | w1_tm0p25_s0p25_d0p98_a075 | 0.075000 | 3 | 0.669900 | 0.011365 |
| 20 | w2_tm0p25_s0p5_d0p98_a120 | 0.120000 | 942 | 0.669800 | 0.011363 |
| 10 | w1_tm0p25_s0p5_d1p00_a095 | 0.095000 | 4 | 0.669800 | 0.011367 |
| 14 | w1_tm0p25_s0p0_d0p98_a095 | 0.095000 | 8 | 0.669800 | 0.011368 |
| 18 | w2_tm0p25_s0p5_d0p98_a095 | 0.095000 | 4 | 0.669700 | 0.011363 |
| 16 | w1_tm0p25_s0p0_d0p98_a120 | 0.120000 | 1931 | 0.669700 | 0.011368 |
| 2 | w1_tm0p25_s0p5_d0p98_a095 | 0.095000 | 3 | 0.669600 | 0.011361 |
| 17 | w2_tm0p25_s0p5_d0p98_a075 | 0.075000 | 11 | 0.669600 | 0.011365 |
| 1 | w1_tm0p25_s0p5_d0p98_a075 | 0.075000 | 0 | 0.669500 | 0.011363 |
| 9 | w1_tm0p25_s0p5_d1p00_a075 | 0.075000 | 13 | 0.669400 | 0.011368 |
| 13 | w1_tm0p25_s0p0_d0p98_a075 | 0.075000 | 30 | 0.669100 | 0.011369 |
| 21 | w1_t0p25_s0p5_d0p98_a075 | 0.075000 | 16 | 0.668900 | 0.011398 |
| 0 | none | 0.000000 | 795 | 0.668600 | 0.011383 |
| 22 | w1_t0p25_s0p5_d0p98_a095 | 0.095000 | 4 | 0.668100 | 0.011405 |
| 23 | w1_t0p25_s0p5_d0p98_a105 | 0.105000 | 6 | 0.667600 | 0.011410 |
| 24 | w1_t0p25_s0p5_d0p98_a120 | 0.120000 | 112 | 0.667200 | 0.011417 |
| 25 | w1_t0p50_s0p5_d0p98_a075 | 0.075000 | 66 | 0.666400 | 0.011424 |
| 26 | w1_t0p50_s0p5_d0p98_a095 | 0.095000 | 33 | 0.664900 | 0.011444 |
| 27 | w1_t0p50_s0p5_d0p98_a105 | 0.105000 | 26 | 0.663600 | 0.011455 |
| 28 | w1_t0p50_s0p5_d0p98_a120 | 0.120000 | 3446 | 0.662700 | 0.011474 |

## Outputs

| rank | submission | strategy | vs_current_best_mean_delta | vs_current_best_median_delta | vs_current_best_p95_delta | vs_current_best_max_delta | vs_temporal_anchor_mean_delta | vs_temporal_anchor_median_delta | vs_temporal_anchor_p95_delta | vs_temporal_anchor_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | multicurv_action_rank1_currentblend35actiontop3p2.csv | currentblend35_action_top3_p2 | 0.000226 | 0.000159 | 0.000596 | 0.005301 | 0.000307 | 0.000184 | 0.000953 | 0.010721 |
| 2 | multicurv_action_rank2_currentblend25actiontop3p2.csv | currentblend25_action_top3_p2 | 0.000162 | 0.000114 | 0.000426 | 0.003787 | 0.000299 | 0.000159 | 0.000970 | 0.010899 |
| 3 | multicurv_action_rank3_currentblend35actiontop5p4.csv | currentblend35_action_top5_p4 | 0.000211 | 0.000148 | 0.000553 | 0.004716 | 0.000299 | 0.000173 | 0.000933 | 0.010776 |
| 4 | multicurv_action_rank4_currentblend50actiontop3p2.csv | currentblend50_action_top3_p2 | 0.000323 | 0.000227 | 0.000851 | 0.007573 | 0.000350 | 0.000235 | 0.000956 | 0.010465 |
| 5 | multicurv_action_rank5_currentblend35actionhard.csv | currentblend35_action_hard | 0.000134 | 0.000058 | 0.000538 | 0.005586 | 0.000240 | 0.000118 | 0.000880 | 0.012009 |
| 6 | multicurv_action_rank6_actiontop3p2.csv | action_top3_p2 | 0.000646 | 0.000454 | 0.001702 | 0.015146 | 0.000581 | 0.000440 | 0.001446 | 0.011240 |

## Notes

- This is a new curvature policy axis, not threshold-only tuning.
- Public risk is controlled by blending action-router outputs back toward the current best.
- If blended outputs improve, expand action pool and train an alpha-bucket policy.
- If pure action output improves, the sample-wise curvature choice is stronger than the previous binary gate.
