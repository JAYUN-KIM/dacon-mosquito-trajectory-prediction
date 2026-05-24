# 2026-05-24 Analog KNN Residual

- created_at: `2026-05-24T15:17:02`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- champion: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogknn_rank1_k64_p10_s012_cap00025_r100.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogknn_rank2_k16_p10_s012_cap0002_r010.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogknn_rank3_k24_p10_s014_cap00025_r012.csv']`

## Idea

- Start over with analog forecasting: find train trajectories with similar recent motion and transfer their champion residuals.
- The update is heavily shrunk and capped, because the target is 1cm hit-rate and wrong residual transfer can easily destroy good champion hits.
- Route fractions test two modes: only high-confidence analog neighborhoods versus applying a tiny residual everywhere.

## Champion OOF Proxy

| name | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm |
| --- | --- | --- | --- | --- | --- |
| champion_oof_proxy | 0.011362 | 0.006874 | 0.023886 | 0.039534 | 0.669900 |

## CV Leaderboard

| config_id | k | power | shrink | cap | route_fraction | mean_r_hit | min_r_hit | mean_baseline_r_hit | mean_delta_hit | min_delta_hit | mean_distance | actual_route_fraction | vs_champion_mean_delta | vs_champion_p95_delta | vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.000000 | 64.000000 | 1.000000 | 0.120000 | 0.002500 | 1.000000 | 0.664500 | 0.653000 | 0.664000 | 0.000500 | -0.000500 | 0.011401 | 1.000000 | 0.000232 | 0.000633 | 0.001740 |
| 2.000000 | 16.000000 | 1.000000 | 0.120000 | 0.002000 | 0.100000 | 0.664333 | 0.651000 | 0.664000 | 0.000333 | 0.000000 | 0.011407 | 0.100000 | 0.000014 | 0.000122 | 0.000559 |
| 3.000000 | 24.000000 | 1.000000 | 0.140000 | 0.002500 | 0.120000 | 0.664333 | 0.651000 | 0.664000 | 0.000333 | 0.000000 | 0.011407 | 0.120000 | 0.000018 | 0.000137 | 0.000538 |
| 7.000000 | 32.000000 | 1.000000 | 0.100000 | 0.002000 | 1.000000 | 0.664333 | 0.654000 | 0.664000 | 0.000333 | -0.003000 | 0.011404 | 1.000000 | 0.000242 | 0.000702 | 0.001933 |
| 6.000000 | 64.000000 | 2.000000 | 0.200000 | 0.004000 | 0.240000 | 0.664167 | 0.650500 | 0.664000 | 0.000167 | 0.000000 | 0.011407 | 0.240000 | 0.000041 | 0.000241 | 0.000870 |
| 1.000000 | 12.000000 | 1.000000 | 0.100000 | 0.001500 | 0.080000 | 0.664000 | 0.650500 | 0.664000 | 0.000000 | 0.000000 | 0.011407 | 0.080000 | 0.000010 | 0.000092 | 0.000470 |
| 4.000000 | 32.000000 | 1.500000 | 0.160000 | 0.003000 | 0.160000 | 0.664000 | 0.651000 | 0.664000 | 0.000000 | -0.000500 | 0.011408 | 0.160000 | 0.000025 | 0.000181 | 0.000513 |
| 5.000000 | 48.000000 | 1.500000 | 0.180000 | 0.003500 | 0.200000 | 0.663833 | 0.650500 | 0.664000 | -0.000167 | -0.000500 | 0.011407 | 0.200000 | 0.000032 | 0.000210 | 0.000665 |
| 9.000000 | 96.000000 | 1.000000 | 0.140000 | 0.003000 | 1.000000 | 0.663833 | 0.651500 | 0.664000 | -0.000167 | -0.002000 | 0.011400 | 1.000000 | 0.000240 | 0.000631 | 0.001479 |

## Outputs

| rank | submission | config_id | cv_mean_delta_hit | cv_min_delta_hit | cv_mean_route_fraction | test_route_fraction | test_uncertainty_p50 | test_uncertainty_p95 | vs_champion_mean_delta | vs_champion_median_delta | vs_champion_p95_delta | vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | analogknn_rank1_k64_p10_s012_cap00025_r100.csv | 8 | 0.000500 | -0.000500 | 1.000000 | 1.000000 | 19.554849 | 54.305740 | 0.000228 | 0.000160 | 0.000633 | 0.002051 |
| 2 | analogknn_rank2_k16_p10_s012_cap0002_r010.csv | 2 | 0.000333 | 0.000000 | 0.100000 | 0.100000 | 17.797572 | 50.751008 | 0.000015 | 0.000000 | 0.000123 | 0.000658 |
| 3 | analogknn_rank3_k24_p10_s014_cap00025_r012.csv | 3 | 0.000333 | 0.000000 | 0.120000 | 0.120000 | 18.241483 | 51.912615 | 0.000019 | 0.000000 | 0.000154 | 0.000764 |

## Notes

- This is a genuinely new axis versus previous curvature/post-process probes.
- If public score drops, the miss residual is not locally transferable by recent trajectory shape; the useful next step would be regime-specific pseudo-labeling rather than KNN residual.
