# 2026-05-24 Champion Micro Tuning

- created_at: `2026-05-24T15:33:06`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- failed_new_axes_today: `regimemiss_rank1 = 0.6906`, `analogknn_rank1 = 0.6886`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champmicro_rank1_gatet520a1075.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champmicro_rank2_ring5254a040.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champmicro_rank3_gatet520a1025.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champmicro_rank4_gatet530a105.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champmicro_rank5_gatet535a105.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champmicro_rank6_ring5254a025.csv']`

## Idea

- Public feedback says new miss-policy/KNN axes are worse, so we return to the robust 0.6912 curvature-gate family.
- The only active search space is a very small neighborhood around the t52/t54 co-champion plateau.
- Candidates either interpolate the t52/t54 disagreement ring or adjust alpha by 0.0025 around the known 0.105 correction.

## OOF Leaderboard

| name | kind | mean_alpha_train | route_fraction_train | mean_alpha_test | route_fraction_test | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | train_vs_champion_mean_delta | train_vs_champion_median_delta | train_vs_champion_p95_delta | train_vs_champion_max_delta | test_vs_champion_mean_delta | test_vs_champion_median_delta | test_vs_champion_p95_delta | test_vs_champion_max_delta | test_vs_t54_mean_delta | test_vs_t54_median_delta | test_vs_t54_p95_delta | test_vs_t54_max_delta | delta_hit_vs_champion | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate_t520_a1075 | gate | 0.066338 | 0.617100 | 0.061555 | 0.572600 | 0.011361 | 0.006879 | 0.023881 | 0.039493 | 0.670000 | 0.000008 | 0.000004 | 0.000028 | 0.000256 | 0.000007 | 0.000003 | 0.000027 | 0.000271 | 0.000039 | 0.000003 | 0.000174 | 0.008400 | 0.000100 | 0.000100 |
| ring52_54_a040 | ring | 0.061604 | 0.617100 | 0.055865 | 0.572600 | 0.011363 | 0.006878 | 0.023884 | 0.039534 | 0.670000 | 0.000017 | 0.000000 | 0.000000 | 0.003025 | 0.000020 | 0.000000 | 0.000102 | 0.005079 | 0.000012 | 0.000000 | 0.000062 | 0.003125 | 0.000100 | 0.000099 |
| gate_t520_a1025 | gate | 0.063253 | 0.617100 | 0.058691 | 0.572600 | 0.011362 | 0.006879 | 0.023891 | 0.039567 | 0.669900 | 0.000008 | 0.000004 | 0.000028 | 0.000256 | 0.000007 | 0.000003 | 0.000027 | 0.000271 | 0.000037 | 0.000003 | 0.000166 | 0.008009 | 0.000000 | -0.000000 |
| gate_t530_a105 | gate | 0.062212 | 0.592500 | 0.056700 | 0.540000 | 0.011363 | 0.006882 | 0.023911 | 0.039534 | 0.669900 | 0.000014 | 0.000000 | 0.000000 | 0.004550 | 0.000014 | 0.000000 | 0.000000 | 0.003274 | 0.000018 | 0.000000 | 0.000000 | 0.008204 | 0.000000 | -0.000001 |
| gate_t535_a105 | gate | 0.060879 | 0.579800 | 0.055104 | 0.524800 | 0.011364 | 0.006884 | 0.023911 | 0.039534 | 0.669900 | 0.000021 | 0.000000 | 0.000000 | 0.004886 | 0.000023 | 0.000000 | 0.000000 | 0.008204 | 0.000009 | 0.000000 | 0.000000 | 0.004781 | 0.000000 | -0.000001 |
| ring52_54_a025 | ring | 0.060867 | 0.617100 | 0.054883 | 0.572600 | 0.011364 | 0.006882 | 0.023886 | 0.039534 | 0.669900 | 0.000021 | 0.000000 | 0.000000 | 0.003723 | 0.000024 | 0.000000 | 0.000125 | 0.006251 | 0.000008 | 0.000000 | 0.000039 | 0.001953 | 0.000000 | -0.000001 |
| gate_t540_a1075 | gate | 0.061060 | 0.568000 | 0.054513 | 0.507100 | 0.011364 | 0.006882 | 0.023908 | 0.039493 | 0.669900 | 0.000035 | 0.000004 | 0.000103 | 0.004886 | 0.000038 | 0.000003 | 0.000170 | 0.008204 | 0.000007 | 0.000001 | 0.000025 | 0.000271 | 0.000000 | -0.000002 |
| gate_t525_a105 | gate | 0.063588 | 0.605600 | 0.058401 | 0.556200 | 0.011363 | 0.006874 | 0.023911 | 0.039534 | 0.669800 | 0.000007 | 0.000000 | 0.000000 | 0.004550 | 0.000007 | 0.000000 | 0.000000 | 0.003274 | 0.000025 | 0.000000 | 0.000000 | 0.008204 | -0.000100 | -0.000100 |
| ring52_54_a080 | ring | 0.063568 | 0.617100 | 0.058486 | 0.572600 | 0.011362 | 0.006874 | 0.023886 | 0.039534 | 0.669800 | 0.000007 | 0.000000 | 0.000000 | 0.001163 | 0.000008 | 0.000000 | 0.000039 | 0.001953 | 0.000024 | 0.000000 | 0.000125 | 0.006251 | -0.000100 | -0.000100 |
| ring52_54_a065 | ring | 0.062831 | 0.617100 | 0.057503 | 0.572600 | 0.011362 | 0.006874 | 0.023886 | 0.039534 | 0.669800 | 0.000010 | 0.000000 | 0.000000 | 0.001861 | 0.000012 | 0.000000 | 0.000062 | 0.003125 | 0.000020 | 0.000000 | 0.000102 | 0.005079 | -0.000100 | -0.000101 |
| sigmoid_c530_t012 | sigmoid | 0.062136 | 0.987800 | 0.056666 | 0.993700 | 0.011363 | 0.006874 | 0.023923 | 0.039492 | 0.669900 | 0.000026 | 0.000000 | 0.000139 | 0.003080 | 0.000029 | 0.000000 | 0.000153 | 0.003799 | 0.000030 | 0.000000 | 0.000156 | 0.004405 | 0.000000 | -0.000201 |
| gate_t540_a1025 | gate | 0.058220 | 0.568000 | 0.051978 | 0.507100 | 0.011365 | 0.006882 | 0.023908 | 0.039567 | 0.669700 | 0.000035 | 0.000004 | 0.000103 | 0.004886 | 0.000038 | 0.000003 | 0.000170 | 0.008204 | 0.000007 | 0.000001 | 0.000025 | 0.000271 | -0.000200 | -0.000202 |
| sigmoid_c535_t016 | sigmoid | 0.060675 | 0.999100 | 0.054929 | 0.999400 | 0.011363 | 0.006882 | 0.023901 | 0.039492 | 0.669900 | 0.000036 | 0.000001 | 0.000186 | 0.003204 | 0.000040 | 0.000002 | 0.000203 | 0.004514 | 0.000035 | 0.000002 | 0.000177 | 0.003690 | 0.000000 | -0.000202 |
| gate_t545_a105 | gate | 0.058349 | 0.555700 | 0.051555 | 0.491000 | 0.011364 | 0.006882 | 0.023911 | 0.039492 | 0.669600 | 0.000034 | 0.000000 | 0.000177 | 0.006434 | 0.000040 | 0.000000 | 0.000254 | 0.008204 | 0.000008 | 0.000000 | 0.000000 | 0.003064 | -0.000300 | -0.000302 |
| gate_t550_a105 | gate | 0.056868 | 0.541600 | 0.050001 | 0.476200 | 0.011365 | 0.006885 | 0.023911 | 0.039492 | 0.669600 | 0.000040 | 0.000000 | 0.000265 | 0.006434 | 0.000048 | 0.000000 | 0.000323 | 0.008204 | 0.000016 | 0.000000 | 0.000000 | 0.004841 | -0.000300 | -0.000302 |

## Outputs

| rank | submission | name | delta_hit_vs_champion_oof | selection_score | test_route_fraction | test_mean_alpha | test_changed_fraction_vs_champion | vs_champion_mean_delta | vs_champion_median_delta | vs_champion_p95_delta | vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | champmicro_rank1_gatet520a1075.csv | gate_t520_a1075 | 0.000100 | 0.000100 | 0.572600 | 0.061555 | 0.572600 | 0.000007 | 0.000003 | 0.000027 | 0.000271 |
| 2 | champmicro_rank2_ring5254a040.csv | ring52_54_a040 | 0.000100 | 0.000099 | 0.572600 | 0.055865 | 0.065500 | 0.000020 | 0.000000 | 0.000102 | 0.005079 |
| 3 | champmicro_rank3_gatet520a1025.csv | gate_t520_a1025 | 0.000000 | -0.000000 | 0.572600 | 0.058691 | 0.572600 | 0.000007 | 0.000003 | 0.000027 | 0.000271 |
| 4 | champmicro_rank4_gatet530a105.csv | gate_t530_a105 | 0.000000 | -0.000001 | 0.540000 | 0.056700 | 0.032600 | 0.000014 | 0.000000 | 0.000000 | 0.003274 |
| 5 | champmicro_rank5_gatet535a105.csv | gate_t535_a105 | 0.000000 | -0.000001 | 0.524800 | 0.055104 | 0.047800 | 0.000023 | 0.000000 | 0.000000 | 0.008204 |
| 6 | champmicro_rank6_ring5254a025.csv | ring52_54_a025 | 0.000000 | -0.000001 | 0.572600 | 0.054883 | 0.065500 | 0.000024 | 0.000000 | 0.000125 | 0.006251 |

## Notes

- Treat this as conservative exploitation, not a new discovery axis.
- If these stay at or below 0.6912, the next meaningful work should be a better gate calibration objective, not more coordinate perturbation.
