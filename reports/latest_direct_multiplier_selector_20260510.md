# 2026-05-10 Direct Multiplier Selector

- 생성 시각: `2026-05-10T16:57:22`
- 데이터 경로: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- 현재 public best: `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv = 0.68300`
- OOF folds: `5`
- selector CV seeds: `[42, 777, 2026]`
- 후보 multiplier 수: `12`
- feature 수: `752` + selector extra
- direct feature family: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- 생성한 제출 파일: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_selector_rank1_selectorconf055.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_selector_rank2_selectorsoft.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_selector_rank3_bestglobal.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_selector_rank4_selectorconf045.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_selector_rank5_bestsoftblendw020.csv']`

## Selector CV

| strategy | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- |
| selector_conf0.55 | 0.657333 | 0.022002 | 0.637500 | 0.011658 | 0.007269 | 0.651833 |
| selector_soft | 0.656833 | 0.021014 | 0.638000 | 0.011644 | 0.007260 | 0.651580 |
| best_global | 0.656667 | 0.021554 | 0.637500 | 0.011665 | 0.007278 | 0.651278 |
| selector_conf0.45 | 0.656667 | 0.022228 | 0.636500 | 0.011647 | 0.007254 | 0.651110 |
| best_soft_blend_w0.20 | 0.656500 | 0.021442 | 0.638000 | 0.011660 | 0.007271 | 0.651140 |
| best_soft_blend_w0.50 | 0.656333 | 0.021014 | 0.637500 | 0.011653 | 0.007275 | 0.651080 |
| selector_conf0.35 | 0.656333 | 0.022684 | 0.635500 | 0.011638 | 0.007245 | 0.650662 |
| best_soft_blend_w0.35 | 0.656167 | 0.021061 | 0.637500 | 0.011657 | 0.007276 | 0.650901 |
| selector_hard | 0.655333 | 0.021014 | 0.636500 | 0.011639 | 0.007242 | 0.650080 |

## 후보별 OOF 성능과 label 분포

| idx | candidate | forward | side | up | label_count | oof_hit | oof_mean_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | side104_up094 | 1.020000 | 1.040000 | 0.940000 | 64 | 0.655900 | 0.011645 |
| 0 | base_f102_s100_u100 | 1.020000 | 1.000000 | 1.000000 | 1343 | 0.655700 | 0.011645 |
| 1 | prev_f102_s104_u096 | 1.020000 | 1.040000 | 0.960000 | 0 | 0.655500 | 0.011647 |
| 2 | best_f102_s106_u094 | 1.020000 | 1.060000 | 0.940000 | 0 | 0.655500 | 0.011648 |
| 3 | side108_up092 | 1.020000 | 1.080000 | 0.920000 | 0 | 0.655200 | 0.011650 |
| 11 | forward103_side108_up092 | 1.030000 | 1.080000 | 0.920000 | 1550 | 0.655200 | 0.011706 |
| 8 | forward103_side106_up094 | 1.030000 | 1.060000 | 0.940000 | 13 | 0.654900 | 0.011704 |
| 5 | side106_up096 | 1.020000 | 1.060000 | 0.960000 | 32 | 0.654800 | 0.011651 |
| 6 | forward103_side104_up096 | 1.030000 | 1.040000 | 0.960000 | 1652 | 0.654800 | 0.011702 |
| 10 | side110_up090 | 1.020000 | 1.100000 | 0.900000 | 967 | 0.654600 | 0.011653 |
| 7 | forward101_side104_up096 | 1.010000 | 1.040000 | 0.960000 | 2250 | 0.654100 | 0.011629 |
| 9 | forward101_side106_up094 | 1.010000 | 1.060000 | 0.940000 | 2129 | 0.653800 | 0.011630 |

## 해석

- 고정 multiplier best 주변 후보를 샘플별로 고르는 selector/routing 실험입니다.
- train OOF direct-step 예측에서 후보별 거리를 계산해 oracle best 후보를 라벨로 만들었습니다.
- public에서 selector가 실패하면, 후보 선택이 과적합이라는 뜻이므로 다음은 smoothing/denoising 또는 per-sample physics coefficient 축으로 이동하는 것이 좋습니다.
