# Direct Step Geometry 실험

- 생성 시각: `2026-05-08T01:52:26`
- 데이터 경로: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- 실험 전 Public 앵커: `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv = 0.6722`
- feature 수: `752`
- CV seed: `[42]`
- 전체 학습 ensemble seed: `[42, 777, 2026]`
- CV best: `ca_delta_scaled_cat_boundary`, hit=`0.676000`, mult=(`0.52`, `0.58`, `0.78`)
- 생성한 제출 파일: `direct_step_rank1~rank5`
- 추가 생성한 pure direct-step 후보: `C:\open\dacon-mosquito-trajectory-prediction\submissions\direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv`
- public 결과: CatBoost CA-scaled `0.67100`, LGBM CA-scaled `0.67340`, pure direct-step `0.67800`

## Top 30 CV 결과

| experiment | base_kind | scaled_target | model_kind | weight_kind | forward_mult | side_mult | up_mult | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ca_delta_scaled_cat_boundary | ca | True | catboost | boundary | 0.520000 | 0.580000 | 0.780000 | 0.676000 | nan | 0.676000 | 0.011817 | 0.006969 | 0.676000 |
| ca_delta_scaled_lgbm_boundary | ca | True | lgbm | boundary | 0.520000 | 0.580000 | 0.780000 | 0.674500 | nan | 0.674500 | 0.011617 | 0.006981 | 0.674500 |
| ca_delta_scaled_cat_boundary | ca | True | catboost | boundary | 0.560000 | 0.580000 | 0.700000 | 0.674500 | nan | 0.674500 | 0.011801 | 0.006964 | 0.674500 |
| ca_delta_scaled_lgbm_boundary | ca | True | lgbm | boundary | 0.520000 | 0.600000 | 0.700000 | 0.673500 | nan | 0.673500 | 0.011626 | 0.007024 | 0.673500 |
| cv_delta_scaled_lgbm_boundary | cv | True | lgbm | boundary | 0.560000 | 0.580000 | 0.700000 | 0.673000 | nan | 0.673000 | 0.011573 | 0.007070 | 0.673000 |
| ca_delta_scaled_lgbm_boundary | ca | True | lgbm | boundary | 0.560000 | 0.580000 | 0.700000 | 0.673000 | nan | 0.673000 | 0.011599 | 0.007026 | 0.673000 |
| ca_delta_scaled_lgbm_boundary | ca | True | lgbm | boundary | 0.520000 | 0.580000 | 0.700000 | 0.673000 | nan | 0.673000 | 0.011629 | 0.007008 | 0.673000 |
| ca_delta_scaled_lgbm_boundary | ca | True | lgbm | boundary | 0.480000 | 0.580000 | 0.700000 | 0.673000 | nan | 0.673000 | 0.011662 | 0.007005 | 0.673000 |
| cv_delta_scaled_lgbm_boundary | cv | True | lgbm | boundary | 0.520000 | 0.580000 | 0.780000 | 0.672500 | nan | 0.672500 | 0.011605 | 0.007038 | 0.672500 |
| cv_delta_scaled_lgbm_boundary | cv | True | lgbm | boundary | 0.520000 | 0.600000 | 0.700000 | 0.672500 | nan | 0.672500 | 0.011608 | 0.007081 | 0.672500 |
| ca_delta_scaled_lgbm_boundary | ca | True | lgbm | boundary | 0.460000 | 0.580000 | 0.700000 | 0.672500 | nan | 0.672500 | 0.011681 | 0.007002 | 0.672500 |
| direct_step_scaled_lgbm_boundary | last | True | lgbm | boundary | 1.020000 | 1.000000 | 1.000000 | 0.672000 | nan | 0.672000 | 0.011568 | 0.006801 | 0.672000 |
| ca_delta_scaled_cat_boundary | ca | True | catboost | boundary | 0.520000 | 0.600000 | 0.700000 | 0.672000 | nan | 0.672000 | 0.011828 | 0.006995 | 0.672000 |
| ca_delta_scaled_cat_boundary | ca | True | catboost | boundary | 0.480000 | 0.580000 | 0.700000 | 0.672000 | nan | 0.672000 | 0.011864 | 0.007035 | 0.672000 |
| cv_delta_scaled_lgbm_boundary | cv | True | lgbm | boundary | 0.520000 | 0.580000 | 0.700000 | 0.671500 | nan | 0.671500 | 0.011608 | 0.007073 | 0.671500 |
| ca_delta_scaled_lgbm_boundary | ca | True | lgbm | boundary | 0.440000 | 0.580000 | 0.700000 | 0.671500 | nan | 0.671500 | 0.011701 | 0.007045 | 0.671500 |
| ca_delta_scaled_cat_boundary | ca | True | catboost | boundary | 0.520000 | 0.580000 | 0.700000 | 0.671500 | nan | 0.671500 | 0.011831 | 0.006992 | 0.671500 |
| ca_delta_scaled_cat_boundary | ca | True | catboost | boundary | 0.460000 | 0.580000 | 0.700000 | 0.671500 | nan | 0.671500 | 0.011881 | 0.007070 | 0.671500 |
| ca_delta_scaled_cat_boundary | ca | True | catboost | boundary | 0.440000 | 0.580000 | 0.700000 | 0.671500 | nan | 0.671500 | 0.011899 | 0.007096 | 0.671500 |
| cv_delta_scaled_lgbm_boundary | cv | True | lgbm | boundary | 0.480000 | 0.580000 | 0.700000 | 0.671000 | nan | 0.671000 | 0.011649 | 0.007100 | 0.671000 |
| cv_delta_scaled_lgbm_boundary | cv | True | lgbm | boundary | 0.460000 | 0.580000 | 0.700000 | 0.668000 | nan | 0.668000 | 0.011670 | 0.007089 | 0.668000 |
| cv_delta_scaled_lgbm_boundary | cv | True | lgbm | boundary | 0.440000 | 0.580000 | 0.700000 | 0.668000 | nan | 0.668000 | 0.011693 | 0.007106 | 0.668000 |
| direct_step_scaled_lgbm_boundary | last | True | lgbm | boundary | 1.000000 | 1.000000 | 1.000000 | 0.664000 | nan | 0.664000 | 0.011572 | 0.006969 | 0.664000 |
| direct_step_scaled_lgbm_boundary | last | True | lgbm | boundary | 1.040000 | 1.000000 | 1.000000 | 0.664000 | nan | 0.664000 | 0.011716 | 0.006906 | 0.664000 |
| direct_step_scaled_lgbm_boundary | last | True | lgbm | boundary | 1.000000 | 1.050000 | 1.050000 | 0.663000 | nan | 0.663000 | 0.011590 | 0.006948 | 0.663000 |
| direct_step_scaled_cat_uniform | last | True | catboost | uniform | 1.040000 | 1.000000 | 1.000000 | 0.663000 | nan | 0.663000 | 0.011777 | 0.007033 | 0.663000 |
| direct_step_scaled_lgbm_boundary | last | True | lgbm | boundary | 0.980000 | 1.000000 | 1.000000 | 0.656500 | nan | 0.656500 | 0.011719 | 0.007289 | 0.656500 |
| direct_step_scaled_lgbm_boundary | last | True | lgbm | boundary | 0.980000 | 0.950000 | 0.950000 | 0.655500 | nan | 0.655500 | 0.011703 | 0.007273 | 0.655500 |
| direct_step_scaled_cat_uniform | last | True | catboost | uniform | 1.020000 | 1.000000 | 1.000000 | 0.651000 | nan | 0.651000 | 0.011752 | 0.007154 | 0.651000 |
| direct_step_scaled_cat_uniform | last | True | catboost | uniform | 1.000000 | 1.000000 | 1.000000 | 0.640500 | nan | 0.640500 | 0.011862 | 0.007432 | 0.640500 |

## 해석

- residual 보정만 계속 밀지 않고, `+80ms 미래 step` 자체를 local frame에서 직접 예측하는 target 전환 실험입니다.
- `direct_step_*`는 마지막 관측 좌표 기준 displacement를 직접 예측합니다.
- `cv_delta_*`, `ca_delta_*`는 scale-normalized target을 유지하면서 물리 origin을 쓰는 편이 나은지 비교합니다.
- CV 1등은 CatBoost CA-scaled였지만 public은 `0.67100`으로 약했습니다.
- public에서 가장 강했던 후보는 순수 direct-step LGBM boundary였고, `0.67800`으로 새 최고점을 만들었습니다.
