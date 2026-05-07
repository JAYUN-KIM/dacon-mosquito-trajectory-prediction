# Regime Hit-Weighted Router 실험

- 생성 시각: `2026-05-08T01:30:34`
- 데이터 경로: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Public 앵커: `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv = 0.6722`
- feature 수: `752`
- hit-boundary weight 설정: `('base_a5_s0045', 'base', 5.0, 0.0045, 0.01)`
- CV seed: `[42]`
- 전체 학습 ensemble seed: `[42, 777, 2026]`
- regime blend alpha 후보: `[0.25, 0.5, 0.75, 1.0]`
- 축별 shrink 후보: `[(0.46, 0.58, 0.7), (0.48, 0.58, 0.7), (0.52, 0.58, 0.7), (0.52, 0.6, 0.7), (0.52, 0.58, 0.78), (0.56, 0.58, 0.7)]`
- regime threshold: `{'speed_high': 1.2178669866149385, 'speed_low': 0.8790600002207473, 'sharp_turn': 0.980940642809788, 'vertical_high': 0.5621494484835731}`
- candidate feature family: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- CV best: `global_hit_weighted`, hit=`0.674000`, shrink=(`0.56`, `0.58`, `0.70`), alpha=`0.00`
- 생성한 제출 파일: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regime_hit_rank1_globalhitweighted_a0.00_f0.56_s0.58_u0.70.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regime_hit_rank2_globalhitweighted_a0.00_f0.52_s0.58_u0.78.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regime_hit_rank3_globalhitweighted_a0.00_f0.52_s0.60_u0.70.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regime_hit_rank4_globalhitweighted_a0.00_f0.52_s0.58_u0.70.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\regime_hit_rank5_regimerouted_a0.25_f0.56_s0.58_u0.70.csv']`

## Train Regime 분포

| regime | count |
| --- | --- |
| vertical_high | 2500 |
| sharp_turn | 1781 |
| speed_high | 1480 |
| speed_low | 1197 |
| default | 3042 |

## Test Regime 분포

| regime | count |
| --- | --- |
| vertical_high | 2771 |
| sharp_turn | 1741 |
| speed_high | 1442 |
| speed_low | 1142 |
| default | 2904 |

## CV Regime 분포

| seed | regime | count |
| --- | --- | --- |
| 42 | vertical_high | 517 |
| 42 | sharp_turn | 332 |
| 42 | speed_high | 269 |
| 42 | speed_low | 253 |
| 42 | default | 629 |

## Top 30 CV 결과

| strategy | route_alpha | forward_shrink | side_shrink | up_shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_hit_weighted | 0.000000 | 0.560000 | 0.580000 | 0.700000 | 0.674000 | nan | 0.674000 | 0.011587 | 0.007028 | 0.674000 |
| global_hit_weighted | 0.000000 | 0.520000 | 0.580000 | 0.780000 | 0.674000 | nan | 0.674000 | 0.011603 | 0.007000 | 0.674000 |
| global_hit_weighted | 0.000000 | 0.520000 | 0.600000 | 0.700000 | 0.673000 | nan | 0.673000 | 0.011611 | 0.007040 | 0.673000 |
| global_hit_weighted | 0.000000 | 0.520000 | 0.580000 | 0.700000 | 0.673000 | nan | 0.673000 | 0.011614 | 0.007042 | 0.673000 |
| regime_routed | 0.250000 | 0.560000 | 0.580000 | 0.700000 | 0.672000 | nan | 0.672000 | 0.011624 | 0.007050 | 0.672000 |
| regime_routed | 0.250000 | 0.520000 | 0.580000 | 0.780000 | 0.672000 | nan | 0.672000 | 0.011637 | 0.007029 | 0.672000 |
| global_hit_weighted | 0.000000 | 0.480000 | 0.580000 | 0.700000 | 0.672000 | nan | 0.672000 | 0.011646 | 0.007054 | 0.672000 |
| regime_routed | 0.250000 | 0.520000 | 0.580000 | 0.700000 | 0.672000 | nan | 0.672000 | 0.011650 | 0.007032 | 0.672000 |
| regime_routed | 0.250000 | 0.520000 | 0.600000 | 0.700000 | 0.671500 | nan | 0.671500 | 0.011647 | 0.007038 | 0.671500 |
| regime_routed | 0.500000 | 0.520000 | 0.580000 | 0.700000 | 0.671500 | nan | 0.671500 | 0.011715 | 0.007106 | 0.671500 |
| global_hit_weighted | 0.000000 | 0.460000 | 0.580000 | 0.700000 | 0.671000 | nan | 0.671000 | 0.011664 | 0.007083 | 0.671000 |
| regime_routed | 0.250000 | 0.480000 | 0.580000 | 0.700000 | 0.671000 | nan | 0.671000 | 0.011681 | 0.007078 | 0.671000 |
| regime_routed | 0.500000 | 0.560000 | 0.580000 | 0.700000 | 0.671000 | nan | 0.671000 | 0.011691 | 0.007093 | 0.671000 |
| regime_routed | 0.250000 | 0.460000 | 0.580000 | 0.700000 | 0.671000 | nan | 0.671000 | 0.011699 | 0.007078 | 0.671000 |
| regime_routed | 0.500000 | 0.480000 | 0.580000 | 0.700000 | 0.671000 | nan | 0.671000 | 0.011744 | 0.007108 | 0.671000 |
| regime_routed | 0.500000 | 0.460000 | 0.580000 | 0.700000 | 0.670000 | nan | 0.670000 | 0.011760 | 0.007125 | 0.670000 |
| regime_routed | 0.500000 | 0.520000 | 0.600000 | 0.700000 | 0.669500 | nan | 0.669500 | 0.011713 | 0.007112 | 0.669500 |
| regime_routed | 0.500000 | 0.520000 | 0.580000 | 0.780000 | 0.669000 | nan | 0.669000 | 0.011703 | 0.007113 | 0.669000 |
| regime_routed | 0.750000 | 0.460000 | 0.580000 | 0.700000 | 0.665500 | nan | 0.665500 | 0.011845 | 0.007150 | 0.665500 |
| regime_routed | 0.750000 | 0.520000 | 0.580000 | 0.780000 | 0.665000 | nan | 0.665000 | 0.011798 | 0.007087 | 0.665000 |
| regime_routed | 0.750000 | 0.480000 | 0.580000 | 0.700000 | 0.665000 | nan | 0.665000 | 0.011831 | 0.007140 | 0.665000 |
| regime_routed | 0.750000 | 0.520000 | 0.600000 | 0.700000 | 0.664000 | nan | 0.664000 | 0.011806 | 0.007132 | 0.664000 |
| regime_routed | 0.750000 | 0.560000 | 0.580000 | 0.700000 | 0.663000 | nan | 0.663000 | 0.011788 | 0.007121 | 0.663000 |
| regime_routed | 0.750000 | 0.520000 | 0.580000 | 0.700000 | 0.663000 | nan | 0.663000 | 0.011807 | 0.007139 | 0.663000 |
| regime_routed | 1.000000 | 0.520000 | 0.580000 | 0.780000 | 0.659500 | nan | 0.659500 | 0.011920 | 0.007199 | 0.659500 |
| regime_routed | 1.000000 | 0.460000 | 0.580000 | 0.700000 | 0.659000 | nan | 0.659000 | 0.011955 | 0.007264 | 0.659000 |
| regime_routed | 1.000000 | 0.480000 | 0.580000 | 0.700000 | 0.658500 | nan | 0.658500 | 0.011943 | 0.007249 | 0.658500 |
| regime_routed | 1.000000 | 0.520000 | 0.580000 | 0.700000 | 0.657500 | nan | 0.657500 | 0.011925 | 0.007193 | 0.657500 |
| regime_routed | 1.000000 | 0.520000 | 0.600000 | 0.700000 | 0.657500 | nan | 0.657500 | 0.011926 | 0.007200 | 0.657500 |
| regime_routed | 1.000000 | 0.560000 | 0.580000 | 0.700000 | 0.657000 | nan | 0.657000 | 0.011912 | 0.007188 | 0.657000 |

## 해석

- 하나의 global hit-weighted local-frame 모델이 motion regime별로 너무 거친지 확인한 실험입니다.
- 전역 z축 기준 vertical ratio, turn cosine, speed ratio로 regime을 나누고, regime 모델을 global 모델에 alpha만큼 섞었습니다.
- 의미 있는 regime 기준으로 정정한 뒤에는 regime routing이 global 모델을 이기지 못했습니다.
- 따라서 현재 결론은 regime별 독립 모델보다 hit-boundary weight와 shrink 조합을 더 밀어보는 쪽이 안전합니다.
