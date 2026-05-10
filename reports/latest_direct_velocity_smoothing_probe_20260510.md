# 2026-05-10 Direct Velocity Smoothing Probe

- 생성 시각: `2026-05-10T17:06:49`
- 데이터 경로: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- source submission: `direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv`
- current best: `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv = 0.68300`
- multiplier: `(1.02, 1.06, 0.94)`
- 생성한 제출 파일: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_smooth_rank1_w631rawscale.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_smooth_rank2_w532rawscale.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_smooth_rank3_w631mixscale.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_smooth_rank4_w4321rawscale.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_smooth_rank5_w532mixscale.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_smooth_rank6_w4321mixscale.csv']`
- OOF CV proxy: `enabled`

## OOF CV Proxy

| smooth | weights | scale_mode | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | delta_hit_vs_current | delta_mean_distance_vs_current |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_best_basis | - | original | 0.011648 | 0.007144 | 0.024535 | 0.039701 | 0.655500 | 0.000000 | 0.000000 |
| w631_rawscale | [0.6, 0.3, 0.1] | raw | 0.012582 | 0.007402 | 0.027253 | 0.045570 | 0.629800 | -0.025700 | 0.000934 |
| w532_rawscale | [0.5, 0.3, 0.2] | raw | 0.013329 | 0.007744 | 0.029518 | 0.049238 | 0.609500 | -0.046000 | 0.001681 |
| w631_mixscale | [0.6, 0.3, 0.1] | mix | 0.013204 | 0.008014 | 0.028984 | 0.047708 | 0.598400 | -0.057100 | 0.001556 |
| w4321_rawscale | [0.4, 0.3, 0.2, 0.1] | raw | 0.014307 | 0.008195 | 0.032668 | 0.052672 | 0.585000 | -0.070500 | 0.002658 |
| w532_mixscale | [0.5, 0.3, 0.2] | mix | 0.013726 | 0.008223 | 0.030553 | 0.049704 | 0.584600 | -0.070900 | 0.002078 |
| w4321_mixscale | [0.4, 0.3, 0.2, 0.1] | mix | 0.014458 | 0.008542 | 0.033040 | 0.052455 | 0.567400 | -0.088100 | 0.002809 |

## 제출 후보

| rank | submission | smooth | weights | scale_mode | mean_delta | median_delta | p95_delta | max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | direct_smooth_rank1_w631rawscale.csv | w631_rawscale | [0.6, 0.3, 0.1] | raw | 0.003769 | 0.001952 | 0.013200 | 0.125137 |
| 2 | direct_smooth_rank2_w532rawscale.csv | w532_rawscale | [0.5, 0.3, 0.2] | raw | 0.005096 | 0.002516 | 0.018556 | 0.151419 |
| 3 | direct_smooth_rank3_w631mixscale.csv | w631_mixscale | [0.6, 0.3, 0.1] | mix | 0.005266 | 0.003201 | 0.016397 | 0.144365 |
| 4 | direct_smooth_rank4_w4321rawscale.csv | w4321_rawscale | [0.4, 0.3, 0.2, 0.1] | raw | 0.006610 | 0.003199 | 0.025008 | 0.177555 |
| 5 | direct_smooth_rank5_w532mixscale.csv | w532_mixscale | [0.5, 0.3, 0.2] | mix | 0.006144 | 0.003583 | 0.020705 | 0.124034 |
| 6 | direct_smooth_rank6_w4321mixscale.csv | w4321_mixscale | [0.4, 0.3, 0.2, 0.1] | mix | 0.007166 | 0.003933 | 0.025615 | 0.146101 |

## 해석

- direct-step local prediction은 유지하고, local frame의 forward 방향만 최근 velocity 평균으로 부드럽게 바꾼 test-time probe입니다.
- 관측 마지막 step의 방향 노이즈가 크다면 개선될 수 있지만, 현재 best 대비 이동량이 큰 후보는 public 리스크도 큽니다.
- OOF CV proxy가 current_best_basis보다 낮으면 제출 우선순위를 낮추고, 높거나 비슷하면서 이동량이 작은 후보를 먼저 제출합니다.
