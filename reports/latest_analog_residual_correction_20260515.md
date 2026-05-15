# 2026-05-15 Analog Residual Correction

- created_at: `2026-05-15T21:47:02`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `direct_selector_rank2_selectorsoft.csv = 0.68440`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogres_rank1_k64s010.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogres_rank2_k96s015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogres_rank3_k128s010.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogres_rank4_k32s010.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\analogres_rank5_k48s015.csv']`

## 외부 자료/논문 검토

- DACON 규칙상 외부 데이터 사용은 허용되지만, 제공 test 데이터는 어떤 형태로도 학습에 활용하면 안 되며 원격 API 기반 모델도 사용할 수 없다.
- 모기/곤충 추적 문헌은 Kalman 계열의 운동 연속성, 속도/가속도 기반 예측, 유사 궤적 기반 보정 아이디어를 반복적으로 사용한다.
- 이번 실험은 외부 궤적 데이터를 직접 섞지 않고, 논문 아이디어만 train-only analog residual prior로 구현했다.
- 외부 모기 궤적 데이터는 센서, 좌표계, 실험 환경이 달라 private generalization을 해칠 위험이 커서 보류한다.
- 참고: https://dacon.io/competitions/official/236716/overview/rules
- 참고: https://arxiv.org/abs/2505.13615
- 참고: https://www.mdpi.com/2079-9292/14/7/1333
- 참고: https://arxiv.org/abs/2007.14216

## Public 제출 결과

| submission | Public LB | 판단 |
| --- | ---: | --- |
| `analogres_rank1_k64s010.csv` | 0.68300 | OOF 1위였지만 public에서는 약함 |
| `analogres_rank2_k96s015.csv` | 0.68300 | correction 강도 증가도 개선 없음 |
| `analogres_rank3_k128s010.csv` | 0.68360 | analog 계열 중 가장 낫지만 best 0.68440에는 미달 |

결론: analog residual correction은 새 축으로는 의미가 있었지만, 현 public split에서는 selector-soft anchor보다 약하므로 주력 축에서 내린다.

## CV

| config | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | anchor_r_hit | delta_hit_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k64_s010 | 0.657667 | 0.021227 | 0.639500 | 0.011642 | 0.007241 | 0.656400 | 0.001267 |
| k96_s015 | 0.656833 | 0.021842 | 0.638500 | 0.011642 | 0.007248 | 0.656400 | 0.000433 |
| k128_s010 | 0.656833 | 0.022496 | 0.636500 | 0.011643 | 0.007236 | 0.656400 | 0.000433 |
| k32_s010 | 0.655833 | 0.020040 | 0.639000 | 0.011643 | 0.007246 | 0.656400 | -0.000567 |
| k48_s015 | 0.655833 | 0.020984 | 0.639500 | 0.011643 | 0.007255 | 0.656400 | -0.000567 |
| k64_s020 | 0.655500 | 0.021500 | 0.638000 | 0.011644 | 0.007252 | 0.656400 | -0.000900 |
| k128_s020 | 0.654833 | 0.022991 | 0.634000 | 0.011643 | 0.007262 | 0.656400 | -0.001567 |
| k32_s020 | 0.654500 | 0.019944 | 0.639000 | 0.011649 | 0.007256 | 0.656400 | -0.001900 |

## Outputs

| rank | submission | name | k | shrink | power | boundary_amp | vs_soft_anchor_mean_delta | vs_soft_anchor_median_delta | vs_soft_anchor_p95_delta | vs_soft_anchor_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | analogres_rank1_k64s010.csv | k64_s010 | 64 | 0.100000 | 1.000000 | 2.500000 | 0.000141 | 0.000127 | 0.000285 | 0.000691 |
| 2 | analogres_rank2_k96s015.csv | k96_s015 | 96 | 0.150000 | 0.750000 | 3.000000 | 0.000178 | 0.000164 | 0.000345 | 0.000848 |
| 3 | analogres_rank3_k128s010.csv | k128_s010 | 128 | 0.100000 | 0.750000 | 3.000000 | 0.000108 | 0.000100 | 0.000210 | 0.000459 |
| 4 | analogres_rank4_k32s010.csv | k32_s010 | 32 | 0.100000 | 1.000000 | 1.500000 | 0.000200 | 0.000171 | 0.000444 | 0.001281 |
| 5 | analogres_rank5_k48s015.csv | k48_s015 | 48 | 0.150000 | 1.000000 | 2.500000 | 0.000236 | 0.000212 | 0.000480 | 0.001320 |

## 메모

- 새 비모수 축이다. 정규화된 최근 motion pattern이 비슷한 train 샘플을 찾고, 해당 샘플들의 OOF selector-soft 잔차를 작게 빌려온다.
- public-best anchor가 이미 강하고 평가 지표가 1cm hit threshold이므로 correction은 강하게 shrink했다.
