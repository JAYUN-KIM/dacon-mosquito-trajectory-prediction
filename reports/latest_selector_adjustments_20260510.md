# 2026-05-10 Selector Adjustment Candidates

- 생성 시각: `2026-05-10T17:14:36`
- base: `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv = 0.68300`
- selector anchor: `direct_selector_rank1_selectorconf055.csv = 0.68340`
- soft reference: `direct_selector_rank2_selectorsoft.csv`
- conf0.45 reference: `direct_selector_rank4_selectorconf045.csv`
- 생성한 제출 파일: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\selector_adjust_rank1_extend115.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\selector_adjust_rank2_shrink075.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\selector_adjust_rank3_softpull015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\selector_adjust_rank4_conf45pull015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\selector_adjust_rank5_extend130.csv']`

## 후보

| rank | submission | note | vs_base_mean_delta | vs_base_median_delta | vs_base_p95_delta | vs_base_max_delta | vs_selector_mean_delta | vs_selector_median_delta | vs_selector_p95_delta | vs_selector_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | selector_adjust_rank1_extend115.csv | public에서 이긴 selector 방향을 15%만 추가로 연장한 1순위 공격 후보 | 0.000043 | 0.000000 | 0.000187 | 0.003132 | 0.000006 | 0.000000 | 0.000024 | 0.000409 |
| 2 | selector_adjust_rank2_shrink075.csv | selector 이동을 75%로 줄인 보수 후보 | 0.000028 | 0.000000 | 0.000122 | 0.002043 | 0.000009 | 0.000000 | 0.000041 | 0.000681 |
| 3 | selector_adjust_rank3_softpull015.csv | selector_conf0.55를 유지하되 soft selector 쪽으로 15% 당긴 후보 | 0.000056 | 0.000015 | 0.000157 | 0.002536 | 0.000022 | 0.000014 | 0.000068 | 0.000187 |
| 4 | selector_adjust_rank4_conf45pull015.csv | route 범위를 조금 넓히는 conf0.45 방향 15% 후보 | 0.000048 | 0.000000 | 0.000185 | 0.002723 | 0.000010 | 0.000000 | 0.000080 | 0.000286 |
| 5 | selector_adjust_rank5_extend130.csv | selector 방향을 30% 더 미는 공격 후보 | 0.000049 | 0.000000 | 0.000211 | 0.003540 | 0.000011 | 0.000000 | 0.000049 | 0.000817 |

## 제출 우선순위

- 1순위는 `selector_adjust_rank1_extend115.csv`입니다. 이미 오른 selector 방향을 아주 작게 더 미는 후보입니다.
- 2순위는 `selector_adjust_rank2_shrink075.csv`입니다. 0.6834가 운 좋게 오른 경우를 대비한 방어형 확인 후보입니다.
- 3순위는 `selector_adjust_rank3_softpull015.csv`입니다. soft selector 신호를 약하게 섞어보는 후보입니다.
- `rank5_extend130`은 이동량은 여전히 작지만 가장 공격적이므로 제출권이 남을 때만 추천합니다.
