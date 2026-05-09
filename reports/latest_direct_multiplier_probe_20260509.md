# 2026-05-09 Direct-Step Multiplier Probe

- 생성 시각: `2026-05-09T21:09:48`
- 데이터 경로: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- source submission: `direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv`
- source multiplier: `(1.02, 1.0, 1.0)`
- current public best: `direct_refine_rank2_caa6s0055c0105_f1.02_s1.04_u0.96_5seed.csv = 0.68260`
- 생성한 제출 파일: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank2_fromcaa6_f1.02_s1.08_u0.92.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank3_fromcaa6_f1.02_s1.04_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank4_fromcaa6_f1.02_s1.06_u0.96.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank5_fromcaa6_f1.03_s1.04_u0.96.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank6_fromcaa6_f1.01_s1.04_u0.96.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank7_fromcaa6_f1.02_s1.02_u0.98.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\direct_micro_rank8_fromcaa6_f1.03_s1.06_u0.94.csv']`

## Probe 후보

| rank | submission | forward_mult | side_mult | up_mult | mean_delta | median_delta | p95_delta | max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv | 1.020000 | 1.060000 | 0.940000 | 0.000054 | 0.000035 | 0.000161 | 0.000933 |
| 2 | direct_micro_rank2_fromcaa6_f1.02_s1.08_u0.92.csv | 1.020000 | 1.080000 | 0.920000 | 0.000107 | 0.000070 | 0.000323 | 0.001866 |
| 3 | direct_micro_rank3_fromcaa6_f1.02_s1.04_u0.94.csv | 1.020000 | 1.040000 | 0.940000 | 0.000030 | 0.000020 | 0.000082 | 0.000928 |
| 4 | direct_micro_rank4_fromcaa6_f1.02_s1.06_u0.96.csv | 1.020000 | 1.060000 | 0.960000 | 0.000037 | 0.000020 | 0.000130 | 0.000907 |
| 5 | direct_micro_rank5_fromcaa6_f1.03_s1.04_u0.96.csv | 1.030000 | 1.040000 | 0.960000 | 0.000467 | 0.000419 | 0.001054 | 0.001142 |
| 6 | direct_micro_rank6_fromcaa6_f1.01_s1.04_u0.96.csv | 1.010000 | 1.040000 | 0.960000 | 0.000467 | 0.000419 | 0.001054 | 0.001142 |
| 7 | direct_micro_rank7_fromcaa6_f1.02_s1.02_u0.98.csv | 1.020000 | 1.020000 | 0.980000 | 0.000054 | 0.000035 | 0.000161 | 0.000933 |
| 8 | direct_micro_rank8_fromcaa6_f1.03_s1.06_u0.94.csv | 1.030000 | 1.060000 | 0.940000 | 0.000476 | 0.000429 | 0.001057 | 0.001201 |

## 해석

- 기존 `0.6824` source prediction에서 local-step을 복원해, 재학습 없이 multiplier만 바꾼 후보입니다.
- 현재 best인 `s1.04/u0.96` 주변에서 side를 더 키우고 up을 줄이는 방향, forward를 소폭 올리는 방향을 동시에 확인합니다.
- 남은 제출 수가 제한적이면 `rank1 f1.02/s1.06/u0.94`와 `rank5 f1.03/s1.04/u0.96`을 우선 추천합니다.
