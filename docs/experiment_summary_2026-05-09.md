# 2026-05-09 direct-step refine 실험 정리

## 요약

- 전날 최고 Public LB는 `direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv = 0.67800`이었습니다.
- 오늘은 새 축을 바로 버리지 않고, pure direct-step branch를 5seed, weight function, axis multiplier 방향으로 공격적으로 확장했습니다.
- `ca_a6_s0055_c0105` direct-step weight와 axis multiplier 조합에서 Public LB `0.68260`까지 크게 올랐습니다.
- 이후 재학습 없이 multiplier만 미세 조정한 probe에서 `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv = 0.68300`으로 현재 최고점을 갱신했습니다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `direct_refine_oldweight_caa5_f1.02_s1.00_u1.00_5seed.csv` | 0.67860 | 기존 0.678 weight를 5seed로 늘린 안정 개선 |
| `direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv` | 0.68240 | 더 넓은 CA-boundary weight가 큰 개선 |
| `direct_refine_rank2_caa6s0055c0105_f1.02_s1.04_u0.96_5seed.csv` | 0.68260 | side를 키우고 up을 줄인 axis multiplier tilt가 추가 개선 |
| `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv` | **0.68300** | 현재 최고점, best 주변 multiplier micro-probe |
| `direct_micro_rank5_fromcaa6_f1.03_s1.04_u0.96.csv` | 0.68260 | forward만 올린 후보는 기존 best와 동률 |

## 실험 축

1. 5seed direct-step refine
   - 기존 3seed 후보를 5seed로 확장했습니다.
   - old weight는 `0.67860`으로 안정 개선에 그쳤고, 새 `ca_a6_s0055_c0105` weight가 큰 폭으로 올랐습니다.

2. CA-boundary weight 재설계
   - 기존 center `0.0100`, sigma `0.0045`, amplitude `5.0`보다 넓고 강한 `center=0.0105`, `sigma=0.0055`, `amplitude=6.0`이 public에서 더 좋았습니다.
   - direct-step target에서는 residual branch와 다른 weighting optimum이 있는 것으로 보입니다.

3. Axis multiplier micro-probe
   - `f1.02/s1.04/u0.96` best 주변에서 재학습 없이 multiplier만 바꿨습니다.
   - `s1.06/u0.94`가 `0.68300`으로 올라, side 방향을 더 키우고 up 방향을 더 줄이는 패턴이 유효했습니다.

## 인사이트

- 오늘 개선은 모델 구조를 크게 바꾼 것이 아니라, direct-step branch 안에서 weight와 축별 multiplier를 맞춘 결과입니다.
- 하지만 `0.6826 -> 0.6830` 이후 개선폭이 다시 작아졌기 때문에, 다음 실험은 이 축의 미세 조정보다 새 축 탐색 비중을 높이는 편이 좋아 보입니다.
- 다음 새 축 후보는 `direct-step confidence routing`, `per-sample physics coefficient prediction`, `sequence denoising/smoothing`, `test-time trajectory regime calibration`입니다.

## 생성한 주요 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_direct_step_refine_20260509.py` | direct-step 5seed, weight spec, multiplier grid 확장 |
| `reports/latest_direct_step_refine_20260509.md` | 1차 direct-step refine CV/후보 기록 |
| `scripts/make_direct_step_multiplier_probe_20260509.py` | 0.6826 주변 multiplier micro-probe 후보 생성 |
| `reports/latest_direct_multiplier_probe_20260509.md` | micro-probe 후보와 best 대비 이동량 기록 |
| `experiments/public_scores.csv` | public score 누적 기록 |

## 다음 방향

1. direct-step branch는 현재 앵커로 유지합니다.
2. 다음 실험 5개 중 2개만 direct-step micro refine에 쓰고, 3개는 새 축 탐색으로 돌리는 것이 좋습니다.
3. 새 축은 per-sample 계수 예측이나 confidence routing처럼 기존 direct-step과 다른 오류 샘플을 맞출 수 있는 방향을 우선합니다.
