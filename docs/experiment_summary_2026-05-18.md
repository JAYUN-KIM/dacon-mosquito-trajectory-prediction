# 2026-05-18 constant-turn curvature breakthrough 정리

## 요약

- temporal-backcast 50% blend 이후 `0.688`대까지 올라왔고, 55% 주변 세밀 탐색에서 `0.6888`까지 개선했다.
- 이후 temporal-backcast strength만으로는 상승폭이 둔화되어 완전히 다른 물리 축을 다시 시도했다.
- 새 축은 최근 속도 방향의 3D 회전량을 추정해 +80ms 동안 같은 방향으로 휘어진다고 보는 `constant-turn curvature correction`이다.
- curvature 단독 예측은 강하지 않았지만, temporal-backcast best 위에 correction을 작게 얹었을 때 public이 `0.6900`까지 상승했다.
- 2026-05-18 기준 새 최고점은 `turncurve_refine_temporalbest_w1tm0p25s0p5d0p98_a09.csv = 0.6900`이다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `temporalbc_refine_r1f102s100u100_w52.csv` | 0.6880 | 50%보다 약간 강한 temporal blend가 추가 개선 |
| `temporalbc_refine_r1f102s100u100_w55.csv` | 0.6888 | temporal-backcast strength 최적점 근처 |
| `temporalbc_refine_avgr1r2_w52.csv` | 0.6882 | temporal 방향 ensemble은 rank1 단독보다 약함 |
| `turncurve_rank1_temporalbest_w1tm0p25s0p5d0p98_a08.csv` | 0.6894 | curvature correction 축 유효성 확인 |
| `turncurve_refine_temporalbest_w1tm0p25s0p5d0p98_a09.csv` | **0.6900** | 5/18 기준 새 최고점 |
| `turncurve_refine_temporalbest_w1tm0p25s0p5d0p98_a10.csv` | 0.6896 | correction을 더 키우면 소폭 하락 |

보조 관찰:

- `temporalbc_refine_truew555_r1f102s100u100.csv = 0.6886`으로, 기존 `w55=0.6888`이 temporal strength 단독 최적점에 가까워 보인다.
- `w57` 계열은 `0.6886`으로 55%보다 약했다.

## 실험 축

1. Temporal-backcast strength refine
   - `w52`, `w55`, `w57`, `truew555`를 통해 temporal-backcast blend 강도를 확인했다.
   - `w55=0.6888`이 가장 강했고, 이후 더 키우면 하락했다.
   - 결론적으로 temporal-backcast 자체는 `55%` 근처에서 거의 포화된 것으로 본다.

2. Constant-turn curvature correction
   - 최근 displacement 방향의 회전 벡터를 계산했다.
   - 최근 1~3개 회전량을 평균해 앞으로 두 40ms step 동안 계속 같은 방향으로 회전한다고 가정했다.
   - 단독 constant-turn 예측은 기존 ML anchor보다 약했지만, `(constant_turn - constant_velocity)` correction을 temporal best에 작게 더했을 때 public이 올랐다.
   - 특히 `rot_window=1`, `turn_scale=-0.25`, `speed_scale=0.5`, `disp_scale=0.98`, `anchor_alpha=0.09`가 가장 좋았다.

## 핵심 인사이트

- temporal-backcast는 큰 방향을 잡고, curvature correction은 그 위에서 휘어지는 샘플을 미세 보정하는 역할을 한다.
- 현재 최고 축은 `temporal-backcast pseudo-supervision + constant-turn curvature correction`이다.
- correction alpha는 `0.08~0.10` 사이에 최적점이 있고, `0.09`가 public에서 가장 좋았다.
- curvature 단독이 강한 것은 아니므로, 반드시 강한 anchor 위에 작은 correction으로 얹어야 한다.
- 다음 실험은 alpha 초정밀 탐색, curvature config 다양화, curvature correction을 적용할 샘플을 고르는 gate 모델 순서가 좋아 보인다.

## 생성 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_constant_turn_curvature_20260518.py` | constant-turn curvature correction 후보 생성 |
| `reports/latest_constant_turn_curvature_20260518.md` | curvature CV, public 결과, 후보 기록 |

## 다음 방향

1. alpha `0.085`, `0.090`, `0.095`를 초정밀하게 만든다.
2. 같은 curvature direction을 `w55` 외에 `w52`, `w57`, selector-soft anchor에도 얹어 비교한다.
3. curvature correction의 크기/회전 신뢰도를 feature로 사용해 “적용할 샘플만 고르는 gate”를 만든다.
4. 다른 turn config도 public에 1~2개만 확인해, `w1_tm0p25_s0p5_d0p98`이 진짜 최선인지 검증한다.
