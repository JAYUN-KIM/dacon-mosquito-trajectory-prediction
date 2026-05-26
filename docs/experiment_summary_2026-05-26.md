# 2026-05-26 recursive one-step gate 돌파 정리

## 요약

- 최고 Public LB를 `0.69140`에서 `0.69180`으로 갱신했다.
- 성공 파일: `recstep_rank4_gate_osc89b005late_f100s100u100_top080_b40.csv`
- 핵심 축: `+40ms` one-step dynamics를 학습한 뒤 test에서 두 번 재귀 적용하고, gain selector가 고른 top 8% 샘플에만 40% 이동한다.
- global blend 후보들은 public에서 실패했다. 사용자 피드백 기준 "나머지는 꽝"이었고, 정확 점수는 별도 기록하지 않았다.
- 결론: recursive one-step 예측값 전체가 강한 것이 아니라, `움직여도 되는 소수 샘플을 고르는 gate`가 점수를 만들었다.

## 오늘의 문제의식

5/25까지 `t52` alpha-down, plateau disagreement, hit-mode retrieval이 모두 `0.69140` 근처에서 포화됐다.
따라서 기존 curvature gate의 threshold/alpha를 더 만지는 대신, 대회 정의를 다시 `+80ms 미래 위치` 문제로 보고 내부 관측 구간의 `+40ms transition`을 더 많이 활용하는 방향을 실험했다.

## 실험 1: recursive one-step dynamics

`scripts/run_recursive_onestep_dynamics_20260526.py`

- train 내부의 관측 좌표에서 cutoff별 `현재 11프레임 -> 다음 40ms` one-step pseudo sample을 만들었다.
- one-step 모델을 test에서 두 번 굴려 `0ms -> +40ms -> +80ms` 형태로 예측했다.
- 단독 recursive 예측은 OOF에서 champion보다 약했다.
- 하지만 champion과 recursive 예측의 차이를 gain selector로 제한 적용하는 후보를 만들었다.

## Public 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `recstep_rank4_gate_osc89b005late_f100s100u100_top080_b40.csv` | **0.69180** | gain selector top 8% 샘플만 40% 이동해 새 최고점 |
| recursive one-step global blend 후보들 | 미기록 | 사용자 피드백상 실패, 전체 이동은 위험 |

## 실험 2: winner 주변 gate refine

`scripts/make_recursive_onestep_gate_refine_20260526.py`

성공한 `top080_b40` 구조를 보존하고 아래 축만 흔들었다.

- 선택 비율: top 5%, 6%, 8%, 10%, 12%
- 이동 강도: 35%, 40%, 45%, 50%
- multiplier: winner `f100s100u100` 중심, side/up tilt 1개 보조 확인

생성 후보:

| 우선순위 | 제출 파일 | 의도 |
|---:|---|---|
| 1 | `recstepgate_refine_rank1_top08_b45_f100s100u100_top080_b45.csv` | 기존 top 8% 유지, 강도 40% -> 45% |
| 2 | `recstepgate_refine_rank2_top06_b40_f100s100u100_top060_b40.csv` | 선택 영역을 top 6%로 좁힘 |
| 3 | `recstepgate_refine_rank3_top10_b40_f100s100u100_top100_b40.csv` | 선택 영역을 top 10%로 넓힘 |
| 4 | `recstepgate_refine_rank4_top08_b35_f100s100u100_top080_b35.csv` | 이동 강도를 35%로 낮춤 |
| 5 | `recstepgate_refine_rank5_top08_b50_f100s100u100_top080_b50.csv` | 이동 강도를 50%로 높임 |

## 다음 판단 규칙

- rank1이 오르면 `top 8%`는 유지하고 강도를 `0.45~0.55`로 더 탐색한다.
- rank2가 오르면 선택 영역을 `5~7%`로 좁힌다.
- rank3이 오르면 선택 영역을 `10~12%`로 넓힌다.
- 모두 하락하거나 동률이면 `top080_b40`을 새 champion으로 두고 gate feature 또는 gain selector 학습 방식을 바꾼다.
- global blend, 전체 recursive 적용, 기존 alpha/gate 미세조정은 우선순위에서 내린다.

## 인사이트

- 이번 개선은 새로운 좌표 보정량 자체보다 sample-wise 선택이 만든 개선이다.
- OOF에서 raw recursive 모델은 champion보다 약했지만, champion이 틀릴 가능성이 높은 일부 샘플에서는 보완 신호가 있었다.
- Public에서 global blend가 실패하고 gate 후보만 오른 것은, `소수 샘플 교정` 전략이 현재 leaderboard에서 가장 안전하다는 신호다.
- 다음 0.7 돌파 후보는 더 강한 예측 모델보다 더 정확한 `hit rescue gate`일 가능성이 높다.
