# 2026-05-08 direct-step target 전환 실험 정리

## 요약

- 기존 최고 Public LB는 `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv = 0.67220`였습니다.
- 오전에는 motion regime별로 모델을 나누는 router 축을 실험했지만, public 개선폭은 작았습니다.
- 이후 residual 보정 축을 일부 내려놓고, `+80ms 미래 step`을 마지막 속도 기준 local frame에서 직접 예측하는 direct-step target을 실험했습니다.
- 순수 direct-step LGBM boundary 후보가 Public LB `0.67800`을 기록하면서 새 최고점을 만들었습니다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `regime_hit_rank1_globalhitweighted_a0.00_f0.56_s0.58_u0.70.csv` | 0.67300 | regime routing 자체보다는 global shrink 변형에 가까운 안전 후보 |
| `regime_hit_rank2_globalhitweighted_a0.00_f0.52_s0.58_u0.78.csv` | 0.67260 | 기존 앵커 근처에서 정체 |
| `direct_step_rank1_cadeltascaledcatboundary_f0.52_s0.58_u0.78.csv` | 0.67100 | CV는 강했지만 public에서는 과적합 또는 분포 불일치 가능성 |
| `direct_step_rank2_cadeltascaledlgbmboundary_f0.52_s0.58_u0.78.csv` | 0.67340 | scale-normalized CA residual이 소폭 개선 |
| `direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv` | **0.67800** | residual anchor를 벗어난 순수 direct-step target이 새 최고점 |

## 실험 축 1: regime hit-weighted router

- speed ratio, turn cosine, 전역 z축 vertical ratio로 motion regime을 나눴습니다.
- regime별 모델을 별도로 학습한 뒤 global 모델과 `route_alpha`로 blending했습니다.
- 의미 있는 전역 z 기준으로 정정한 뒤에는 regime routing이 global hit-weighted 모델을 이기지 못했습니다.
- 결론적으로 regime별 독립 모델은 현재 우선순위에서 내리고, routing보다 target 설계 변경을 더 밀기로 했습니다.

## 실험 축 2: direct-step geometry

- 기존 방식은 constant-acceleration physics 예측값을 anchor로 두고 local-frame residual을 보정했습니다.
- 새 방식은 마지막 관측 좌표 기준 `+80ms displacement`를 직접 예측하는 target을 추가했습니다.
- target을 sample별 motion scale로 나눠 정규화한 뒤, 추론 시 다시 scale을 곱해 좌표로 복원했습니다.
- LGBM direct-step boundary 후보가 CV에서는 최상위는 아니었지만 public에서 가장 강했습니다.

## 핵심 인사이트

- CV 1등이던 CatBoost CA-scaled 후보는 public에서 `0.67100`으로 밀렸습니다.
- public에서는 오히려 더 독립적인 pure direct-step target이 `0.67800`으로 크게 올랐습니다.
- 이는 현재 leaderboard 분포에서 residual 보정만으로는 한계가 있고, target coordinate 자체를 바꾸는 축이 더 큰 돌파를 만들 수 있음을 시사합니다.
- 다음 연구는 direct-step branch를 중심으로 seed ensemble, CatBoost/LGBM blend, hit-boundary weighting 재설계, direct-step 전용 shrink/scale grid를 확장하는 방향이 좋습니다.

## 생성한 주요 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_regime_hit_weighted_router.py` | motion regime별 hit-weighted router 실험 |
| `reports/latest_regime_hit_weighted_router.md` | regime router CV/제출 후보 기록 |
| `scripts/run_direct_step_geometry.py` | direct-step target, scale-normalized CA residual, CatBoost/LGBM 비교 |
| `reports/latest_direct_step_geometry.md` | direct-step geometry 실험 결과 |
| `experiments/public_scores.csv` | public score 누적 기록 |

## 다음 방향

1. direct-step branch를 메인 축으로 승격합니다.
2. `direct_step_pure_lgbmboundary`의 seed 수와 shrink/multiplier grid를 확장합니다.
3. direct-step LGBM과 기존 0.6722 hit-weighted anchor의 selective blend를 실험합니다.
4. CatBoost는 CV-public 괴리가 커서 단독보다 diversity blend 재료로만 사용합니다.
5. pure direct-step에서 hit boundary weight center와 sigma를 다시 탐색합니다.
