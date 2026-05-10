# 2026-05-10 selector routing 실험 정리

## 요약

- 전날 최고 Public LB는 `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv = 0.68300`이었습니다.
- 오늘은 완전히 새 축을 찾는 대신, 먼저 direct-step multiplier를 샘플별로 고르는 selector/routing 축을 확인했습니다.
- `direct_selector_rank1_selectorconf055.csv = 0.68340`으로 public에서 실제 개선이 확인됐습니다.
- 이후 selector 방향을 여러 방식으로 조정했고, `selector_adjust_rank4_conf45pull015.csv = 0.68360`으로 현재 최고점을 갱신했습니다.
- 별도 새 축으로 velocity smoothing/local frame denoising도 확인했지만, OOF proxy가 크게 하락해 제출 우선순위에서 제외했습니다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `direct_selector_rank1_selectorconf055.csv` | 0.68340 | selector confidence routing이 기존 direct-step best보다 개선 |
| `selector_adjust_rank1_extend115.csv` | 0.68340 | selector 이동 방향을 15% 더 밀었지만 anchor와 동률 |
| `selector_adjust_rank2_shrink075.csv` | 0.68320 | selector 이동을 줄이면 하락, route 신호를 너무 약하게 만들면 손해 |
| `selector_adjust_rank4_conf45pull015.csv` | **0.68360** | conf0.45 route 방향을 15% 섞은 후보가 현재 최고점 |
| `selector_adjust_rank5_extend130.csv` | 0.68340 | selector 방향을 30% 더 밀어도 추가 개선 없음 |

## 실험 축

1. Direct multiplier selector
   - `f/s/u` axis multiplier 후보를 여러 개 만들고, train OOF에서 oracle best 후보를 라벨로 삼아 LightGBM multiclass selector를 학습했습니다.
   - hard routing은 과적합 위험이 있어 `selector_conf0.55`처럼 confidence가 높은 샘플에만 route를 적용했습니다.
   - Public에서 `0.68300 -> 0.68340`으로 올라 selector 축이 얇게 살아있음을 확인했습니다.

2. Selector adjustment candidates
   - 이미 public에서 오른 selector 후보를 기준으로 `base -> selector` 이동 벡터를 줄이거나 늘렸습니다.
   - 단순 extension은 `0.68340` 동률에 그쳤고, 오히려 `conf0.45` 방향을 아주 약하게 섞은 후보가 `0.68360`으로 가장 좋았습니다.
   - 즉, 개선의 핵심은 이동량 자체보다 "어떤 샘플을 route할지"에 더 가까워 보입니다.

3. Velocity smoothing probe
   - direct-step local prediction은 유지하고, local frame forward 방향만 최근 velocity 평균으로 부드럽게 바꾸는 후보를 만들었습니다.
   - OOF proxy에서 current best basis가 `0.65550`인데, 가장 좋은 smoothing 후보도 `0.62980`에 그쳐 크게 악화됐습니다.
   - 마지막 step 방향 노이즈 제거라는 아이디어는 직관적이지만, 현재 모델은 마지막 velocity basis에 강하게 맞춰져 있어 frame 자체를 바꾸면 hit가 깨지는 것으로 보입니다.

## 인사이트

- selector/routing은 대형 breakthrough 축은 아니지만 public에서 재현된 작은 개선 축입니다.
- `selector_adjust_rank4_conf45pull015` 결과를 보면 confidence threshold와 route 대상 샘플 정의가 multiplier 세기보다 중요합니다.
- shrink/extend 계열만 반복하면 개선폭이 작으므로, 다음은 route confidence feature, boundary sample 정의, selector label weighting을 바꾸는 쪽이 더 좋습니다.
- velocity smoothing은 OOF와 public 제출 전 판단 모두 좋지 않아 당분간 폐기합니다.
- 현재 전략은 `direct-step best anchor 유지 + selector routing 얇은 보정 + 새 축 탐색 병행`이 가장 합리적입니다.

## 생성한 주요 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_direct_multiplier_selector_20260510.py` | direct-step multiplier 후보를 샘플별로 선택하는 selector/routing 실험 |
| `reports/latest_direct_multiplier_selector_20260510.md` | selector CV와 후보별 OOF label 분포 기록 |
| `scripts/make_selector_adjustment_candidates_20260510.py` | public에서 오른 selector 방향 주변 조정 후보 생성 |
| `reports/latest_selector_adjustments_20260510.md` | selector 조정 후보와 best 대비 이동량 기록 |
| `scripts/make_direct_velocity_smoothing_probe_20260510.py` | velocity smoothing/local frame denoising probe |
| `reports/latest_direct_velocity_smoothing_probe_20260510.md` | smoothing OOF proxy와 후보 이동량 기록 |
| `experiments/public_scores.csv` | public score 누적 기록 |

## 다음 방향

1. `0.68360`을 새 anchor로 두고 selector route threshold/feature를 더 정교하게 봅니다.
2. 단순 vector extension보다 `route할 샘플을 어떻게 고를지`를 바꾸는 실험을 우선합니다.
3. 제출 5개 중 2개는 selector route refine, 3개는 per-sample coefficient 또는 boundary-specialized direct-step 같은 새 축에 씁니다.
