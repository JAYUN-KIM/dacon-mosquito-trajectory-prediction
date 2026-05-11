# 2026-05-11 selector soft routing 실험 정리

## 요약

- 전날 최고 Public LB는 `selector_adjust_rank4_conf45pull015.csv = 0.68360`이었습니다.
- 오전 연구 방향은 `conf0.45 pull`을 더 세밀하게 조정하고, 새 축으로 continuous multiplier regression을 확인하는 것이었습니다.
- `route_refine_rank1/2/3`은 모두 `0.68360`으로 기존 best와 동률이어서, threshold pull 미세 조정은 포화된 것으로 판단했습니다.
- 남은 제출권 2개는 성격을 바꿔 `selector_soft`와 full `selector_conf0.45`를 제출했습니다.
- `direct_selector_rank2_selectorsoft.csv = 0.68440`으로 현재 최고점을 갱신했고, `direct_selector_rank4_selectorconf045.csv = 0.68420`도 기존 best를 넘었습니다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `route_refine_rank2_conf45pull200.csv` | 0.68360 | conf0.45 pull 0.20, 기존 best와 동률 |
| `route_refine_rank1_conf45pull100.csv` | 0.68360 | conf0.45 pull 0.10, 기존 best와 동률 |
| `route_refine_rank3_conf45pull250.csv` | 0.68360 | conf0.45 pull 0.25, 기존 best와 동률 |
| `direct_selector_rank2_selectorsoft.csv` | **0.68440** | 확률 가중 soft selector, 현재 최고점 |
| `direct_selector_rank4_selectorconf045.csv` | 0.68420 | full conf0.45 routing, 기존 best 대비 개선 |

## 실험 축

1. Conf0.45 pull grid
   - 전날 best였던 `selector_conf0.55 + 0.15 * (selector_conf0.45 - selector_conf0.55)` 주변을 `0.10`, `0.20`, `0.25`, `0.30`으로 탐색했습니다.
   - 제출한 `0.10`, `0.20`, `0.25`가 모두 `0.68360`으로 동일해, 이 축은 public 해상도 안에서 포화된 것으로 보입니다.

2. Continuous multiplier regression
   - OOF grid-oracle multiplier를 라벨로 두고 sample-wise `forward/side/up` multiplier를 연속값으로 예측했습니다.
   - CV에서 `current_mult 0.656667`보다 낮아 제출 우선순위를 낮췄습니다.
   - 연속 multiplier 자체는 흥미롭지만, 현재 label 설계는 public 상승으로 이어질 가능성이 낮아 보입니다.

3. Selector soft routing
   - hard route 또는 threshold route 대신, selector probability로 여러 multiplier 후보를 부드럽게 평균했습니다.
   - 이 방식이 `0.68440`으로 가장 강했습니다.
   - threshold 하나로 route 여부를 자르는 것보다, 후보 확률 분포를 그대로 쓰는 쪽이 public hidden split에서 더 안정적으로 보입니다.

## 인사이트

- 오늘의 핵심 전환은 `threshold routing`에서 `probability-weighted soft routing`으로 넘어간 것입니다.
- `conf0.45` full route도 `0.68420`이라 route 범위를 넓히는 방향은 맞지만, hard decision보다 soft averaging이 더 좋았습니다.
- continuous multiplier regression은 CV가 약해 당장은 보류합니다. 다만 soft selector가 좋았기 때문에, 다음에는 continuous regression 단독보다 `selector probability calibration`이나 `candidate probability temperature`를 먼저 보는 편이 낫습니다.
- 다음 연구는 `selector_soft temperature`, `top-k probability truncation`, `boundary-only soft routing`, `soft selector와 0.6836 anchor의 미세 blend`가 우선입니다.

## 생성한 주요 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_selector_route_refine_20260511.py` | conf0.45 pull grid와 continuous multiplier regression 후보 생성 |
| `reports/latest_selector_route_refine_20260511.md` | route pull grid, continuous multiplier CV, 후보 이동량 기록 |
| `experiments/public_scores.csv` | public score 누적 기록 |

## 다음 방향

1. `direct_selector_rank2_selectorsoft.csv = 0.68440`을 새 anchor로 둡니다.
2. 다음 제출은 selector probability의 temperature를 바꿔 soft/hard 중간 영역을 찾습니다.
3. 이미 threshold pull grid가 포화됐으므로, 단순 pull 계수 미세 조정보다는 probability calibration을 우선합니다.
