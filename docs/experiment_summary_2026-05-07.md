# 2026-05-07 실험 정리

## 오늘의 결론

오늘의 핵심 성과는 `hit-boundary weighted local-frame` 축에서 Public LB **0.67220**까지 도달한 것입니다.

어제까지는 local-frame residual과 축별 shrink가 가장 강한 방향이었고, 오늘 초반에는 retrieval route blend로 `0.66040`까지 확인했습니다. 이후 미세 조정이 정체되자, 평균 거리 대신 `1cm hit 경계` 근처 샘플을 직접 강화하는 방식으로 전환했고 이 선택이 가장 큰 돌파를 만들었습니다.

## Public Score 흐름

| 제출 파일 | Public score | 해석 |
|---|---:|---|
| `local_axis_fine_rank2_f0.46_s0.55_u0.66.csv` | 0.65880 | local-axis 미세 조정은 기존 best를 넘지 못함 |
| `retr_blend_rank1_confidentrouteblend_axisfine048052070_retrlocalmotionlocalresk50softmax075_w0.15_r0.20.csv` | 0.66040 | retrieval을 일부 샘플에만 섞는 방식이 개선 신호를 만듦 |
| `retr_route_refine_rank1_axisfine048052070_retrk32softmax075_combomeanagreement_w0.36_r0.25.csv` | 0.66000 | aggressive route refine은 정체 |
| `hit_weighted_rank1_l2_base_boundary_f0.46_s0.58_u0.70.csv` | 0.67100 | hit-boundary weighted local-frame breakthrough |
| `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv` | 0.67220 | 5-seed breakthrough refine으로 최신 최고점 |

## 오늘 만든 주요 축

1. Fine axis search
   - 어제 best 주변 forward/side/up shrink를 더 촘촘하게 탐색했습니다.
   - CV상으로는 비슷했지만 public에서는 `0.65880`으로 정체되어 미세 조정 한계를 확인했습니다.

2. Trajectory retrieval
   - train에서 유사한 과거 11프레임 궤적을 찾아 미래 residual을 가져오는 kNN 계열을 실험했습니다.
   - retrieval 단독은 약했지만, confidence가 높은 샘플에만 일부 섞는 route blend가 `0.66040`까지 개선했습니다.

3. Hit-boundary weighted local-frame
   - local-frame residual 모델에 normalized trajectory geometry feature를 추가했습니다.
   - base physics 예측 오차가 1cm 근처인 샘플에 sample weight를 크게 줬습니다.
   - 이 방식이 기존 0.660대 정체를 뚫고 `0.67100`을 만들었습니다.

4. Breakthrough refine
   - 0.671을 만든 구조를 유지한 채 base-boundary weight 강도/폭과 shrink를 다시 탐색했습니다.
   - 5-seed full ensemble 후보 `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv`가 `0.67220`으로 최신 최고점을 기록했습니다.

## 핵심 인사이트

- 평균 거리 기준으로 좋아 보이는 후보가 반드시 `R-Hit@1cm`에 강하지는 않았습니다.
- public에서 가장 크게 오른 축은 `1cm 경계 근처 샘플을 더 세게 학습`하는 metric-aware weighting이었습니다.
- seed를 늘리는 것이 항상 좋아지는 것은 아니지만, 이번 5-seed refine은 public에서 추가 개선을 만들었습니다.
- retrieval은 메인 모델은 아니지만, 향후 selective routing이나 feature로 재활용할 여지가 있습니다.
- 앞으로는 local-frame 구조를 유지하되, weight 함수와 regime별 모델 분리를 새 연구 축으로 삼는 것이 좋습니다.

## 생성한 주요 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/run_local_frame_fine_axis_search.py` | local-frame axis shrink 정밀 탐색 |
| `scripts/run_trajectory_retrieval.py` | trajectory retrieval/kNN 후보 생성 |
| `scripts/run_retrieval_blend_router.py` | retrieval을 일부 샘플에만 섞는 route blend |
| `scripts/run_retrieval_route_refine.py` | route blend 세부 조정 |
| `scripts/run_hit_weighted_local_frame.py` | hit-boundary weighted local-frame breakthrough |
| `scripts/run_hit_weighted_breakthrough_refine.py` | 0.671 breakthrough 5-seed 안정화 |

## 다음 실험 방향

1. Regime별 hit-weighted 모델
   - 속도, 회전량, 상하 이동 비율에 따라 궤적을 나누고 weight/shrink를 다르게 적용합니다.

2. Weight 함수 확장
   - base physics distance뿐 아니라 candidate oracle distance, retrieval confidence, local-frame residual 크기를 함께 사용합니다.

3. Hit-aware routing
   - 전체 test에 같은 모델을 쓰지 않고, hit 경계 근처로 추정되는 샘플에만 aggressive correction을 적용합니다.

4. Sequence model 실험
   - public probe용으로 작은 TCN/GRU residual 모델을 만들어, LightGBM과 다른 오류 패턴을 확보합니다.

