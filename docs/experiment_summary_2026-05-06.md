# 2026-05-06 실험 정리

## 오늘의 결론

오늘 가장 중요한 발견은 `local-frame residual` 방향이 확실히 유효하다는 점이다.

기존 `x, y, z` 좌표계에서 residual을 직접 예측하는 방식보다, 마지막 속도 방향을 기준으로 로컬 좌표계를 만들고 residual을 `진행방향 / 횡방향 / 상하방향`으로 분해해서 예측하는 방식이 public score를 크게 끌어올렸다.

## Public Score 흐름

| 제출 파일 | Public score | 해석 |
| --- | ---: | --- |
| `physics_param_search_best.csv` | 0.61539 | 물리 기반 + threshold-aware 보정 첫 기준점 |
| `aggressive_lgbm_residual.csv` | 0.63420 | LGBM residual이 public에서 강하게 확인됨 |
| `candidate_binary_hit_selector.csv` | 0.62600 | 후보 선택기는 oracle 여지는 있었지만 현재 feature로는 부족 |
| `residual_zoo_rank1_lgbm_wide_a0275_s0.25.csv` | 0.63480 | LGBM wide residual이 소폭 개선 |
| `public_push_lgbm_wide_a0275_s0.40_5seed.csv` | 0.64120 | 5-seed residual ensemble + 강한 shrink가 유효 |
| `local_frame_lgbm_a0275_s0.55_5seed.csv` | 0.65900 | local-frame residual target이 큰 점프를 만듦 |
| `local_axis_rank1_f0.48_s0.55_u0.62.csv` | 0.65940 | 축별 shrink로 추가 소폭 개선 |

## 현재 베스트 방향

메인 방향은 다음과 같이 잡는다.

1. 물리 baseline은 유지한다.
2. 마지막 속도 방향 기준 local frame을 만든다.
3. residual을 local frame에서 예측한다.
4. 진행방향 residual은 덜 믿고, 횡방향/상하방향 residual은 더 살린다.

축별 shrink 실험에서 좋은 조합은 다음 근방이었다.

```text
forward_shrink = 0.48
side_shrink    = 0.55
up_shrink      = 0.62 ~ 0.80
```

## 해석

모기 궤적은 짧은 시간의 미래 예측이므로 마지막 속도 방향이 강한 기준축이 된다. 글로벌 `x, y, z` 좌표계에서 오차를 직접 예측하면 같은 물리적 흔들림이 샘플마다 다른 방향으로 보이지만, local frame으로 바꾸면 오차 구조가 더 일관되게 정렬된다.

특히 진행방향은 물리 baseline이 이미 잘 잡고 있고, 모델 보정은 과하게 넣으면 오히려 흔들릴 수 있다. 반대로 횡방향/상하방향은 미세한 곡률과 흔들림을 residual 모델이 보정할 여지가 더 큰 것으로 보인다.

## 생성한 주요 스크립트

- `scripts/run_residual_model_zoo.py`: LGBM/CatBoost residual 모델 비교
- `scripts/run_public_push_residual.py`: public에서 좋았던 LGBM wide residual family를 5-seed로 확장
- `scripts/run_feature_rich_residual.py`: 다항식/물리 후보 기반 feature-rich residual 실험
- `scripts/run_local_frame_residual.py`: local-frame residual target 실험
- `scripts/run_local_frame_axis_shrink.py`: local-frame residual의 축별 shrink 탐색
- `scripts/blend_submissions.py`: 제출 파일 간 좌표 블렌딩

## 다음 실험 제안

다음 라운드는 local-frame을 더 세게 민다.

1. forward/side/up 축별 모델 용량을 다르게 주기
2. up 축 shrink를 `0.62~0.95` 근방에서 더 촘촘하게 탐색
3. local-frame residual을 거리 loss가 아니라 hit@1cm 근방에 맞춘 sample weighting으로 학습
4. public/private 괴리를 줄이기 위해 seed별 CV 분산이 낮은 후보와 public 점수 높은 후보를 함께 관리

지금은 딥러닝으로 전환하기보다, local-frame residual + 피처/타깃/축별 calibration을 더 파는 것이 효율적이다.
