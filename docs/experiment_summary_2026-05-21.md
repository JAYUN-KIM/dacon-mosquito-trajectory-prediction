# 2026-05-21 post-process 새 축 손절과 co-champion 안정성 확인

## 요약

- 5/20 최고점은 `curvgate_refine_rank2_gatet52a105.csv = 0.6912`와 `curvgate_rank4_gatet54a105.csv = 0.6912`였다.
- 오늘은 current best 주변 미세조정보다 새 축을 우선했다.
- local target manifold projection은 내부 OOF에서 좋아 보였지만 public은 `0.6898`로 크게 하락했다.
- hit-rescue specialist는 champion을 대부분 유지하고 일부 샘플만 temporal55로 되돌렸지만 public은 `0.6906`으로 하락했다.
- co-champion blend는 `w50`, `w65`, `w35`가 모두 `0.6912` 동률을 기록해, `t52~t54`, `alpha=0.105` 주변이 안정권임을 재확인했다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `manifoldproj_rank2_k256_b060_cap0003.csv` | 0.6898 | train target-local manifold로 약하게 투영했지만 public에서 크게 하락 |
| `hitrescue_rank1_temporal55_top075.csv` | 0.6906 | 일부 샘플만 temporal55로 hard swap했지만 champion 대비 손실 |
| `cochamp_blend_t52_t54_w50.csv` | **0.6912** | t52/t54 동률 champion 50:50 blend도 동률 유지 |
| `cochamp_blend_t52_t54_w65.csv` | **0.6912** | t52 쪽 가중 blend도 동률 유지 |
| `cochamp_blend_t52_t54_w35.csv` | **0.6912** | t54 쪽 가중 blend도 동률 유지 |

## 실험 축

1. Local target manifold projection
   - champion 예측의 local-frame displacement를 train 정답 local manifold 쪽으로 약하게 투영했다.
   - OOF proxy에서는 `+0.0016~+0.0017` hit 개선처럼 보였지만 public은 `0.6898`로 실패했다.
   - train 정답 분포를 직접 끌어오는 방식은 public split에서 hit 경계를 깨는 과적합 신호로 판단한다.

2. Hit-rescue specialist
   - champion이 놓치고 특정 후보가 맞출 가능성이 있는 샘플만 분류해 hard swap하는 구조를 만들었다.
   - 가장 안전한 후보인 `temporal55 top 7.5%`만 제출했지만 `0.6906`으로 하락했다.
   - 현재 champion miss 샘플은 OOF feature만으로 정밀하게 분리하기 어렵다.

3. Co-champion blend
   - `curvgate_refine_rank2_gatet52a105.csv`와 `curvgate_rank4_gatet54a105.csv`는 모두 `0.6912`였다.
   - 두 후보 사이의 평균 이동량은 약 `0.000032m`로 매우 작다.
   - `w50`, `w65`, `w35`가 모두 `0.6912`라, 이 구간은 안정적이지만 돌파력은 제한적이다.

## 핵심 인사이트

- 오늘 실패한 두 새 축은 모두 post-process 성격이다.
- current best 주변 좌표를 직접 끌거나 일부 샘플을 교체하는 방식은 public에서 바로 손실이 발생했다.
- 지금 필요한 것은 champion 위에 얹는 작은 보정이 아니라, temporal-backcast급으로 학습 데이터/타깃 정의를 바꾸는 새 supervision 축이다.
- `0.6912` 안정권은 확보했지만, 0.7 돌파를 위해서는 기존 curvature gate를 버릴 각오로 큰 후보를 만들어야 한다.
- 내일은 제출 후보를 만들기 전에 train 내부 oracle 분석으로 “어떤 새 타깃/후보군이 0.7 hit에 필요한 추가 hit를 만들 수 있는지”를 먼저 확인한다.

## 생성 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_local_target_manifold_projection_20260521.py` | champion local displacement를 train target-local manifold로 약하게 투영 |
| `scripts/run_hit_rescue_specialist_20260521.py` | champion miss 가능 샘플만 hard swap하는 rescue specialist |
| `reports/latest_local_target_manifold_projection_20260521.md` | manifold projection 실험 리포트 |
| `reports/latest_hit_rescue_specialist_20260521.md` | hit-rescue specialist 실험 리포트 |

## 다음 방향

1. post-process류는 우선순위를 낮춘다.
2. 0.7 돌파용으로 새 pseudo-label curriculum을 만든다.
3. train 내부 시점 전체를 활용해 `+80ms`뿐 아니라 다양한 cutoff/속도 regime별 pseudo target을 재설계한다.
4. 후보 생성 단계에서 OOF oracle hit potential을 먼저 보고, champion이 못 맞추는 샘플을 실제로 살릴 수 있는 후보군만 public probe한다.
5. 내일 목표는 `0.6912` 미세 개선이 아니라, 실패 가능성을 감수하고 `0.7` 근처를 노릴 수 있는 큰 새 축 발견이다.
