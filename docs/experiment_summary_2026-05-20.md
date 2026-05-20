# 2026-05-20 새 축 재탐색과 gate 재현성 정리

## 요약

- 5/19 최고점은 `curvgate_refine_rank2_gatet52a105.csv = 0.6912`였다.
- 오늘은 curvature gate 미세조정보다 완전히 다른 새 축을 우선 시도했다.
- mirror-symmetry temporal TTA, multi-curvature action router, MLP sequence pseudo-supervision 세 축을 만들었지만 첫 public probe가 모두 `0.6902`로 하락했다.
- 이후 새 축 제출을 멈추고 검증된 curvature gate 주변으로 돌아갔다.
- `curvgate_rank4_gatet54a105.csv = 0.6912`로 기존 최고점 동률을 재현했고, `curvgate_refine_rank8_gatet52a110.csv = 0.6908`로 alpha 0.110은 과보정임을 확인했다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `mirror_tta_rank1_mirrortta_temporal_w55_gate_t52_a105_bestblend35.csv` | 0.6902 | 좌우 반사 symmetry TTA는 public에서 미재현 |
| `multicurv_action_rank2_currentblend25actiontop3p2.csv` | 0.6902 | 여러 curvature action hit-router도 current best를 흐트러뜨림 |
| `mlpseq_rank2_blend08base.csv` | 0.6902 | MLP sequence pseudo-supervision bias도 보완 효과 없음 |
| `curvgate_rank4_gatet54a105.csv` | **0.6912** | threshold 0.54, alpha 0.105가 기존 최고점 동률 재현 |
| `curvgate_refine_rank8_gatet52a110.csv` | 0.6908 | alpha 0.110은 과보정 |

## 실험 축

1. Mirror-symmetry temporal TTA
   - train과 test를 `y -> -y`로 반사해 좌우 대칭성을 이용하려 했다.
   - mirror-augmented temporal-backcast 모델을 만들고, 원본/반사 test 예측을 평균했다.
   - current best와 35% blend했지만 public은 `0.6902`로 하락했다.

2. Multi-curvature action router
   - 기존 binary curvature gate를 확장해 여러 turn config와 alpha action 중 hit 가능성이 높은 action을 고르게 했다.
   - OOF에서는 action 후보별 hit rate 차이가 있었지만 public에서는 `0.6902`로 하락했다.
   - action router의 OOF 신호는 public에 바로 믿기 어렵다.

3. MLP sequence pseudo-supervision
   - tree/physics 계열에서 벗어나 normalized sequence feature로 `MLPRegressor`를 학습했다.
   - temporal pseudo rows를 섞고 current best에 8%만 blend했지만 public은 `0.6902`로 하락했다.
   - 새 모델군 bias가 현재 champion과 보완적으로 작동하지 않았다.

4. Curvature gate 재현성 확인
   - 새 축들이 실패한 뒤 검증된 gate 주변 후보만 추가 제출했다.
   - `t54_a105`는 `0.6912`로 `t52_a105`와 동률을 만들었다.
   - `t52_a110`은 `0.6908`로 하락해, alpha는 `0.105` 근처가 안정적이다.

## 핵심 인사이트

- 새 축 3개가 서로 다른 방식인데도 모두 `0.6902`로 수렴해, current best 주변을 조금만 흐트러뜨려도 hit가 깨지는 구간으로 보인다.
- 지금은 OOF에서 보이는 미세 개선을 public에 곧장 믿으면 안 된다.
- `curvature gate + alpha 0.105` 축은 최소 두 threshold에서 `0.6912`를 재현했으므로 우연 한 방은 아니다.
- champion은 `curvgate_refine_rank2_gatet52a105.csv`, backup champion은 `curvgate_rank4_gatet54a105.csv`로 둔다.
- 다음 실험은 제출 파일 생성보다 검증 실패 원인 분석을 먼저 해야 한다.

## 생성 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_mirror_tta_temporal_gate_20260520.py` | mirror-symmetry temporal TTA와 curvature gate blend 후보 생성 |
| `scripts/run_multi_curvature_action_router_20260520.py` | multi-curvature action hit-probability router 후보 생성 |
| `scripts/run_mlp_sequence_pseudo_blend_20260520.py` | MLP sequence pseudo-supervision blend 후보 생성 |
| `reports/latest_mirror_tta_temporal_gate_20260520.md` | mirror-TTA 실험 리포트 |
| `reports/latest_multi_curvature_action_router_20260520.md` | multi-curvature action router 리포트 |
| `reports/latest_mlp_sequence_pseudo_blend_20260520.md` | MLP sequence blend 리포트 |

## 다음 방향

1. 새 후보를 더 만들기 전에 public에서 성공한 후보와 실패한 후보의 이동 벡터 분포를 비교한다.
2. current best 대비 평균 이동량, p95 이동량, gate route fraction, action별 이동 방향을 한 표로 묶어 실패 패턴을 자동 필터링한다.
3. 제출은 당분간 `t52~t54`, `alpha 0.102~0.106` 근처만 매우 조심스럽게 확인한다.
4. 새로운 축은 바로 제출하지 말고, 먼저 champion 대비 이동량이 어떤 샘플군에 집중되는지 진단한 뒤 진행한다.
