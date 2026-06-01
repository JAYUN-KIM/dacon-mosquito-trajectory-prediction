# 2026-06-01 최종 정리

## 최종 성과

- 최종 Public LB: **0.69200**
- 최종 선택 제출: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv`
- 핵심 축: recursive one-step dynamics + top 9% gain-gated routing + 45% move
- 최종 확인일: 2026-06-01

## 최종 선택 이유

`0.69200` 후보는 단순한 전체 평균 보정이 아니라, `+40ms` one-step dynamics를 두 번 재귀 적용한 후보를 만들고, 그 후보가 이득일 가능성이 높은 소수 샘플에만 제한적으로 섞은 방식입니다.

이후 강도 `47.5%`, winner residual calibrator, self-consistency router 등 여러 안전 보정이 모두 `0.69200` 동률에 머물렀고, 더 공격적인 새 축들은 대부분 하락했습니다. 따라서 최종 제출은 public에서 가장 반복적으로 안정성이 확인된 recursive gate 계열로 확정합니다.

## 마지막 clean-room 실험 결론

5/31에는 기존 submission, champion anchor, cache를 모두 배제하고 raw trajectory/label만 사용하는 clean-room physics action selector를 새로 만들었습니다.

- best single raw action OOF hit: `0.59930`
- raw action oracle OOF hit: `0.77670`
- soft selector OOF hit: `0.62490`
- public probe: `cleanroom31_rank1_soft_p3_blend90.csv = 0.65140`

OOF에서는 action pool과 selector 가능성이 있어 보였지만 public에서 크게 하락했습니다. 이 결과는 raw-only selector의 OOF-public transfer가 불안정하며, 기존 `0.69200` 계열의 누적 temporal/gate calibration이 실제 public hit를 지키는 데 중요했다는 신호로 해석합니다.

## 최종 인사이트

- R-Hit@1cm에서는 평균 거리보다 1cm 경계 샘플의 생존 여부가 훨씬 중요했습니다.
- global residual보다 마지막 속도 방향 기준 local-frame target이 안정적이었습니다.
- direct-step local target과 temporal-backcast pseudo-supervision이 가장 큰 중간 돌파를 만들었습니다.
- curvature correction은 전체 적용보다 gate가 고른 샘플에만 약하게 적용할 때 효과가 있었습니다.
- recursive one-step dynamics도 전체 적용은 약했지만, gain gate로 top fraction을 제한하면 최고점을 만들었습니다.
- OOF에서 좋아 보이는 새 축이 public에서 반복적으로 무너졌으므로, 이 대회에서는 public-stable low-risk routing이 강한 전략이었습니다.

## 점수 흐름 요약

| 단계 | Public LB | 핵심 변화 |
|---|---:|---|
| Physics baseline | 0.61539 | constant velocity/acceleration 계열 |
| LGBM residual | 0.63420 | 물리 anchor residual 보정 |
| Local-frame residual | 0.65900 | 마지막 속도 방향 local 좌표계 |
| Hit-boundary weighted | 0.67100 | 1cm 경계 샘플 직접 가중 |
| Direct-step local target | 0.67800 | residual 대신 +80ms displacement 직접 예측 |
| Selector soft routing | 0.68440 | 후보 probability-weighted averaging |
| Temporal-backcast | 0.68780 | 궤적 내부 pseudo-supervision |
| Curvature correction | 0.69000 | constant-turn correction |
| Curvature gate | 0.69120 | correction 적용 샘플 gate |
| Recursive one-step gate | 0.69200 | 최종 최고점 |
| Clean-room raw selector | 0.65140 | OOF-public transfer 실패 |

## 최종 파일

- 최종 선택 후보 생성 축: `scripts/run_recursive_onestep_dynamics_20260526.py`
- 최고점 주변 리파인: `scripts/make_recursive_onestep_gate_jitter_20260527.py`
- 최고점 peak 확인: `scripts/make_recursive_onestep_gate_peak_20260527.py`
- 최종 실패 축 진단: `scripts/run_cleanroom_action_selector_20260531.py`
- 전체 점수 로그: `experiments/public_scores.csv`

## 비고

원본 데이터와 제출 CSV는 대회 규정 및 용량 관리를 위해 GitHub에 포함하지 않습니다. 저장소에는 실험 코드, 결과 리포트, 점수 기록, 재현 가능한 연구 메모만 보관합니다.
