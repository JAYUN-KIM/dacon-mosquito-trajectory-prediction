# 2026-05-23 temporal curriculum과 공격적 물리 새 축 실패 정리

## 요약

- 5/21 기준 최고점은 `curvgate_refine_rank2_gatet52a105.csv = 0.6912`였다.
- 오늘은 시간이 부족해 빠르게 낼 수 있는 새 축을 우선 생성했다.
- temporal-backcast를 더 넓게 확장한 fast temporal curriculum 후보는 제출한 3개가 모두 `0.6906`으로 하락했다.
- 이후 완전히 다른 물리 가정으로 exp-weighted polynomial smoothing과 jerk/snap rebound extrapolation 후보를 제출했다.
- snap rebound 후보는 `0.6866`, smoothing 후보는 `0.6716`으로 크게 하락했다.
- 결론적으로 champion 주변 후처리, temporal curriculum 확장, smoothing/snap 물리축은 모두 현재 public split에서 돌파력이 없다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `tempcurr_rank1_tcc5678w012v6789w006champblend15f102s100u100.csv` | 0.6906 | 넓은 temporal curriculum 15% blend, champion 대비 하락 |
| `tempcurr_rank4_tcc678w022v89w004champblend15f102s100u100.csv` | 0.6906 | 기존 temporal cutoffs에 가까운 보수 후보도 하락 |
| `tempcurr_rank5_tcc5678w012v6789w006cochampblend20f102s100u100.csv` | 0.6906 | co-champion 기반 temporal curriculum blend도 하락 |
| `fastnew_rank2_snapsnapv102a035jm0p2d096blend18.csv` | 0.6866 | jerk/snap rebound 물리축은 크게 하락 |
| `fastnew_rank1_smoothewpolyw11d2r55blend18.csv` | 0.6716 | exp-weighted polynomial smoothing 축은 완전 실패 |

## 실험 축

1. Fast temporal curriculum
   - 기존 temporal-backcast를 더 넓은 cutoff와 약한 horizon1 velocity pseudo-label로 확장했다.
   - champion과 15~20%만 blend해 위험을 줄였지만 결과는 모두 `0.6906`이었다.
   - temporal-backcast 계열은 이미 `w55 + curvature gate` 위에서 포화됐다고 본다.

2. Exp-weighted polynomial smoothing
   - 최근 관측치에 더 큰 가중치를 주는 polynomial extrapolation으로 관측 노이즈 제거 효과를 기대했다.
   - champion과 18%만 섞었는데도 `0.6716`까지 급락했다.
   - 이 문제에서는 smoothing/denoising이 아니라 마지막 순간의 국소 동역학을 살리는 쪽이 훨씬 중요하다.

3. Jerk/snap rebound extrapolation
   - constant-turn과 다른 물리 가정으로, 최근 가속도 변화량을 반동처럼 사용하는 discrete Taylor 계열을 만들었다.
   - smoothing보다는 낫지만 `0.6866`으로 champion과 거리가 컸다.
   - jerk/snap은 OOF proxy가 좋아 보여도 public에 직접 섞기 어렵다.

4. Heavy action distillation
   - train OOF oracle에서 후보별 hit rescue 가능성을 학습하는 action selector도 시도했다.
   - 계산량이 커서 최종 제출 축으로는 쓰지 않았고, 리포트에 diagnostic만 남겼다.
   - oracle 상 rescue 후보는 있지만 harm rate가 너무 높아, 단순 action switch는 위험하다.

## 핵심 인사이트

- `0.6912` champion은 매우 얇은 hit boundary 위에 있어, 새로운 물리 후보를 조금만 섞어도 hit가 깨진다.
- temporal-backcast 확장은 추가 pseudo-label을 늘려도 개선이 아니라 과보정으로 이어졌다.
- smoothing 계열은 이 대회에서 위험하다. 실제 모기 궤적은 단순 노이즈 제거보다 순간 turn/acceleration 보존이 중요해 보인다.
- jerk/snap처럼 공격적인 물리축도 단독 public 신호가 약하다.
- 내일은 기존 좌표 예측을 더 흔드는 방식이 아니라, train 내부 검증 구조 자체를 다시 설계해야 한다.

## 생성 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_temporal_curriculum_fast_20260522.py` | 빠른 temporal curriculum 후보 생성 |
| `scripts/run_aggressive_new_axes_20260523.py` | smoothing/action distillation 공격 축 진단 |
| `scripts/run_fast_two_new_axes_20260523.py` | smoothing/snap 두 물리축 후보 생성 |
| `reports/latest_temporal_curriculum_fast_20260522.md` | temporal curriculum 후보 리포트 |
| `reports/latest_aggressive_new_axes_20260523.md` | heavy aggressive axes 진단 리포트 |
| `reports/latest_fast_two_new_axes_20260523.md` | fast smoothing/snap 두 후보 리포트 |

## 다음 방향

1. 후처리, smoothing, snap, temporal curriculum 확장은 모두 우선순위를 낮춘다.
2. 2026-05-24에는 제출 파일부터 만들지 말고, train OOF에서 champion이 틀린 샘플만 따로 분해한다.
3. 샘플을 속도/회전/가속도/높이 변화 regime으로 나눈 뒤, regime별로 champion miss의 공통 패턴을 찾는다.
4. 새 모델은 후보 좌표를 직접 섞는 방식보다 `hit 가능성 calibration` 또는 `uncertainty-aware abstention` 기반으로 설계한다.
5. 0.7 돌파를 위해서는 단일 새 물리식보다, miss regime을 정확히 분리하는 validation/selector 설계가 먼저다.
