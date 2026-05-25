# 2026-05-25 plateau 포화와 hit-mode retrieval 손절 정리

## 요약

- 오늘 제출한 후보들은 모두 `0.69140` 또는 그 이하로 나왔다.
- 대회 정의에서 다시 출발해 만든 local hit-mode retrieval은 `0.68980`으로 하락해 새 축 후보에서 내렸다.
- `t52` alpha-down 초미세 후보와 plateau disagreement 후보는 모두 `0.69140`에 묶였다.
- 결론적으로 `0.69140` 주변 alpha/gate/plateau 미세조정은 포화로 본다.

## Public 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `hitmode_rank1_localshape_k32_s0008_clustermean_r30_b18_c00008.csv` | 0.68980 | local target mode retrieval은 public 하락 |
| `champalpha2_rank1_t52a1010.csv` | 0.69140 | alpha-down 초미세 probe도 최고점 동률 |
| `champalpha2_rank5_t52a1005.csv` | 0.69140 | 더 낮은 alpha도 최고점 동률 |
| `plateaudis_rank2_stablemeanplateau.csv` | 0.69140 | plateau 후보 평균도 동률 |
| `plateaudis_rank4_towarda1005top15b50.csv` | 0.69140 | plateau disagreement 부분 이동도 동률 |

## 오늘 실험 축

1. Local hit-mode retrieval
   - 기존 retrieval 평균 대신 이웃 local target 중 가장 조밀한 mode/cluster mean을 찾았다.
   - CV에서는 champion proxy 대비 `+0.001` 이상 신호가 있었지만 public은 `0.68980`으로 하락했다.
   - 결론: local future mode는 train OOF에서 분리돼 보여도 public 일반화가 약하다.

2. Champion alpha ultrafine
   - `t52` threshold를 고정하고 alpha를 `0.0995~0.1028`까지 더 촘촘히 확인했다.
   - 제출된 `0.1010`, `0.1005` 모두 `0.69140`으로 동률이었다.
   - 결론: alpha-down 구간은 넓은 plateau이며, 더 촘촘히 찍어도 상한 돌파 가능성이 낮다.

3. Plateau disagreement
   - `0.69140` plateau 후보와 기존 stable champion 후보 간의 아주 작은 disagreement만 이용했다.
   - 평균 후보와 부분 이동 후보 모두 `0.69140` 동률이었다.
   - 결론: public-stable 후보들이 서로 보완한다기보다 같은 public hit set에 묶여 있을 가능성이 크다.

## 핵심 판단

- `0.69140`은 현재 curvature-gate alpha-down 계열의 public plateau다.
- alpha-up은 `0.69100`으로 하락했고, alpha-down은 `0.69140` 이상을 못 만들었다.
- hit-mode retrieval처럼 champion을 새 좌표축으로 당기는 방식은 public에서 위험하다.
- 단기 미세조정은 멈추고, 다음은 더 큰 supervision/target formulation 축으로 돌아가야 한다.

## 다음 액션

- 더 이상 `t52` alpha를 촘촘히 찍지 않는다.
- `plateaudis`, `champalpha2` 후속 후보도 우선순위에서 내린다.
- 다음 연구는 새 target formulation, pseudo-label 품질 개선, 또는 trajectory segment-level 학습처럼 기존 gate 후처리와 다른 구조를 우선한다.
- 새 축 후보를 만들더라도 `0.69140` anchor 대비 이동량과 기존 hit 파괴 위험을 먼저 기록한다.

## 관련 리포트

- `reports/latest_local_hit_mode_retrieval_20260525.md`
- `reports/latest_champion_alpha_ultrafine_20260525.md`
- `reports/latest_plateau_disagreement_20260525.md`
