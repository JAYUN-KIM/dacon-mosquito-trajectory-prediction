# 2026-05-24 champion alpha calibration 정리

## 요약

- 오늘 최고 Public LB는 `0.69140`이다.
- 기존 champion `curvgate_refine_rank2_gatet52a105.csv = 0.69120`에서 `t52` gate는 유지하고 alpha만 낮추는 방향이 유효했다.
- 새 축으로 시도한 regime miss policy와 analog KNN residual은 각각 `0.69060`, `0.68860`으로 하락해 주력에서 내렸다.
- alpha를 올린 `t52_a1075`는 `0.69100`으로 떨어졌고, alpha를 낮춘 `t52_a1025`는 `0.69140`으로 올랐다.
- 후속 alpha band probe도 `0.69140` 동률을 기록해, 단기적으로는 `threshold=0.52` 고정 + alpha 세부 보정이 가장 믿을 만하다.

## Public 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `regimemiss_rank1_c64_min55_net0003_p165.csv` | 0.69060 | champion miss regime selector는 OOF 신호 대비 public 하락 |
| `analogknn_rank1_k64_p10_s012_cap00025_r100.csv` | 0.68860 | 유사 궤적 KNN residual 전이는 현재 champion을 크게 흔듦 |
| `champmicro_rank1_gatet520a1075.csv` | 0.69100 | `alpha=0.1075`는 과보정 |
| `champmicro_rank3_gatet520a1025.csv` | **0.69140** | `alpha=0.1025`가 새 최고점 |
| `champalpha_rank1_t52a1015.csv` | **0.69140** | alpha-down 주변 후속 probe도 최고점 동률 |

## 오늘 만든 실험 축

1. Regime miss policy
   - 궤적 regime을 클러스터링하고, cluster별로 champion 대신 temporal/fixed-alpha 후보를 쓰는 selector를 만들었다.
   - CV상 약한 개선 신호가 있었지만 public은 `0.69060`으로 하락했다.
   - 결론: 현재 feature/cluster 기준으로는 champion miss를 안정적으로 분리하지 못한다.

2. Consensus hit mode
   - 여러 후보 좌표를 작은 coordinate cloud로 보고 weighted local consensus mode를 선택했다.
   - train OOF 개선폭이 `+0.0003` 수준에 그쳐 제출 우선순위에서 밀렸다.
   - 결론: 후보끼리 가까운 위치가 꼭 1cm hit 가능성이 높은 위치는 아니다.

3. Analog KNN residual
   - 유사한 train 궤적의 champion residual을 test에 전이했다.
   - public `0.68860`으로 크게 하락했다.
   - 결론: champion residual은 최근 궤적 모양만으로 국소 전이하기 어렵다.

4. Champion micro tuning
   - `t52/t54` co-champion 주변에서 threshold와 alpha만 얇게 바꿨다.
   - `alpha=0.1075`는 하락, `alpha=0.1025`는 상승했다.
   - 결론: threshold보다 alpha 방향이 더 강한 public 신호를 냈다.

5. Champion alpha refine
   - `0.1025` 주변 alpha band 후보를 추가 생성했다.
   - 후속 probe가 `0.69140` 동률을 만들었다.
   - 결론: 당장 새 축보다 alpha calibration을 조금 더 촘촘히 보는 편이 낫다.

## 핵심 인사이트

- 새 모델이나 새 좌표 후보를 섞으면 현재 champion의 기존 hit를 쉽게 깨뜨린다.
- `t52` gate의 sample 선택은 아직 유효하고, 조정해야 할 것은 route 여부보다 correction 강도다.
- alpha-up은 과보정으로 보이고, alpha-down은 public에서 실제 개선을 만들었다.
- `0.6914` 이상을 보려면 `threshold=0.52`를 고정한 상태에서 alpha `0.100~0.103` 구간을 더 촘촘히 확인하는 것이 우선이다.
- alpha band가 막히면 그때 다시 큰 축으로 돌아가야 한다.

## 다음 액션

- `t52_a1005`, `t52_a1010`, `t52_a1022`, `t52_a1028`처럼 `0.1025` 주변을 더 촘촘히 확인한다.
- 이미 하락한 `alpha >= 0.1075`, KNN residual, smoothing, snap rebound, broad miss-policy hard swap은 보류한다.
- 제출 수가 부족하면 `0.69140` 동률 후보를 anchor로 두고 아주 작은 ensemble이나 rounding 안정성만 점검한다.

## 관련 리포트

- `reports/latest_regime_miss_policy_20260524.md`
- `reports/latest_consensus_hit_mode_20260524.md`
- `reports/latest_analog_knn_residual_20260524.md`
- `reports/latest_champion_micro_tuning_20260524.md`
- `reports/latest_champion_alpha_refine_20260524.md`
