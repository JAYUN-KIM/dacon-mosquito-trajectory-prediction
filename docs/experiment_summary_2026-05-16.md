# 2026-05-16 temporal-backcast breakthrough 정리

## 요약

- 기존 selector, route gain, analog residual 계열은 `0.684` 근처에서 막혔다.
- 오늘은 기존 후보 routing을 버리고, 궤적 내부 시간 구조를 활용하는 새 축을 실험했다.
- 핵심 아이디어는 train의 과거 11개 좌표 안에서 pseudo `+80ms` 문제를 만들어 학습 데이터를 늘리는 `temporal backcast augmentation`이다.
- 첫 public probe에서 기존 최고 `0.68440`을 넘어 `0.68780`까지 상승했다.
- 따라서 2026-05-16 기준 새 주력축은 `public best anchor + temporal-backcast direct model blend`다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `hitprob_rank1_anchorblendtop3p4w015.csv` | 0.68420 | 후보별 hit 확률 라우터는 best보다 약함 |
| `hitprob_extra_top3blend030.csv` | 0.68420 | hit-prob 방향을 더 키워도 개선 없음 |
| `temporalbc_rank1_anchorblend35_tbc678w020_f1.02_s1.00_u1.00.csv` | 0.68620 | temporal-backcast 축 유효성 확인 |
| `temporalbc_rank1_anchorblend50_tbc678w020_f1.02_s1.00_u1.00.csv` | **0.68780** | 5/16 기준 새 최고점 |
| `temporalbc_rank1_tbc678w020_f1.02_s1.00_u1.00.csv` | 0.68640 | temporal-backcast 단독은 강하지만 anchor blend보다 약함 |

## 실험 축

1. Hit-probability router
   - 기존 selector는 “가장 가까운 후보”를 고르는 방식이었다.
   - 이를 평가식에 더 맞춰, 후보별 `distance <= 0.01m` hit 여부를 직접 예측하는 binary 문제로 바꿨다.
   - CV에서는 약한 개선 신호가 있었지만 public은 `0.68420`으로 기존 best `0.68440`을 넘지 못했다.
   - 결론: 후보 라우팅 계열은 더 이상 주력으로 보기 어렵다.

2. Temporal backcast augmentation
   - 각 train 궤적의 내부 시점 `c=6,7,8`을 현재로 보고, `c+2` 시점 좌표를 pseudo target으로 삼았다.
   - 입력 길이를 항상 11점으로 맞추기 위해 부족한 앞쪽 history는 초기 속도 기반으로 backcast했다.
   - pseudo row는 실제 +80ms label보다 신뢰도가 낮으므로 낮은 weight로 섞었다.
   - CV에서 `tb_c678_w020` 계열이 가장 강했고, public에서도 명확한 개선을 만들었다.

3. Public blend strength probe
   - temporal-backcast 단독을 바로 쓰기보다 기존 public best anchor와 blend했다.
   - `35% = 0.68620`, `50% = 0.68780`, `100% = 0.68640`이었다.
   - 최적 강도는 완전 교체가 아니라 `50%` 전후로 보인다.

## 핵심 인사이트

- 이 대회에서는 train label 1만 개만 쓰는 것보다, 궤적 내부 시간 구조를 pseudo-supervision으로 활용하는 것이 강한 새 신호를 만든다.
- 다만 pseudo target 분포가 실제 `0ms -> +80ms` target과 완전히 같지는 않아, anchor와 섞는 것이 더 안정적이다.
- 기존 selector-soft anchor는 여전히 좋은 중심점이며, temporal-backcast 모델은 이 중심점에서 이동할 방향을 제공한다.
- public 결과상 50% blend가 가장 좋아, 다음 제출은 `w52`, `w55`, 그리고 주변 multiplier/ensemble 변형을 우선해야 한다.

## 내일 제출 후보

| 우선순위 | 파일 | 의도 |
|---:|---|---|
| 1 | `temporalbc_refine_r1f102s100u100_w52.csv` | 50% 승리 지점 바로 주변 52% |
| 2 | `temporalbc_refine_r1f102s100u100_w55.csv` | 50%보다 조금 더 강한 55% |
| 3 | `temporalbc_refine_avgr1r2_w52.csv` | rank1/rank2 temporal 방향 평균 ensemble |
| 4 | `temporalbc_refine_r2f102s104u096_w52.csv` | side/up multiplier 변형 |
| 5 | `temporalbc_refine_avgr1r2r3_w52.csv` | rank1~3 평균 ensemble |

## 생성 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_hit_probability_router_20260516.py` | 후보별 1cm hit probability 라우터 실험 |
| `reports/latest_hit_probability_router_20260516.md` | hit-prob router CV와 제출 후보 기록 |
| `scripts/run_temporal_backcast_augmentation_20260516.py` | temporal backcast pseudo-supervision 모델 생성 |
| `reports/latest_temporal_backcast_augmentation_20260516.md` | temporal-backcast CV, public probe, 후보 기록 |
| `scripts/make_temporal_backcast_refine_candidates_20260516.py` | 50% 주변 blend strength와 temporal ensemble 후보 생성 |
| `reports/latest_temporal_backcast_refine_20260516.md` | 내일 제출 후보와 refine 후보 전체 기록 |

## 다음 방향

1. 내일은 우선 `w52`, `w55`, `avg_r1r2_w52`를 제출해 50% 주변 최적점을 확인한다.
2. 만약 `w52/w55`가 `0.68780`보다 오르면, blend strength grid를 `50~60%` 사이에서 더 촘촘하게 판다.
3. ensemble 변형이 오르면 temporal direction을 여러 개 평균하는 쪽으로 확장한다.
4. 이후에는 pseudo cutoff/weight curriculum, regime별 pseudo weight, pseudo sample denoising을 실험한다.
