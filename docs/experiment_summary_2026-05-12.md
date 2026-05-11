# 2026-05-12 selector soft 후속 연구 정리

## 요약

- 전날 최고 Public LB는 `direct_selector_rank2_selectorsoft.csv = 0.68440`이었습니다.
- 오늘은 selector soft routing을 더 개선하기 위해 probability temperature, top-k truncation, expanded candidate pool, selector seed ensemble을 순서대로 확인했습니다.
- `top3_t1.00`과 `soft_t0.75`는 CV에서는 좋아 보였지만 public은 모두 `0.68420`으로 하락했습니다.
- expanded multiplier candidate pool은 `0.68400`으로 더 약했습니다.
- selector seed ensemble blend는 `seedens_rank3_seedens3blend35.csv = 0.68440`으로 최고점과 동률을 만들었지만, 새 최고점 갱신은 없었습니다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `softtemp_rank8_top3t100.csv` | 0.68420 | top-3 truncation은 full soft보다 약함 |
| `softtemp_rank1_softt075.csv` | 0.68420 | sharper temperature도 full soft보다 약함 |
| `seedens_rank1_seedens3.csv` | 0.68420 | selector seed ensemble 단독은 하락 |
| `seedens_rank3_seedens3blend35.csv` | **0.68440** | seed ensemble 35% blend는 현재 최고점과 동률 |
| `expanded_selector_rank1_expandedsoftblend015.csv` | 0.68400 | expanded candidate pool은 public에서 약함 |

## 실험 축

1. Selector soft temperature
   - selector probability를 `T=0.75`, `0.85`, `0.95`, `1.10` 등으로 조정했습니다.
   - CV에서는 낮은 temperature와 top-3 truncation이 좋아 보였지만 public은 `0.68420`으로 하락했습니다.
   - 결론적으로 현재 public split에서는 full soft probability를 그대로 쓰는 것이 더 안정적입니다.

2. Top-k truncation
   - 확률 상위 후보만 남기는 방식으로 soft average를 더 날카롭게 만들었습니다.
   - `top3_t1.00`이 CV 1위였지만 public은 하락했습니다.
   - 후보 확률 꼬리까지 남겨두는 것이 hidden split에서 도움이 되는 것으로 보입니다.

3. Expanded candidate pool
   - 기존 12개 multiplier 후보를 32개로 늘려 selector가 더 넓게 고를 수 있게 했습니다.
   - OOF label은 넓은 후보에 많이 분산됐지만, CV와 public 모두 약했습니다.
   - 후보 pool을 무작정 넓히면 selector label noise가 커지는 것으로 보입니다.

4. Selector seed ensemble
   - selector 모델 seed 여러 개의 확률을 평균해 probability를 안정화했습니다.
   - 단독 seed ensemble은 `0.68420`으로 하락했지만, 현재 best와 35% blend한 후보는 `0.68440` 동률이었습니다.
   - 안정화 방향은 완전히 틀리지는 않았지만, 현재 best를 넘기기에는 신호가 약했습니다.

## 인사이트

- `0.68440` 이후에는 단순 probability sharpening, top-k truncation, candidate pool 확장 모두 새 돌파로 이어지지 않았습니다.
- selector soft 자체는 여전히 anchor로 유효하지만, 후속 미세 보정은 public에서 쉽게 손해가 납니다.
- 다음 연구는 selector probability 후처리보다 `selector label 설계`, `boundary-only soft routing`, `hit 개선 여부를 직접 예측하는 route gain model` 쪽으로 넘어가는 것이 좋아 보입니다.
- expanded pool 결과가 약했으므로 후보 수를 늘리기보다, 후보를 고르는 목적 함수를 `oracle distance 최소`에서 `1cm hit 전환 가능성`으로 바꾸는 편이 더 중요합니다.

## 생성한 주요 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_selector_soft_temperature_20260512.py` | selector soft probability temperature/top-k truncation 후보 생성 |
| `reports/latest_selector_soft_temperature_20260512.md` | temperature/top-k CV와 후보 이동량 기록 |
| `scripts/run_expanded_selector_pool_20260512.py` | expanded multiplier candidate pool 실험 |
| `reports/latest_expanded_selector_pool_20260512.md` | expanded pool CV와 후보 분포 기록 |
| `scripts/run_selector_seed_ensemble_20260512.py` | selector probability seed ensemble 후보 생성 |
| `reports/latest_selector_seed_ensemble_20260512.md` | seed ensemble 후보 이동량 기록 |
| `experiments/public_scores.csv` | public score 누적 기록 |

## 다음 방향

1. `direct_selector_rank2_selectorsoft.csv = 0.68440`을 계속 anchor로 둡니다.
2. 다음 연구는 `boundary-only soft routing`과 `route gain binary model`을 우선합니다.
3. temperature/top-k, expanded pool 단순 확장은 당분간 중단합니다.
4. 자동 연구는 2026-05-13 수요일 23:00부터, GitHub 정리 업로드는 23:30부터 실행되도록 조정했습니다.
