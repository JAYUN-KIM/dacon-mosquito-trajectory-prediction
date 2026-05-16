# DACON 모기 비행 궤적 예측 AI 경진대회

40ms 간격으로 관측된 과거 11개 3D 좌표를 활용해 모기의 `+80ms` 미래 위치 `(x, y, z)`를 예측하는 프로젝트입니다.  
단순 평균 거리보다 대회 지표인 `R-Hit@1cm`를 직접 끌어올리는 것을 목표로, 물리 기반 baseline에서 출발해 local-frame residual, selective retrieval routing, hit-boundary weighted modeling까지 확장했습니다.

## 프로젝트 개요

- 대회: 모기 비행 궤적 예측 AI 경진대회
- 플랫폼: DACON
- 평가 지표: R-Hit@1cm
- 입력 데이터: 40ms 간격 과거 11개 3D 좌표 `-400ms ~ 0ms`
- 목표: 마지막 관측 시점 기준 `+80ms` 미래 좌표 `(x, y, z)` 예측
- hit 기준: 예측 좌표와 실제 좌표의 3D Euclidean distance가 `0.01m` 이하
- 좌표 단위: meter

## 현재 성과

<!-- AUTO:PROJECT_STATUS:START -->
- 최고 Public LB: **0.68780**
- 최신 최고점 확인일: **2026-05-16**
- 핵심 개선 축: temporal-backcast pseudo-supervision + public-best anchor 50% blend
- 최신 새 축 검토: 2026-05-16 temporal-backcast augmentation이 기존 최고 `0.68440`에서 **0.68780**으로 개선
- 상세 실험 기록은 `docs/`, `reports/`, `experiments/` 디렉토리에 분리 보관
<!-- AUTO:PROJECT_STATUS:END -->

## 예측 타겟

| 타겟 | 의미 |
|---|---|
| x | sensor-local forward 방향 미래 위치 |
| y | sensor-local left 방향 미래 위치 |
| z | sensor-local up 방향 미래 위치 |

## 핵심 접근법

1. 물리 기반 baseline
   - constant velocity, constant acceleration, polynomial extrapolation을 먼저 비교했습니다.
   - `R-Hit@1cm` 기준으로 velocity/acceleration 계수를 직접 탐색했습니다.

2. Residual ML
   - 물리 예측값을 anchor로 두고 LightGBM이 residual을 보정하도록 구성했습니다.
   - residual 예측을 그대로 더하지 않고 shrink를 적용해 과보정을 줄였습니다.

3. Local-frame residual
   - 마지막 속도 방향을 forward 축으로 하는 local coordinate frame을 만들었습니다.
   - global `x, y, z` residual 대신 `진행 방향 / 좌우 방향 / 상하 방향` residual을 예측했습니다.
   - 이 전환으로 Public LB가 `0.64120`에서 `0.65900`까지 크게 올랐습니다.

4. Selective retrieval routing
   - train에서 유사한 과거 궤적을 검색해 residual을 가져오는 kNN/retrieval 계열을 실험했습니다.
   - retrieval 단독은 약했지만, confidence가 높은 일부 샘플에만 섞는 route blend가 `0.66040`까지 개선했습니다.

5. Hit-boundary weighted local-frame
   - 평균 거리 최적화가 아니라 `1cm hit 경계` 근처 샘플을 더 중요하게 학습했습니다.
   - normalized trajectory geometry feature를 추가하고, base physics error가 1cm 근방인 샘플에 가중치를 줬습니다.
   - 이 축에서 `0.67100`, 이후 5-seed refine으로 `0.67220`까지 상승했습니다.

6. Direct-step local target
   - residual anchor 보정만으로는 정체가 보여, 마지막 관측 좌표 기준 `+80ms displacement`를 local frame에서 직접 예측했습니다.
   - sample별 motion scale로 target을 정규화한 뒤 추론 시 다시 복원하는 방식을 적용했습니다.
   - 순수 direct-step LGBM boundary 후보가 Public LB `0.67800`으로 새 최고점을 만들었습니다.

7. Selector confidence routing
   - direct-step 후보의 forward/side/up multiplier를 샘플별로 고르는 selector를 구성했습니다.
   - hard routing은 과적합 위험이 있어 confidence가 높은 샘플만 선택적으로 route했습니다.
   - `selector_conf0.55`가 `0.68340`, 이후 conf0.45 방향을 약하게 섞은 후보가 `0.68360`까지 개선했습니다.

8. Probability-weighted selector soft routing
   - threshold로 후보를 하나 고르는 대신, selector probability로 여러 multiplier 후보를 부드럽게 평균했습니다.
   - `conf0.45` pull grid는 `0.68360`에서 포화됐지만, `selector_soft`가 Public LB `0.68440`으로 새 최고점을 만들었습니다.

9. Selector soft 후속 검증
   - temperature, top-k truncation, expanded candidate pool, seed ensemble을 확인했습니다.
   - seed ensemble blend는 `0.68440` 동률을 만들었지만, temperature/top-k와 expanded pool은 public에서 하락했습니다.

10. 새 축 재탐색
   - route gain binary fallback과 analog residual correction을 테스트했습니다.
   - OOF에서는 약한 개선 신호가 있었지만, public에서는 `0.68300 ~ 0.68360`으로 하락해 주력 축에서 내렸습니다.

11. Temporal-backcast pseudo-supervision
   - train 궤적 내부 시점 `c=6,7,8`을 현재로 보고 `c+2`를 pseudo target으로 삼아 학습 데이터를 확장했습니다.
   - 부족한 앞쪽 history는 초기 속도 기반 backcast로 채워 11점 입력 구조를 유지했습니다.
   - temporal-backcast 모델 단독은 `0.68640`, 기존 public-best anchor와 50% blend한 후보는 `0.68780`으로 새 최고점을 만들었습니다.

## 주요 인사이트

- 단순 좌표계 residual보다 마지막 속도 방향 기준 local-frame residual이 훨씬 안정적이었습니다.
- forward/side/up 축별 shrink를 다르게 주는 것이 단일 shrink보다 유리했습니다.
- retrieval은 단독 모델로는 약하지만, 일부 high-confidence 샘플 보정 재료로는 효과가 있었습니다.
- 가장 큰 돌파는 feature 수를 무작정 늘린 것이 아니라, `1cm hit 경계`를 직접 겨냥한 sample weighting에서 나왔습니다.
- 2026-05-08 기준으로는 residual 보정 축보다 pure direct-step target 전환이 더 큰 public 개선을 만들었습니다.
- Public과 CV가 완전히 일치하지 않으므로, 새 축은 빠르게 public probe하고 강한 신호가 나온 축만 확장하는 전략이 효과적입니다.
- 2026-05-10 기준 selector/routing은 큰 돌파 축은 아니지만 public에서 재현된 얇은 개선 축입니다.
- velocity smoothing/local frame denoising은 OOF proxy에서 크게 하락해 당분간 폐기합니다.
- 2026-05-11 기준 hard/threshold routing보다 selector 확률 분포를 그대로 쓰는 soft routing이 더 강했습니다.
- 2026-05-12 기준 selector soft 후처리보다 route label 설계와 hit 전환 가능성 예측이 다음 연구 우선순위입니다.
- 2026-05-15 기준 route gain과 analog residual correction은 public에서 약해, 다음은 완전히 다른 새 축을 우선합니다.
- 2026-05-16 기준 궤적 내부 pseudo-supervision을 활용하는 temporal-backcast 축이 가장 강한 새 돌파구입니다.
- temporal-backcast는 단독보다 기존 selector-soft anchor와 `50%` 전후로 섞을 때 public에서 더 강했습니다.

## Public Score 흐름

| 제출 파일 | Public score | 요약 |
|---|---:|---|
| `physics_param_search_best.csv` | 0.61539 | threshold-aware 물리 baseline |
| `aggressive_lgbm_residual.csv` | 0.63420 | LGBM residual 유효성 확인 |
| `candidate_binary_hit_selector.csv` | 0.62600 | 후보 선택기는 residual보다 약함 |
| `public_push_lgbm_wide_a0275_s0.40_5seed.csv` | 0.64120 | 5-seed residual ensemble |
| `local_frame_lgbm_a0275_s0.55_5seed.csv` | 0.65900 | local-frame residual target |
| `local_axis_rank1_f0.48_s0.55_u0.62.csv` | 0.65940 | 축별 shrink calibration |
| `retr_blend_rank1_confidentrouteblend...csv` | 0.66040 | selective retrieval route blend |
| `hit_weighted_rank1_l2_base_boundary_f0.46_s0.58_u0.70.csv` | 0.67100 | hit-boundary weighted local-frame breakthrough |
| `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv` | 0.67220 | 5-seed hit-weighted breakthrough refine |
| `regime_hit_rank1_globalhitweighted_a0.00_f0.56_s0.58_u0.70.csv` | 0.67300 | regime router 축은 큰 돌파 없이 안전 변형 |
| `direct_step_rank2_cadeltascaledlgbmboundary_f0.52_s0.58_u0.78.csv` | 0.67340 | scale-normalized CA residual LGBM |
| `direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv` | **0.67800** | pure direct-step local target 새 최고점 |
| `direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv` | 0.68240 | CA-boundary direct-step 5seed jump |
| `direct_refine_rank2_caa6s0055c0105_f1.02_s1.04_u0.96_5seed.csv` | 0.68260 | side/up multiplier tilt |
| `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv` | 0.68300 | multiplier micro-probe |
| `direct_selector_rank1_selectorconf055.csv` | 0.68340 | sample-wise multiplier selector confidence routing |
| `selector_adjust_rank1_extend115.csv` | 0.68340 | selector 방향 15% extension은 anchor와 동률 |
| `selector_adjust_rank2_shrink075.csv` | 0.68320 | selector 이동 축소는 하락 |
| `selector_adjust_rank4_conf45pull015.csv` | 0.68360 | conf0.45 route 방향 약한 보정 |
| `selector_adjust_rank5_extend130.csv` | 0.68340 | selector 방향 30% extension은 추가 개선 없음 |
| `route_refine_rank2_conf45pull200.csv` | 0.68360 | conf0.45 pull grid는 기존 best와 동률 |
| `route_refine_rank1_conf45pull100.csv` | 0.68360 | conf0.45 pull 0.10도 기존 best와 동률 |
| `route_refine_rank3_conf45pull250.csv` | 0.68360 | conf0.45 pull 0.25도 기존 best와 동률 |
| `direct_selector_rank4_selectorconf045.csv` | 0.68420 | full conf0.45 routing, 기존 best 대비 개선 |
| `direct_selector_rank2_selectorsoft.csv` | **0.68440** | 현재 최고점, probability-weighted selector soft routing |
| `softtemp_rank8_top3t100.csv` | 0.68420 | top-3 truncation은 full soft보다 약함 |
| `softtemp_rank1_softt075.csv` | 0.68420 | sharper temperature도 full soft보다 약함 |
| `seedens_rank1_seedens3.csv` | 0.68420 | selector seed ensemble 단독은 하락 |
| `seedens_rank3_seedens3blend35.csv` | **0.68440** | seed ensemble blend는 현재 최고점과 동률 |
| `expanded_selector_rank1_expandedsoftblend015.csv` | 0.68400 | expanded candidate pool은 public에서 약함 |
| `route_gain_top_candidates` | 0.68420 | route gain fallback 후보 2개 모두 best 아래 |
| `analogres_rank1_k64s010.csv` | 0.68300 | analog residual correction OOF 1위였지만 public 약세 |
| `analogres_rank2_k96s015.csv` | 0.68300 | stronger analog correction도 개선 없음 |
| `analogres_rank3_k128s010.csv` | 0.68360 | analog 계열 중 최고지만 best 미달 |
| `hitprob_rank1_anchorblendtop3p4w015.csv` | 0.68420 | 후보별 hit 확률 라우터는 best보다 약함 |
| `hitprob_extra_top3blend030.csv` | 0.68420 | hit-prob 방향을 더 키워도 개선 없음 |
| `temporalbc_rank1_anchorblend35_tbc678w020_f1.02_s1.00_u1.00.csv` | 0.68620 | temporal-backcast 축 유효성 확인 |
| `temporalbc_rank1_anchorblend50_tbc678w020_f1.02_s1.00_u1.00.csv` | **0.68780** | 현재 최고점, temporal-backcast 50% blend |
| `temporalbc_rank1_tbc678w020_f1.02_s1.00_u1.00.csv` | 0.68640 | temporal-backcast 단독은 강하지만 50% blend보다 약함 |

## 대표 실험 코드

| 파일 | 역할 |
|---|---|
| `scripts/run_physics_baselines.py` | 기본 물리 baseline 평가 및 제출 생성 |
| `scripts/search_physics_params.py` | velocity/acceleration 계수 grid search |
| `scripts/run_local_frame_residual.py` | local-frame residual target 실험 |
| `scripts/run_local_frame_axis_shrink.py` | forward/side/up 축별 shrink 탐색 |
| `scripts/run_local_frame_fine_axis_search.py` | local-axis shrink 주변 정밀 탐색 |
| `scripts/run_trajectory_retrieval.py` | 유사 궤적 retrieval/kNN 후보 생성 |
| `scripts/run_retrieval_blend_router.py` | local-frame anchor와 retrieval 선택적 blend |
| `scripts/run_retrieval_route_refine.py` | retrieval route blend 세부 탐색 |
| `scripts/run_hit_weighted_local_frame.py` | hit-boundary weighted local-frame breakthrough |
| `scripts/run_hit_weighted_breakthrough_refine.py` | 0.671 breakthrough 5-seed 안정화 |
| `scripts/run_regime_hit_weighted_router.py` | motion regime별 hit-weighted router 실험 |
| `scripts/run_direct_step_geometry.py` | direct-step target, scale-normalized residual, CatBoost/LGBM 비교 |
| `scripts/run_direct_step_refine_20260509.py` | 0.678 direct-step branch의 weight/multiplier/5seed 확장 |
| `scripts/make_direct_step_multiplier_probe_20260509.py` | 0.6826 주변 multiplier micro-probe 후보 생성 |
| `scripts/run_direct_multiplier_selector_20260510.py` | direct-step multiplier 후보를 샘플별로 고르는 selector/routing |
| `scripts/make_selector_adjustment_candidates_20260510.py` | public에서 오른 selector 방향 주변 조정 후보 생성 |
| `scripts/make_direct_velocity_smoothing_probe_20260510.py` | velocity smoothing/local frame denoising probe |
| `scripts/run_selector_route_refine_20260511.py` | conf0.45 pull grid와 continuous multiplier regression 후보 생성 |
| `scripts/run_selector_soft_temperature_20260512.py` | selector soft temperature/top-k truncation 후보 생성 |
| `scripts/run_expanded_selector_pool_20260512.py` | expanded multiplier candidate pool selector 실험 |
| `scripts/run_selector_seed_ensemble_20260512.py` | selector probability seed ensemble 후보 생성 |
| `scripts/run_route_gain_model_20260515.py` | selector soft 실패 위험 샘플 fallback 후보 생성 |
| `scripts/run_analog_residual_correction_20260515.py` | 유사 궤적 OOF 잔차 보정 후보 생성 |
| `scripts/run_hit_probability_router_20260516.py` | 후보별 1cm hit probability 라우터 실험 |
| `scripts/run_temporal_backcast_augmentation_20260516.py` | 궤적 내부 pseudo-supervision temporal-backcast 후보 생성 |
| `scripts/make_temporal_backcast_refine_candidates_20260516.py` | temporal-backcast 50% 주변 blend/refine 후보 생성 |
| `scripts/validate_submission.py` | 제출 파일 shape/null/finite/id 검증 |
| `scripts/publish_to_github.py` | 코드/리포트 범위만 GitHub commit/push |

## 프로젝트 구조

```text
dacon-mosquito-trajectory-prediction/
├── data/                  # 원본/가공 데이터, gitignore
├── docs/                  # 인수인계 및 일별 실험 요약
├── experiments/           # public score와 실험 로그
├── notebooks/             # EDA 및 모델링 노트북
├── reports/               # 최신 실험 리포트
├── scripts/               # 실험/후보 생성/검증 스크립트
├── src/                   # 공통 모듈
├── submissions/           # 제출 파일, gitignore
└── README.md
```

## 데이터 배치

DACON에서 받은 `open.zip` 압축 해제 후 아래 형태를 기대합니다.

```text
data/raw/
├── train/
│   ├── TRAIN_00001.csv
│   └── ...
├── test/
│   ├── TEST_00001.csv
│   └── ...
├── train_labels.csv
└── sample_submission.csv
```

압축 해제 결과가 `data/raw/open (3)/...`처럼 한 단계 더 들어가 있어도 주요 스크립트가 자동 탐지합니다.

## 재현 흐름

```powershell
cd C:\open\dacon-mosquito-trajectory-prediction

# 최신 breakthrough 계열 후보 생성
python scripts/run_hit_weighted_breakthrough_refine.py

# direct-step geometry 계열 후보 생성
python scripts/run_direct_step_geometry.py

# 제출 파일 검증 예시
python scripts/validate_submission.py submissions/direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv

# GitHub 업로드
python scripts/publish_to_github.py --message "Document 2026-05-08 direct-step breakthrough"
```

## 상세 기록

- [2026-05-06 local-frame 실험 정리](docs/experiment_summary_2026-05-06.md)
- [2026-05-07 hit-weighted breakthrough 정리](docs/experiment_summary_2026-05-07.md)
- [2026-05-08 direct-step target 전환 실험 정리](docs/experiment_summary_2026-05-08.md)
- [2026-05-09 direct-step refine 정리](docs/experiment_summary_2026-05-09.md)
- [2026-05-10 selector routing 실험 정리](docs/experiment_summary_2026-05-10.md)
- [2026-05-11 selector soft routing 실험 정리](docs/experiment_summary_2026-05-11.md)
- [2026-05-12 selector soft 후속 연구 정리](docs/experiment_summary_2026-05-12.md)
- [2026-05-15 새 축 재탐색 정리](docs/experiment_summary_2026-05-15.md)
- [2026-05-16 temporal-backcast breakthrough 정리](docs/experiment_summary_2026-05-16.md)
- [public score 기록](experiments/public_scores.csv)
- [hit-weighted breakthrough refine 리포트](reports/latest_hit_weighted_breakthrough_refine.md)
- [retrieval blend/router 리포트](reports/latest_retrieval_blend_router.md)
- [direct-step geometry 리포트](reports/latest_direct_step_geometry.md)
- [direct-step refine 리포트](reports/latest_direct_step_refine_20260509.md)
- [direct-step multiplier probe 리포트](reports/latest_direct_multiplier_probe_20260509.md)
- [direct multiplier selector 리포트](reports/latest_direct_multiplier_selector_20260510.md)
- [selector adjustment 리포트](reports/latest_selector_adjustments_20260510.md)
- [velocity smoothing probe 리포트](reports/latest_direct_velocity_smoothing_probe_20260510.md)
- [selector route refine 리포트](reports/latest_selector_route_refine_20260511.md)
- [selector soft temperature 리포트](reports/latest_selector_soft_temperature_20260512.md)
- [expanded selector pool 리포트](reports/latest_expanded_selector_pool_20260512.md)
- [selector seed ensemble 리포트](reports/latest_selector_seed_ensemble_20260512.md)
- [route gain model 리포트](reports/latest_route_gain_model_20260515.md)
- [analog residual correction 리포트](reports/latest_analog_residual_correction_20260515.md)
- [hit probability router 리포트](reports/latest_hit_probability_router_20260516.md)
- [temporal-backcast augmentation 리포트](reports/latest_temporal_backcast_augmentation_20260516.md)
- [temporal-backcast refine 리포트](reports/latest_temporal_backcast_refine_20260516.md)

## 비고

원본 데이터와 제출 파일은 용량 및 대회 규정 관리를 위해 GitHub에 포함하지 않습니다.  
이 저장소는 실험 코드, 핵심 결과, 재현 가능한 연구 기록을 중심으로 정리합니다.
