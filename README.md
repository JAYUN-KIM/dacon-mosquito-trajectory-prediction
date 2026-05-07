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
- 최고 Public LB: **0.67800**
- 최신 최고점 갱신일: **2026-05-08**
- 핵심 개선 축: scale-normalized pure direct-step local target + hit-boundary weighting
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

## 주요 인사이트

- 단순 좌표계 residual보다 마지막 속도 방향 기준 local-frame residual이 훨씬 안정적이었습니다.
- forward/side/up 축별 shrink를 다르게 주는 것이 단일 shrink보다 유리했습니다.
- retrieval은 단독 모델로는 약하지만, 일부 high-confidence 샘플 보정 재료로는 효과가 있었습니다.
- 가장 큰 돌파는 feature 수를 무작정 늘린 것이 아니라, `1cm hit 경계`를 직접 겨냥한 sample weighting에서 나왔습니다.
- 2026-05-08 기준으로는 residual 보정 축보다 pure direct-step target 전환이 더 큰 public 개선을 만들었습니다.
- Public과 CV가 완전히 일치하지 않으므로, 새 축은 빠르게 public probe하고 강한 신호가 나온 축만 확장하는 전략이 효과적입니다.

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
- [public score 기록](experiments/public_scores.csv)
- [hit-weighted breakthrough refine 리포트](reports/latest_hit_weighted_breakthrough_refine.md)
- [retrieval blend/router 리포트](reports/latest_retrieval_blend_router.md)
- [direct-step geometry 리포트](reports/latest_direct_step_geometry.md)

## 비고

원본 데이터와 제출 파일은 용량 및 대회 규정 관리를 위해 GitHub에 포함하지 않습니다.  
이 저장소는 실험 코드, 핵심 결과, 재현 가능한 연구 기록을 중심으로 정리합니다.
