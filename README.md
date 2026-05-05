# DACON 모기 비행 궤적 예측 AI 경진대회

40ms 간격으로 관측된 과거 11개 3D 좌표를 활용해 모기의 `+80ms` 미래 위치를 예측하는 프로젝트입니다.

짧은 시계열 궤적에서 물리 기반 extrapolation을 강한 기준점으로 잡고, local-frame residual 모델링을 통해 `R-Hit@1cm`를 높이는 것을 목표로 합니다.

## 프로젝트 개요

- 대회: 모기 비행 궤적 예측 AI 경진대회
- 플랫폼: DACON
- 대회 링크: https://dacon.io/competitions/official/236716/overview/description
- 평가 지표: R-Hit@1cm
- 입력 데이터: 40ms 간격 11개 과거 3D 좌표 `-400ms ~ 0ms`
- 목표: 마지막 관측 시점 기준 `+80ms` 미래 좌표 `(x, y, z)` 예측
- 좌표 단위: meter
- hit 기준: 예측 좌표와 실제 좌표의 3D Euclidean distance가 `0.01m` 이하

## 현재 성과

<!-- AUTO:PROJECT_STATUS:START -->
- 최고 Public LB: **0.65940**
- 최신 최고점 갱신일: **2026-05-06**
- 핵심 개선 축: 마지막 속도 방향 기준 local-frame residual target과 축별 shrink calibration
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
   - constant position, constant velocity, constant acceleration, polynomial extrapolation을 먼저 비교했습니다.
   - 초기부터 평균 거리보다 `R-Hit@1cm`를 우선 지표로 보고 후보를 선택했습니다.

2. Threshold-aware physics parameter search
   - 마지막 velocity와 acceleration 보정 계수를 직접 grid search했습니다.
   - 단순 velocity extrapolation보다 약한 acceleration 보정이 public과 CV에서 더 좋은 신호를 보였습니다.

3. LightGBM residual modeling
   - 물리 baseline 예측값을 anchor로 두고, 정답과 baseline의 residual을 LightGBM으로 보정했습니다.
   - residual 예측값은 그대로 더하지 않고 shrink를 적용해 과보정을 줄였습니다.

4. Feature-rich residual
   - 여러 물리 후보, polynomial extrapolation, weighted recent velocity, 후보 간 spread를 feature로 추가했습니다.
   - 단순 raw 좌표 feature보다 후보 기반 feature가 residual 모델에 더 유용했습니다.

5. Local-frame residual target
   - 마지막 속도 방향을 forward 축으로 하는 local coordinate frame을 구성했습니다.
   - residual을 global `x, y, z`가 아니라 `진행방향 / 횡방향 / 상하방향`으로 변환해 예측했습니다.
   - 이 전환에서 public score가 `0.64120`에서 `0.65900`으로 크게 개선됐습니다.

6. Axis-wise shrink calibration
   - local-frame residual에 하나의 shrink만 쓰지 않고, forward/side/up 축별로 shrink를 다르게 적용했습니다.
   - 진행방향 residual은 덜 믿고, 횡방향/상하방향 residual은 더 살리는 구성이 가장 좋았습니다.

## 주요 인사이트

- 이 문제는 딥러닝보다 강한 물리 baseline과 residual 보정이 먼저 먹혔습니다.
- 평균 거리 최소화보다 `1cm` 안에 들어오는 hit rate를 직접 보는 것이 중요했습니다.
- global 좌표계 residual보다 마지막 속도 방향 기준 local-frame residual이 훨씬 안정적이었습니다.
- 진행방향 위치는 물리 baseline이 이미 잘 잡고 있어 모델 보정을 과하게 넣으면 흔들릴 수 있습니다.
- 횡방향/상하방향 residual은 미세한 곡률과 흔들림을 모델이 보정할 여지가 더 컸습니다.
- 후보 선택기 방식은 oracle headroom은 있었지만, 현재 feature로는 residual 모델보다 약했습니다.
- 당분간은 새로운 모델 계열보다 local-frame target, local-axis feature, 축별 calibration을 더 파는 것이 효율적입니다.

## Public Score 흐름

| 제출 파일 | Public score | 요약 |
|---|---:|---|
| `physics_param_search_best.csv` | 0.61539 | threshold-aware 물리 baseline |
| `aggressive_lgbm_residual.csv` | 0.63420 | LGBM residual이 public에서 유효함을 확인 |
| `candidate_binary_hit_selector.csv` | 0.62600 | 후보 선택기는 현재 feature로는 부족 |
| `residual_zoo_rank1_lgbm_wide_a0275_s0.25.csv` | 0.63480 | LGBM wide residual 소폭 개선 |
| `public_push_lgbm_wide_a0275_s0.40_5seed.csv` | 0.64120 | 5-seed residual ensemble |
| `local_frame_lgbm_a0275_s0.55_5seed.csv` | 0.65900 | local-frame residual target으로 큰 점프 |
| `local_axis_rank1_f0.48_s0.55_u0.62.csv` | 0.65940 | local-frame 축별 shrink로 추가 개선 |

## 대표 실험 코드

| 파일 | 역할 |
|---|---|
| `scripts/run_physics_baselines.py` | 기본 물리 baseline 평가 및 제출 생성 |
| `scripts/search_physics_params.py` | velocity/acceleration 계수 grid search |
| `scripts/run_aggressive_experiments.py` | multi-seed physics CV 및 LGBM residual 실험 |
| `scripts/run_residual_model_zoo.py` | LGBM/CatBoost residual model zoo 비교 |
| `scripts/run_public_push_residual.py` | public에서 강한 residual family의 5-seed 확장 |
| `scripts/run_feature_rich_residual.py` | 물리/다항식 후보 기반 feature-rich residual 실험 |
| `scripts/run_local_frame_residual.py` | local-frame residual target 실험 |
| `scripts/run_local_frame_axis_shrink.py` | forward/side/up 축별 shrink 탐색 |
| `scripts/blend_submissions.py` | 제출 파일 좌표 블렌딩 |
| `scripts/validate_submission.py` | 제출 파일 shape/null/finite/id 검증 |

## 프로젝트 구조

```text
dacon-mosquito-trajectory-prediction/
├── data/                  # 원본/가공 데이터, gitignore
├── docs/                  # 인수인계 및 실험 요약
├── experiments/           # public score와 실험 로그
├── notebooks/             # EDA 및 모델링 노트북
├── reports/               # 최신 실험 리포트
├── scripts/               # 실험/후보 생성/검증 스크립트
├── src/                   # 공통 모듈
├── submissions/           # 제출 파일, gitignore
└── README.md
```

## 데이터 배치

DACON에서 받은 `open.zip` 압축을 풀어 아래 형태로 둡니다.

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

압축 해제 결과가 `data/raw/open (3)/...`처럼 한 단계 안쪽에 있어도 주요 스크립트가 자동 탐지합니다.

각 train/test CSV 컬럼:

```text
timestep_ms, x, y, z
```

`train_labels.csv`, `sample_submission.csv` 컬럼:

```text
id, x, y, z
```

## 재현 흐름

```powershell
cd C:\open\dacon-mosquito-trajectory-prediction

# 기본 물리 baseline
python scripts/run_physics_baselines.py

# threshold-aware 물리 파라미터 탐색
python scripts/search_physics_params.py

# local-frame residual 실험
python scripts/run_local_frame_residual.py

# local-frame 축별 shrink 탐색
python scripts/run_local_frame_axis_shrink.py

# 제출 파일 검증
python scripts/validate_submission.py submissions/local_axis_rank1_f0.48_s0.55_u0.62.csv
```

## 상세 기록

- [2026-05-06 local-frame 실험 정리](docs/experiment_summary_2026-05-06.md)
- [public score 기록](experiments/public_scores.csv)
- [최신 local-frame axis shrink 리포트](reports/latest_local_frame_axis_shrink.md)
- [실험 로그 JSON](experiments/log.json)

## 비고

원본 데이터와 제출 파일은 용량 및 대회 규정 관리를 위해 GitHub에 포함하지 않습니다.

이 저장소는 실험 코드, 핵심 결과, 재현 가능한 연구 기록을 중심으로 정리합니다.
