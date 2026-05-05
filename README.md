# DACON 모기 비행 궤적 예측

DACON `모기 비행 궤적 예측 AI 경진대회` 실험 레포지토리입니다.

과거 400ms 동안의 3D 좌표 11개를 이용해 마지막 관측 시점 기준 `+80ms` 미래 좌표 `(x, y, z)`를 예측합니다.

- 대회: [모기 비행 궤적 예측 AI 경진대회](https://dacon.io/competitions/official/236716/overview/description)
- 평가: `R-Hit@1cm`
- hit 기준: 예측 좌표와 정답 좌표의 3D Euclidean distance가 `0.01m` 이하
- 현재 최고 public score: `0.65940`

## 현재 핵심 결론

현재 가장 강한 방향은 `물리 기반 baseline + local-frame residual 모델링`입니다.

초기에는 constant velocity, acceleration, polynomial extrapolation을 비교했고, 이후 LightGBM residual을 얹어 public score를 크게 올렸습니다. 2026-05-06 기준으로 가장 중요한 개선은 마지막 속도 방향을 기준으로 local coordinate frame을 만든 뒤 residual을 `진행방향 / 횡방향 / 상하방향`으로 예측한 것입니다.

현재 해석은 다음과 같습니다.

- 진행방향 위치는 물리 baseline이 이미 꽤 잘 잡는다.
- 모델 보정은 진행방향보다 횡방향/상하방향 residual에서 더 효과적이다.
- global `x, y, z` residual보다 local-frame residual target이 훨씬 안정적이다.
- 당분간 딥러닝 전환보다 local-frame feature, target, axis calibration을 더 파는 것이 효율적이다.

## Public Score 기록

| 제출 파일 | Public score | 요약 |
| --- | ---: | --- |
| `physics_param_search_best.csv` | 0.61539 | threshold-aware 물리 baseline |
| `aggressive_lgbm_residual.csv` | 0.63420 | LightGBM residual이 public에서 유효함을 확인 |
| `candidate_binary_hit_selector.csv` | 0.62600 | 후보 선택기는 현재 feature로는 부족 |
| `residual_zoo_rank1_lgbm_wide_a0275_s0.25.csv` | 0.63480 | LGBM wide residual 소폭 개선 |
| `public_push_lgbm_wide_a0275_s0.40_5seed.csv` | 0.64120 | 5-seed residual ensemble |
| `local_frame_lgbm_a0275_s0.55_5seed.csv` | 0.65900 | local-frame residual target으로 큰 점프 |
| `local_axis_rank1_f0.48_s0.55_u0.62.csv` | 0.65940 | local-frame 축별 shrink로 추가 개선 |

상세 정리는 [2026-05-06 실험 정리](docs/experiment_summary_2026-05-06.md)를 참고합니다.

## 문제 구조

각 샘플은 40ms 간격의 좌표 11개로 구성됩니다.

```text
Input  : -400ms, -360ms, ..., -40ms, 0ms
Target : +80ms 좌표
```

좌표계는 sensor-local 3D coordinate입니다.

- `x`: forward
- `y`: left
- `z`: up
- 단위: meter

평균 거리 오차만 낮추는 것이 목표가 아닙니다. 최종 평가는 `1cm` 이내 hit rate이므로, 평균 거리보다 `R-Hit@1cm`를 우선 기준으로 봅니다.

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

## 주요 실행 명령

제출 파일 형식 검증:

```powershell
python scripts/validate_submission.py submissions/example.csv
```

기본 물리 baseline 실행:

```powershell
python scripts/run_physics_baselines.py
```

전체 자동 파이프라인:

```powershell
python scripts/auto_pipeline.py
```

threshold-aware 물리 파라미터 탐색:

```powershell
python scripts/search_physics_params.py
```

LightGBM residual 공격 실험:

```powershell
python scripts/run_aggressive_experiments.py
```

residual model zoo:

```powershell
python scripts/run_residual_model_zoo.py
```

5-seed public-push residual:

```powershell
python scripts/run_public_push_residual.py
```

feature-rich residual:

```powershell
python scripts/run_feature_rich_residual.py
```

local-frame residual:

```powershell
python scripts/run_local_frame_residual.py
```

local-frame 축별 shrink 탐색:

```powershell
python scripts/run_local_frame_axis_shrink.py
```

제출 파일 블렌딩:

```powershell
python scripts/blend_submissions.py --left submissions/aggressive_lgbm_residual.csv --right submissions/residual_zoo_rank1_lgbm_wide_a0275_s0.25.csv --left-weight 0.7 --output submissions/blend_lgbm_residual_zoo_rank1_w70.csv
```

## 레포 구조

```text
dacon-mosquito-trajectory-prediction/
├── data/                  # raw/processed 데이터, git 제외
├── docs/                  # 인수인계 및 실험 요약
├── experiments/           # public score와 실험 로그
├── notebooks/             # EDA 노트북
├── reports/               # 최신 실험 리포트
├── scripts/               # 실행 스크립트
├── src/                   # 재사용 모듈
├── submissions/           # 제출 파일, git 제외
└── README.md
```

## GitHub 업로드 정책

GitHub에는 재현 가능한 코드, 리포트, 실험 기록만 올립니다.

올리지 않는 것:

- `data/raw/`
- `data/processed/`
- `submissions/`
- 모델 artifact

업로드 요청 시에는 코드 검증 후 한글 요약과 함께 commit/push합니다.
