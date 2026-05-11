# 새 채팅 인수인계: DACON 모기 비행 궤적 예측 AI 경진대회

이 문서를 새 채팅 첫 메시지에 붙여넣으면 바로 이어서 작업할 수 있다.

## 작업 경로

- Windows: `C:\open\dacon-mosquito-trajectory-prediction`
- WSL: `/mnt/c/open/dacon-mosquito-trajectory-prediction`

## GitHub 레포

- 원격 저장소: `https://github.com/JAYUN-KIM/dacon-mosquito-trajectory-prediction`
- 브랜치: `main`

사용자가 "깃허브 업로드해줘"라고 하면, 코드/문서/리포트 변경만 정리해서 한글 커밋 메시지로 push한다. 원본 데이터와 제출 CSV는 `.gitignore`로 제외한다.

## 대회 정보

- 대회명: 모기 비행 궤적 예측 AI 경진대회
- 링크: https://dacon.io/competitions/official/236716/overview/description
- 유형: 알고리즘, 시계열, 정형, 회귀, LiDAR
- 목표: 40ms 간격 11개 과거 3D 좌표로 마지막 관측 시점 기준 +80ms 미래 좌표 `(x, y, z)` 예측
- 좌표계: sensor-local 3D coordinate
- 좌표 단위: meter
- 축 정의: `x = forward`, `y = left`, `z = up`

## 데이터 구조

다운로드한 `open.zip`을 풀면 다음 구조가 예상된다.

```text
open.zip
├── train/
│   ├── TRAIN_00001.csv
│   ├── TRAIN_00002.csv
│   └── ...
├── test/
│   ├── TEST_00001.csv
│   ├── TEST_00002.csv
│   └── ...
├── train_labels.csv
└── sample_submission.csv
```

각 train/test 파일 컬럼:

```text
timestep_ms, x, y, z
```

정답/제출 파일 컬럼:

```text
id, x, y, z
```

## 평가 지표

- R-Hit@1cm
- 예측 좌표와 실제 좌표의 3D Euclidean distance가 `0.01m` 이하이면 hit
- 최종 점수는 전체 샘플 평균 hit rate
- Public: test 중 50%
- Private: test 100%

공식 예시:

```python
import numpy as np

def r_hit(pred, true):
    R_HIT = 0.01
    distance = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)
    return np.mean(distance <= R_HIT)
```

## 규칙 요약

- 1일 최대 제출: 5회
- 사용 가능 언어: Python
- 외부 데이터 사용 가능
- 단, 평가 데이터(test)는 어떤 형태로도 모델 학습에 활용 불가
- 원격 서버 API 기반 모델 사용 불가
- 사전학습모델은 공개 가중치 및 허용 라이선스일 때만 사용 가능
- 대회 기간 중 팀 외 비공식 코드/인사이트 공유는 Private Sharing으로 간주될 수 있음

## 최신 성과

- 현재 최고 Public LB: `0.68440`
- 최고 제출 파일: `direct_selector_rank2_selectorsoft.csv`
- 갱신일: `2026-05-11`
- 핵심 축: `CA-boundary pure direct-step local target + probability-weighted selector soft routing`

주요 흐름:

1. 물리 baseline: `0.61539`
2. LGBM residual: `0.63420`
3. local-frame residual: `0.65900`
4. hit-boundary weighted local-frame: `0.67100`
5. direct-step local target: `0.67800`
6. CA-boundary direct-step 5seed/multiplier: `0.68300`
7. selector confidence routing: `0.68360`
8. selector soft routing: `0.68440`

최신 판단:

- selector/routing은 public에서 재현된 얇은 개선 축이다.
- 2026-05-11에는 threshold/hard routing보다 selector probability를 그대로 평균하는 soft routing이 더 강했다.
- `conf0.45` pull grid는 `0.68360`에서 포화됐고, `direct_selector_rank2_selectorsoft.csv`가 `0.68440`으로 최고점을 갱신했다.
- velocity smoothing/local frame denoising은 OOF proxy에서 크게 하락해 당분간 폐기한다.
- 다음은 `selector_soft temperature`, `top-k probability truncation`, `boundary-only soft routing`, `soft selector와 anchor blend`를 우선한다.

## 일정

- 시작: 2026-05-04
- 팀 병합 마감: 2026-05-25
- 종료: 2026-06-01
- 코드/PPT 제출 마감: 2026-06-04
- 코드 검증: 2026-06-12
- 최종 발표: 2026-06-15

## 초기 전략

이 문제는 짧은 시간의 미래 좌표 예측이므로, 처음부터 복잡한 딥러닝보다 물리 기반 baseline을 강하게 잡는 것이 좋다.

우선순위:

1. 데이터 로드 및 EDA
2. constant position baseline
3. constant velocity extrapolation
4. constant acceleration extrapolation
5. 최근 구간 가중 선형/2차 fitting
6. 관측 노이즈 제거 및 smoothing
7. baseline residual을 LightGBM/CatBoost로 보정
8. R-Hit@1cm에 맞춘 threshold-aware validation
9. 필요 시 GRU/TCN/Transformer 계열 sequence model 추가

## 새 채팅에서 요청할 첫 작업

```text
DACON 모기 비행 궤적 예측 대회를 시작하자.
작업 경로는 C:\open\dacon-mosquito-trajectory-prediction 이고,
인수인계 문서는 docs/new_chat_handoff.md에 있어.
먼저 레포 상태와 데이터 위치를 확인하고, 데이터가 있으면 EDA + baseline 제출 파일 생성까지 진행해줘.
데이터가 없으면 data/raw/에 어떤 구조로 넣어야 하는지 안내해줘.
```
