# 새 채팅 인수인계: DACON 모기 비행 궤적 예측 AI 경진대회

이 문서를 새 채팅 첫 메시지에 붙여넣으면 바로 이어서 작업할 수 있다.

## 작업 경로

- Windows: `C:\open\dacon-mosquito-trajectory-prediction`
- WSL: `/mnt/c/open/dacon-mosquito-trajectory-prediction`

## GitHub 레포명 추천

- `dacon-mosquito-trajectory-prediction`

현재 로컬 레포 골격은 만들어졌지만, 이 환경에는 `gh` CLI가 없어 GitHub 원격 레포 자동 생성은 아직 못 했다. GitHub에서 빈 레포를 만든 뒤 URL을 알려주면 원격 연결과 push를 진행하면 된다.

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

