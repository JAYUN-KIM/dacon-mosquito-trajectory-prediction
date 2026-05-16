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

- 현재 최고 Public LB: `0.68780`
- 최고 제출 파일: `temporalbc_rank1_anchorblend50_tbc678w020_f1.02_s1.00_u1.00.csv`
- 최신 최고점 확인일: `2026-05-16`
- 핵심 축: `temporal-backcast pseudo-supervision + public-best anchor 50% blend`
- 최신 새 축 검토일: `2026-05-16`
- 최신 판단: temporal-backcast augmentation이 기존 selector-soft 최고 `0.68440`을 `0.68780`으로 갱신했으므로 새 주력축으로 승격

주요 흐름:

1. 물리 baseline: `0.61539`
2. LGBM residual: `0.63420`
3. local-frame residual: `0.65900`
4. hit-boundary weighted local-frame: `0.67100`
5. direct-step local target: `0.67800`
6. CA-boundary direct-step 5seed/multiplier: `0.68300`
7. selector confidence routing: `0.68360`
8. selector soft routing: `0.68440`
9. selector seed ensemble blend: `0.68440`
10. route gain / analog residual correction: `0.68360` 이하
11. temporal-backcast 50% anchor blend: `0.68780`

최신 판단:

- selector/routing은 public에서 재현된 얇은 개선 축이다.
- 2026-05-11에는 threshold/hard routing보다 selector probability를 그대로 평균하는 soft routing이 더 강했다.
- `conf0.45` pull grid는 `0.68360`에서 포화됐고, `direct_selector_rank2_selectorsoft.csv`가 `0.68440`으로 최고점을 갱신했다.
- 2026-05-12에는 temperature/top-k truncation은 `0.68420`, expanded pool은 `0.68400`, seed ensemble blend는 `0.68440`으로 최고점 동률이었다.
- 2026-05-15에는 route gain fallback 후보 2개가 `0.68420`, analog residual correction 후보 3개가 `0.68300`, `0.68300`, `0.68360`으로 나와 새 축이지만 public에서는 약했다.
- analog residual correction은 외부 논문 아이디어를 참고했지만 외부 데이터는 직접 섞지 않고 train-only nearest-neighbor residual prior로 구현했다.
- 2026-05-16에는 hit-probability router 후보 2개가 `0.68420`으로 약했지만, temporal-backcast pseudo-supervision이 `0.68620`, `0.68780`, `0.68640`을 기록하며 새 돌파구가 됐다.
- temporal-backcast 단독보다 기존 selector-soft anchor와 50% blend한 후보가 가장 강했다. 현재는 `50~55%` 주변 blend strength와 nearby temporal direction ensemble을 내일 우선 제출한다.
- velocity smoothing/local frame denoising은 OOF proxy에서 크게 하락해 당분간 폐기한다.
- 다음은 `temporalbc_refine_r1f102s100u100_w52.csv`, `temporalbc_refine_r1f102s100u100_w55.csv`, `temporalbc_refine_avgr1r2_w52.csv`, `temporalbc_refine_r2f102s104u096_w52.csv`, `temporalbc_refine_avgr1r2r3_w52.csv` 순서로 public probe한다.
- 자동화는 사용자가 돌아온 뒤 해제 요청했으므로 현재 PAUSED 상태다.

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
