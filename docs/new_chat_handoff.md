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

- 현재 최고 Public LB: `0.69120`
- 최고 제출 파일: `curvgate_refine_rank2_gatet52a105.csv`
- 보조 최고 제출 파일: `curvgate_rank4_gatet54a105.csv`
- 최신 최고점 확인일: `2026-05-20`
- 핵심 축: `temporal-backcast pseudo-supervision + constant-turn curvature correction + sample-wise curvature gate`
- 최신 새 축 검토일: `2026-05-20`
- 최신 판단: mirror-TTA, multi-curvature action router, MLP sequence pseudo-supervision은 모두 첫 public probe가 `0.69020`으로 하락했다. 반면 기존 curvature gate `threshold=0.54`, `alpha=0.105`가 `0.69120` 동률을 재현했다. 현재는 새 축을 바로 제출하기보다 검증 실패 원인 분석과 gate 주변 정밀 검증이 우선이다.

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
11. temporal-backcast 55% anchor blend: `0.68880`
12. constant-turn curvature correction: `0.69000`
13. curvature gate: `0.69120`

최신 판단:

- selector/routing은 public에서 재현된 얇은 개선 축이다.
- 2026-05-11에는 threshold/hard routing보다 selector probability를 그대로 평균하는 soft routing이 더 강했다.
- `conf0.45` pull grid는 `0.68360`에서 포화됐고, `direct_selector_rank2_selectorsoft.csv`가 `0.68440`으로 최고점을 갱신했다.
- 2026-05-12에는 temperature/top-k truncation은 `0.68420`, expanded pool은 `0.68400`, seed ensemble blend는 `0.68440`으로 최고점 동률이었다.
- 2026-05-15에는 route gain fallback 후보 2개가 `0.68420`, analog residual correction 후보 3개가 `0.68300`, `0.68300`, `0.68360`으로 나와 새 축이지만 public에서는 약했다.
- analog residual correction은 외부 논문 아이디어를 참고했지만 외부 데이터는 직접 섞지 않고 train-only nearest-neighbor residual prior로 구현했다.
- 2026-05-16에는 hit-probability router 후보 2개가 `0.68420`으로 약했지만, temporal-backcast pseudo-supervision이 `0.68620`, `0.68780`, `0.68640`을 기록하며 새 돌파구가 됐다.
- 2026-05-17/18에는 temporal-backcast refine으로 `w52=0.68800`, `w55=0.68880`, `avg_r1r2_w52=0.68820`, `truew555=0.68860`을 확인했다. temporal-only 최적은 55% 근처로 보인다.
- 2026-05-18에는 constant-turn curvature correction을 새 축으로 추가했다. `a08=0.68940`, `a09=0.69000`, `a10=0.68960`으로 alpha 0.09가 현재 최고다.
- 2026-05-19에는 curvature correction을 적용할 샘플을 고르는 gate 모델을 추가했다. `gate_t50_a105=0.69100`, `gate_t38_a105=0.69000`, `gate_t52_a105=0.69120`으로 threshold 0.50~0.52가 강했다.
- low-confidence 샘플에 약한 correction을 남긴 `gate_t50_a105_low025`와 residual-on-gate 후보는 각각 `0.69040`으로 하락했다. 현재는 residual 보정보다 multi-curvature gate 또는 alpha bucket policy가 우선이다.
- 2026-05-20에는 mirror-symmetry temporal TTA, multi-curvature action router, MLP sequence pseudo-supervision을 새 축으로 시도했지만 모두 `0.69020`으로 하락했다. 서로 다른 축이 같은 수준으로 하락해 current best를 조금만 흐트러뜨려도 hit가 깨지는 구간으로 판단한다.
- 같은 날 검증된 curvature gate 주변으로 돌아와 `curvgate_rank4_gatet54a105.csv=0.69120`을 확인했다. `curvgate_refine_rank8_gatet52a110.csv=0.69080`으로 alpha 0.110은 과보정이다.
- velocity smoothing/local frame denoising은 OOF proxy에서 크게 하락해 당분간 폐기한다.
- 다음은 새 제출 파일을 바로 만들기보다, public 성공/실패 후보의 current best 대비 이동 벡터 분포를 비교해 실패 패턴을 자동 필터링하는 진단을 우선한다. 제출은 `threshold=0.52~0.54`, `alpha=0.102~0.106` 근처만 매우 조심스럽게 확인한다.
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
