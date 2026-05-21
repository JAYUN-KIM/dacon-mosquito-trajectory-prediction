# 새 채팅 인수인계: DACON 모기 비행 궤적 예측 AI 경진대회

이 문서는 새 채팅을 시작할 때 가장 먼저 읽는 인수인계 문서다.

## 작업 경로

- Windows: `C:\open\dacon-mosquito-trajectory-prediction`
- WSL: `/mnt/c/open/dacon-mosquito-trajectory-prediction`

## GitHub 저장소

- 원격 저장소: `https://github.com/JAYUN-KIM/dacon-mosquito-trajectory-prediction`
- 브랜치: `main`

사용자가 "깃허브 업로드해줘", "깃허브 정리해서 올려줘"라고 하면 코드, 문서, 리포트 변경만 정리해 커밋하고 `main`에 push한다. 원본 데이터와 제출 CSV는 `.gitignore`로 제외한다.

## 대회 정보

- 대회명: 모기 비행 궤적 예측 AI 경진대회
- 링크: https://dacon.io/competitions/official/236716/overview/description
- 목표: 40ms 간격 과거 11개 3D 좌표 `-400ms ~ 0ms`를 보고 마지막 관측 시점 기준 `+80ms` 미래 좌표 `(x, y, z)` 예측
- 좌표계: sensor-local 3D coordinate
- 좌표 단위: meter
- 축 정의: `x = forward`, `y = left`, `z = up`

## 평가 지표

- R-Hit@1cm
- 예측 좌표와 실제 좌표의 3D Euclidean distance가 `0.01m` 이하이면 hit
- Public: test 중 50%
- Private: test 100%

```python
import numpy as np

def r_hit(pred, true):
    distance = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)
    return np.mean(distance <= 0.01)
```

## 데이터 구조

현재 데이터는 `data/raw/open (3)/` 아래에 있다.

```text
data/raw/open (3)/
├── train/
│   ├── TRAIN_00001.csv
│   └── ...
├── test/
│   ├── TEST_00001.csv
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

## 규칙 요약

- 1일 최대 제출: 5회
- 사용 가능 언어: Python
- 외부 데이터 사용 가능
- 단, 평가 데이터(test)는 어떤 형태로도 모델 학습에 사용할 수 없음
- 원격 서버 API 기반 모델 사용 불가
- 사전학습모델은 공개 가중치와 허용 라이선스일 때만 사용 가능
- 대회 기간 중 비공개 코드/인사이트 공유는 Private Sharing으로 간주될 수 있음

## 현재 최고 성과

- 최고 Public LB: `0.69120`
- 최신 최고점 확인일: `2026-05-21`
- Champion: `curvgate_refine_rank2_gatet52a105.csv`
- Backup champion: `curvgate_rank4_gatet54a105.csv`
- 안정 동률 blend:
  - `cochamp_blend_t52_t54_w50.csv = 0.69120`
  - `cochamp_blend_t52_t54_w65.csv = 0.69120`
  - `cochamp_blend_t52_t54_w35.csv = 0.69120`
- 핵심 축: temporal-backcast pseudo-supervision + constant-turn curvature correction + sample-wise curvature gate

## 주요 점수 흐름

| 단계 | Public LB | 핵심 |
|---|---:|---|
| 물리 baseline | 0.61539 | constant velocity/acceleration 탐색 |
| LGBM residual | 0.63420 | 물리 residual 보정 |
| local-frame residual | 0.65900 | 마지막 속도 방향 기준 좌표계 |
| hit-boundary weighted local-frame | 0.67100 | 1cm 경계 가중치 |
| direct-step local target | 0.67800 | residual 대신 +80ms displacement 직접 예측 |
| selector confidence routing | 0.68360 | multiplier 후보 sample-wise routing |
| selector soft routing | 0.68440 | selector probability로 후보 soft average |
| temporal-backcast pseudo-supervision | 0.68780 | 궤적 내부 pseudo target 학습 |
| temporal-backcast refine | 0.68880 | temporal blend 55% 근처 |
| constant-turn curvature correction | 0.69000 | 최근 회전량 기반 nonlinear physics correction |
| curvature gate | 0.69120 | correction 적용 샘플을 gate로 선택 |

## 최근 실험 기록

### 2026-05-19

- `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- `curvgate_rank2_gatet50a105.csv = 0.69100`
- `curvgate_refine_rank6_gatet50a105low025.csv = 0.69040`
- `gate_residual_exp_sh085_cap0022.csv = 0.69040`
- 결론: low-confidence correction과 residual-on-gate는 하락. `threshold=0.50~0.52`, `alpha=0.105`가 강함.

### 2026-05-20

- `mirror_tta_rank1_mirrortta_temporal_w55_gate_t52_a105_bestblend35.csv = 0.69020`
- `multicurv_action_rank2_currentblend25actiontop3p2.csv = 0.69020`
- `mlpseq_rank2_blend08base.csv = 0.69020`
- `curvgate_rank4_gatet54a105.csv = 0.69120`
- `curvgate_refine_rank8_gatet52a110.csv = 0.69080`
- 결론: mirror-TTA, multi-curvature action router, MLP sequence blend는 모두 current best를 흐트러뜨림. `t54_a105`는 champion 동률 재현. `alpha=0.110`은 과보정.

### 2026-05-21

- `manifoldproj_rank2_k256_b060_cap0003.csv = 0.68980`
- `hitrescue_rank1_temporal55_top075.csv = 0.69060`
- `cochamp_blend_t52_t54_w50.csv = 0.69120`
- `cochamp_blend_t52_t54_w65.csv = 0.69120`
- `cochamp_blend_t52_t54_w35.csv = 0.69120`
- 결론: local target manifold projection과 hit-rescue hard swap은 실패. co-champion blend는 안정적이지만 돌파력은 없음. 내일은 post-process가 아니라 새 pseudo-label/supervision 축으로 0.7을 노려야 함.

## 핵심 인사이트

- 평균 거리보다 `1cm hit 경계` 샘플을 직접 겨냥해야 점수가 오른다.
- local-frame target이 global coordinate residual보다 훨씬 강하다.
- selector hard routing보다 probability-weighted soft routing이 안정적이었다.
- temporal-backcast pseudo-supervision이 가장 큰 중반 돌파구였다.
- constant-turn curvature correction은 단독 모델이 아니라 강한 anchor 위에 작게 얹을 때만 유효하다.
- curvature gate는 `threshold=0.52~0.54`, `alpha=0.105` 부근이 안정권이다.
- current best 주변 post-process는 OOF가 좋아 보여도 public에서 과적합이 반복된다.
- 0.7을 보려면 기존 champion을 조금 만지는 방식이 아니라 학습 데이터/타깃 정의를 다시 크게 바꿔야 한다.

## 내일 연구 방향

목표는 `0.6912` 주변 미세조정이 아니라 `0.7` 근처를 노릴 수 있는 큰 축 발견이다.

우선순위:

1. post-process 후보는 중단한다.
2. train 내부 oracle 분석으로 champion miss를 실제로 살릴 수 있는 후보군을 찾는다.
3. temporal-backcast를 확장해 다양한 cutoff, horizon, regime별 pseudo-label curriculum을 만든다.
4. 전체 sequence를 쓰는 새 supervision 모델을 만들되, public 제출 전 OOF oracle hit potential을 먼저 본다.
5. 후보를 만들 때 current best 대비 이동량보다 "추가 hit 가능성"과 "기존 hit 파괴 위험"을 같이 기록한다.

## 대표 파일

| 파일 | 역할 |
|---|---|
| `README.md` | 전체 프로젝트 현황과 score 흐름 |
| `experiments/public_scores.csv` | public 제출 점수 기록 |
| `docs/experiment_summary_2026-05-21.md` | 최신 일별 정리 |
| `reports/latest_local_target_manifold_projection_20260521.md` | manifold projection 실패 리포트 |
| `reports/latest_hit_rescue_specialist_20260521.md` | hit-rescue specialist 실패 리포트 |
| `scripts/run_curvature_gate_20260519.py` | 현재 champion 계열 생성 스크립트 |
| `scripts/run_local_target_manifold_projection_20260521.py` | manifold projection 실험 |
| `scripts/run_hit_rescue_specialist_20260521.py` | hard-swap rescue 실험 |
| `scripts/validate_submission.py` | 제출 파일 검증 |

## 재현/검증 예시

```powershell
cd C:\open\dacon-mosquito-trajectory-prediction

# 제출 파일 검증
python scripts/validate_submission.py submissions/cochamp_blend_t52_t54_w50.csv

# 최신 실패 축 리포트 확인
Get-Content reports/latest_hit_rescue_specialist_20260521.md -Encoding UTF8
```

## 비고

- `data/`, `submissions/`, `outputs/`는 GitHub에 올리지 않는다.
- 자동화는 사용자가 해제했으므로 현재는 수동 연구/업로드 흐름이다.
- 사용자는 공격적인 새 축을 선호하지만, public에서 반복 하락하면 바로 손절하는 판단을 선호한다.
