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

- 최고 Public LB: `0.69200`
- 최신 최고점 확인일: `2026-05-27`
- Champion: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv`
- 이전 champion: `recstep_rank4_gate_osc89b005late_f100s100u100_top080_b40.csv = 0.69180`
- alpha champion: `champmicro_rank3_gatet520a1025.csv = 0.69140`
- Follow-up 동률: `champalpha_rank1_t52a1015.csv = 0.69140`
- 이전 안정 champion: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- Backup champion: `curvgate_rank4_gatet54a105.csv`
- 안정 동률 blend:
  - `cochamp_blend_t52_t54_w50.csv = 0.69120`
  - `cochamp_blend_t52_t54_w65.csv = 0.69120`
  - `cochamp_blend_t52_t54_w35.csv = 0.69120`
- 핵심 축: recursive one-step dynamics + narrow gain-gated sample routing
- 최신 판단일: `2026-05-27`
- 최신 판단: top 9%/45% recursive gate가 `0.69200`으로 올랐지만, 47.5% 강도와 residual calibrator는 모두 동률이었다. recursive gate 미세조정은 단기 포화로 보고, 내일은 완전히 새로운 축을 우선한다.

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
| champion alpha calibration | 0.69140 | t52 gate 유지, alpha를 0.1025 근처로 낮춤 |
| recursive one-step gain gate | 0.69180 | +40ms dynamics를 두 번 적용하되 top 8% 샘플만 이동 |
| recursive gate top9 refine | 0.69200 | top 9% 샘플에 45% 이동 |

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

### 2026-05-23

- `tempcurr_rank1_tcc5678w012v6789w006champblend15f102s100u100.csv = 0.69060`
- `tempcurr_rank4_tcc678w022v89w004champblend15f102s100u100.csv = 0.69060`
- `tempcurr_rank5_tcc5678w012v6789w006cochampblend20f102s100u100.csv = 0.69060`
- `fastnew_rank2_snapsnapv102a035jm0p2d096blend18.csv = 0.68660`
- `fastnew_rank1_smoothewpolyw11d2r55blend18.csv = 0.67160`
- 결론: temporal curriculum 확장도, smoothing/denoising도, jerk/snap 물리축도 모두 아니다. 2026-05-24에는 새 좌표 후보를 바로 섞지 말고 champion miss 샘플을 regime별로 분해하는 연구부터 시작한다.

### 2026-05-24

- `regimemiss_rank1_c64_min55_net0003_p165.csv = 0.69060`
- `analogknn_rank1_k64_p10_s012_cap00025_r100.csv = 0.68860`
- `champmicro_rank1_gatet520a1075.csv = 0.69100`
- `champmicro_rank3_gatet520a1025.csv = 0.69140`
- `champalpha_rank1_t52a1015.csv = 0.69140`
- 결론: 새 miss-policy/KNN residual 축은 현재 champion을 못 이겼다. 반면 `t52` gate의 alpha를 `0.105`에서 낮추는 방향이 public에서 작지만 재현된 개선을 만들었다.

### 2026-05-25

- `hitmode_rank1_localshape_k32_s0008_clustermean_r30_b18_c00008.csv = 0.68980`
- `champalpha2_rank1_t52a1010.csv = 0.69140`
- `champalpha2_rank5_t52a1005.csv = 0.69140`
- `plateaudis_rank2_stablemeanplateau.csv = 0.69140`
- `plateaudis_rank4_towarda1005top15b50.csv = 0.69140`
- 결론: hit-mode retrieval 새 축은 하락했고, alpha/plateau disagreement 계열은 모두 `0.69140`에서 포화됐다. 이 구간을 더 파는 것은 제출권 낭비 가능성이 높다.

### 2026-05-26

- `recstep_rank4_gate_osc89b005late_f100s100u100_top080_b40.csv = 0.69180`
- recursive one-step global blend 후보들은 public에서 실패했다. 정확 점수는 별도 기록하지 않았고, 사용자 피드백은 "나머지는 꽝"이었다.
- 핵심 신호: `+40ms` one-step dynamics를 두 번 적용하는 예측값 자체가 강한 것이 아니라, gain selector가 고른 top 8% 샘플에만 40% 이동할 때만 점수가 올랐다.
- 후속 후보: `recstepgate_refine_rank1_top08_b45_f100s100u100_top080_b45.csv`, `rank2_top06_b40`, `rank3_top10_b40` 순서로 제출 후보를 만들었다.
- 결론: 다음 연구는 recursive 모델 확장보다 gate feature, 선택 비율, 이동 강도, selector calibration 쪽을 먼저 파야 한다.

### 2026-05-27

- `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- `recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv = 0.69160`
- `recstepgate27b_rank1_top090_b475_f100s100u100_top090_b475.csv = 0.69200`
- `wincal27_rank1_top03s010c0008.csv = 0.69200`
- 결론: top 9% 선택은 맞고 40%보다 45%가 좋았지만, 47.5% 강도와 residual calibrator는 추가 상승 없이 동률이었다. 이 축은 `0.69200`에서 포화로 보고 내일은 새 축을 파야 한다.

## 핵심 인사이트

- 평균 거리보다 `1cm hit 경계` 샘플을 직접 겨냥해야 점수가 오른다.
- local-frame target이 global coordinate residual보다 훨씬 강하다.
- selector hard routing보다 probability-weighted soft routing이 안정적이었다.
- temporal-backcast pseudo-supervision이 가장 큰 중반 돌파구였다.
- constant-turn curvature correction은 단독 모델이 아니라 강한 anchor 위에 작게 얹을 때만 유효하다.
- curvature gate는 `threshold=0.52~0.54`, `alpha=0.105` 부근이 안정권이다.
- current best 주변 post-process는 OOF가 좋아 보여도 public에서 과적합이 반복된다.
- temporal curriculum 확장과 smoothing/snap 물리축도 실패했다.
- 2026-05-24에는 miss regime selector도 public에서 하락했으나, `t52` gate alpha-down은 `0.69140`을 만들었다.
- 2026-05-25에는 alpha-down과 plateau disagreement가 모두 `0.69140`에 묶여 미세조정 포화가 확인됐다.
- 2026-05-26에는 recursive one-step dynamics가 global blend로는 약했지만, top 8% gain-gated routing으로 `0.69180`까지 상승했다.
- 2026-05-27에는 top 9%/45% recursive gate가 `0.69200`으로 올랐지만, 47.5% 강도와 residual calibrator가 모두 동률이라 미세조정 포화가 확인됐다.
- 다음은 recursive gate selector 주변이 아니라 새로운 target formulation, pseudo-supervision, 또는 다른 hit rescue selector 구조를 우선한다.

## 내일 연구 방향

목표는 새 최고점 `0.69200`을 anchor로 유지하되, recursive gate 미세조정은 멈추고 0.7 근처를 노릴 수 있는 새 축을 찾는 것이다.

우선순위:

1. `recstepgate27_rank1_top090_b450...`을 새 anchor로 둔다.
2. top fraction/strength 미세조정과 winner residual calibrator는 우선 중단한다.
3. 새 후보는 recursive gate와 독립적인 supervision/target formulation이어야 한다.
4. 그래도 후보를 만들 때는 `0.69200` anchor 대비 기존 hit 파괴 위험과 이동량을 먼저 기록한다.
5. public에서 새 축이 `0.6920`을 못 넘으면 빠르게 손절하고 다음 축으로 넘어간다.

## 대표 파일

| 파일 | 역할 |
|---|---|
| `README.md` | 전체 프로젝트 현황과 score 흐름 |
| `experiments/public_scores.csv` | public 제출 점수 기록 |
| `docs/experiment_summary_2026-05-27.md` | 최신 일별 정리 |
| `reports/latest_local_target_manifold_projection_20260521.md` | manifold projection 실패 리포트 |
| `reports/latest_hit_rescue_specialist_20260521.md` | hit-rescue specialist 실패 리포트 |
| `reports/latest_temporal_curriculum_fast_20260522.md` | temporal curriculum 실패 리포트 |
| `reports/latest_fast_two_new_axes_20260523.md` | smoothing/snap 물리축 실패 리포트 |
| `scripts/run_curvature_gate_20260519.py` | 현재 champion 계열 생성 스크립트 |
| `scripts/run_local_target_manifold_projection_20260521.py` | manifold projection 실험 |
| `scripts/run_hit_rescue_specialist_20260521.py` | hard-swap rescue 실험 |
| `scripts/run_temporal_curriculum_fast_20260522.py` | temporal curriculum 확장 실험 |
| `scripts/run_fast_two_new_axes_20260523.py` | smoothing/snap 새 물리축 실험 |
| `scripts/run_regime_miss_policy_20260524.py` | champion miss regime policy 실험 |
| `scripts/run_analog_knn_residual_20260524.py` | 유사 궤적 KNN residual 전이 실험 |
| `scripts/make_champion_micro_tuning_20260524.py` | t52/t54 champion 주변 미세조정 |
| `scripts/make_champion_alpha_refine_20260524.py` | public feedback 기반 t52 alpha band 후보 생성 |
| `scripts/run_local_hit_mode_retrieval_20260525.py` | hit-mode retrieval 새 축 실험 |
| `scripts/make_champion_alpha_ultrafine_20260525.py` | t52 alpha-down plateau 초미세 확인 |
| `scripts/make_plateau_disagreement_candidates_20260525.py` | plateau 후보 disagreement 저위험 실험 |
| `scripts/run_recursive_onestep_dynamics_20260526.py` | +40ms one-step dynamics recursive gate 실험 |
| `scripts/make_recursive_onestep_gate_refine_20260526.py` | 0.69180 winner 주변 gate refine 후보 생성 |
| `scripts/make_recursive_onestep_gate_jitter_20260527.py` | top 9%/45% recursive gate 후보 생성 |
| `scripts/make_recursive_onestep_gate_peak_20260527.py` | 0.69200 winner 주변 strength/fraction peak 탐색 |
| `scripts/run_winner_residual_calibrator_20260527.py` | 0.692 winner miss-risk residual calibrator 실험 |
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
