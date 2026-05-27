# 2026-05-27 recursive gate plateau와 새 축 전환 정리

## 요약

- 최고 Public LB를 `0.69180`에서 `0.69200`으로 갱신했다.
- 새 최고 파일: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv`
- 같은 top 9%에서 strength `0.400`은 `0.69160`으로 하락했고, strength `0.475`는 `0.69200` 동률이었다.
- winner residual calibrator rank1도 `0.69200` 동률이었다.
- 결론: recursive gate 주변 미세조정은 `0.69200`에서 단기 포화로 보고, 다음은 완전히 새로운 축을 찾아야 한다.

## Public 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv` | **0.69200** | top 9% 선택과 45% 이동으로 새 최고점 |
| `recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv` | 0.69160 | top 9%에서 40% 이동은 약함 |
| `recstepgate27b_rank1_top090_b475_f100s100u100_top090_b475.csv` | **0.69200** | 47.5% 이동도 동률, 강도 상승 포화 |
| `wincal27_rank1_top03s010c0008.csv` | **0.69200** | residual calibrator도 동률, 추가 보정 신호 약함 |

## 실험 1: recursive gate jitter

`scripts/make_recursive_onestep_gate_jitter_20260527.py`

- 전날 `top080_b40 = 0.69180` winner를 기준으로, 선택 비율과 이동 강도를 작게 흔들었다.
- OOF와 public winner 근접성을 함께 보고 `top090_b450`을 1순위로 제출했다.
- Public에서 `0.69200`으로 상승해 top 8%보다 top 9%가 더 좋다는 신호를 얻었다.
- 같은 top 9%에서 `b400`은 `0.69160`으로 하락해, 이동 강도는 40%보다 45%가 맞았다.

## 실험 2: recursive gate peak

`scripts/make_recursive_onestep_gate_peak_20260527.py`

- `top090_b450` 성공 후, strength를 `0.475~0.525`로 올리고 fraction을 `9.5~10%`까지 넓히는 후보를 만들었다.
- `top090_b475`는 public `0.69200` 동률이었다.
- 따라서 강도를 45%보다 더 키우는 것만으로는 추가 상승이 없다고 판단했다.

## 실험 3: winner residual calibrator

`scripts/run_winner_residual_calibrator_20260527.py`

- `0.69200` winner를 고정 anchor로 두고, OOF winner residual을 학습해 miss-risk 상위 샘플에만 mm 단위 보정을 시도했다.
- rank1 `top03_s010_c0008`은 public `0.69200` 동률이었다.
- OOF도 강한 상승이 없었고 public에서도 동률이라, 이 residual calibrator 축은 우선 손절한다.

## 다음 방향

- `recstepgate27_rank1_top090_b450...`을 새 anchor로 고정한다.
- top fraction/strength 미세조정은 추가 제출권을 쓰지 않는다.
- winner residual calibrator도 우선순위에서 내린다.
- 다음 연구는 새 target formulation, 다른 pseudo-supervision, 또는 완전히 다른 hit rescue selector 구조를 찾는다.
- 새 축 후보라도 제출 전에는 `0.69200` anchor 대비 이동량과 기존 hit 파괴 위험을 먼저 기록한다.

## 인사이트

- recursive one-step gate는 분명히 public에서 유효했지만, 하루 만에 top 9%/45% 근처에서 plateau가 확인됐다.
- `40% -> 45%`는 의미 있는 개선이었지만 `45% -> 47.5%`는 추가 개선이 없었다.
- 아주 작은 residual correction은 안전하지만 돌파력은 없었다.
- 내일은 익숙한 축을 다듬기보다 새 축 발견에 제출권을 쓰는 편이 낫다.
