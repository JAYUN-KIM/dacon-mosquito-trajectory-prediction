# 2026-05-19 curvature gate breakthrough 정리

## 요약

- 5/18 최고점은 `turncurve_refine_temporalbest_w1tm0p25s0p5d0p98_a09.csv = 0.6900`이었다.
- 오늘은 같은 curvature correction을 모든 샘플에 적용하는 대신, 샘플별로 correction 적용 여부를 고르는 `curvature gate`를 새 축으로 실험했다.
- OOF에서 `alpha=0.09` correction이 anchor보다 가까워지는 샘플을 label로 만들고, motion/curvature feature로 LightGBM gate를 학습했다.
- Public에서는 `gate_t50_a105`가 `0.6910`, 이후 threshold를 살짝 올린 `gate_t52_a105`가 `0.6912`로 새 최고점을 만들었다.
- 마지막 도전으로 residual-on-gate 보정을 시도했지만 public은 `0.6904`로 하락했다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `curvgate_rank2_gatet50a105.csv` | 0.6910 | gate probability 0.50 이상 샘플에 alpha 0.105 적용 |
| `curvgate_rank3_gatet38a105.csv` | 0.6900 | threshold를 0.38로 너무 넓히면 기존 curvature best 수준으로 하락 |
| `curvgate_refine_rank2_gatet52a105.csv` | **0.6912** | 5/19 기준 새 최고점, threshold 0.52가 가장 좋았음 |
| `curvgate_refine_rank6_gatet50a105low025.csv` | 0.6904 | low-confidence 샘플에 약한 correction을 남긴 변형은 하락 |
| `gate_residual_exp_sh085_cap0022.csv` | 0.6904 | residual-on-gate 새 축은 public에서 미재현 |

## 실험 축

1. Curvature gate
   - temporal-best anchor와 constant-turn correction을 기준으로 OOF label을 만들었다.
   - label은 `fixed alpha=0.09` correction이 anchor보다 실제 정답에 가까워지는지 여부다.
   - gate feature는 기존 motion feature, 최근 회전 벡터, correction 크기, anchor/correction 방향 관계를 포함했다.

2. Public-guided gate refine
   - `gate_t50_a105 = 0.6910`, `gate_t38_a105 = 0.6900` 결과를 보고 넓은 적용보다 중간 확신 구간이 유효하다고 판단했다.
   - threshold `0.48`, `0.52`, `0.56`과 alpha `0.100`, `0.110`을 추가 생성했다.
   - 최종적으로 `threshold=0.52`, `alpha=0.105`가 `0.6912`로 가장 강했다.

3. Gate residual correction
   - gate base 이후 남은 train OOF residual vector를 학습해 아주 작게 보정했다.
   - residual vector는 cap을 걸어 평균 이동량을 매우 작게 제한했다.
   - OOF에서는 미세 개선이 있었지만 public은 `0.6904`로 하락해 과적합 또는 public/private 분포 차이 가능성이 있다.

## 핵심 인사이트

- curvature correction 자체는 여전히 강하지만, 모든 샘플에 동일하게 적용하는 것보다 적용 샘플을 고르는 gate가 한 단계 더 좋았다.
- 너무 넓게 적용하는 threshold `0.38`은 `0.6900`으로 하락해, correction이 오히려 손해인 샘플이 꽤 존재한다.
- `threshold=0.50~0.52`, `alpha=0.105`가 현재 가장 안정적인 영역으로 보인다.
- low-confidence 샘플에 약한 correction을 남기는 방식은 실패했다. correction을 안 줄 샘플은 확실히 안 주는 편이 낫다.
- residual-on-gate는 OOF 개선이 public으로 이어지지 않았다. 현재는 residual 보정보다 gate policy와 curvature config 확장이 우선이다.

## 생성 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_curvature_gate_20260519.py` | temporal/selector OOF anchor 위 curvature gate 학습 및 후보 생성 |
| `scripts/make_curvature_gate_refine_candidates_20260519.py` | public 신호 기반 threshold/alpha refine 후보 생성 |
| `scripts/make_gate_residual_experimental_20260519.py` | curvature gate 이후 residual vector correction 실험 |
| `reports/latest_curvature_gate_20260519.md` | curvature gate OOF/후보 리포트 |
| `reports/latest_curvature_gate_refine_20260519.md` | public-guided gate refine 리포트 |
| `reports/latest_gate_residual_experimental_20260519.md` | residual-on-gate 실험 리포트 |

## 다음 방향

1. `threshold=0.50~0.54`, `alpha=0.100~0.110` 주변을 더 정교하게 보되, 제출 수를 많이 쓰기보다 새 curvature config를 같이 섞는다.
2. 현재 gate label을 alpha=0.09 하나가 아니라, 여러 alpha 중 최적 alpha bucket을 고르는 multi-class policy로 바꾼다.
3. `w1_tm0p25_s0p5_d0p98` 외의 turn config를 gate feature와 후보 correction으로 추가해 multi-curvature gate를 만든다.
4. residual-on-gate는 당분간 후순위로 내리고, public에서 검증된 gate policy/curvature physics 축을 더 세게 민다.
