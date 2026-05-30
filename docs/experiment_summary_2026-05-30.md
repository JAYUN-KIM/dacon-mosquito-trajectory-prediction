# 2026-05-30 외부 AI 새 축 실험 정리

## 요약

- 현재 최고 Public LB는 여전히 `0.69200`이다.
- 5/29 metric-center bias와 5/30 외부 AI 제안 축을 확인했지만 모두 최고점을 넘지 못했다.
- 오늘 확인한 핵심은 기존 winner 주변의 작은 보정뿐 아니라, 외부 AI가 제안한 Regime-MoE/Analog transport 계열도 public에서 `0.6910~0.6914`로 내려간다는 점이다.
- 내일은 기존 submission, cache, OOF proxy, champion 주변 post-process를 모두 끊고 raw trajectory/label에서 clean-room 방식으로 다시 시작한다.

## Public 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `metricbias29_rank1_localfull.csv` | 0.69120 | local up +0.25mm constant bias. CV 안정성은 좋았지만 public 하락 |
| `metricbias29_rank2_localshrink075.csv` | 0.69120 | local up bias 75% shrink도 동일하게 하락 |
| `metricbias29_rank3_localshrink050.csv` | 0.69140 | bias를 더 줄여도 기존 best 미달 |
| `regimemoe30_diag_blend03_k5a05.csv` | 0.69100 | 외부 AI Regime-MoE 방향 3% 진단 blend도 하락 |
| `analog_transport_20260530_winner_move_mean_k64_p2_top050_a25.csv` | 0.69140 | orientation-equivariant analog transport의 저위험 winner-selective probe도 하락 |

## 오늘 확인한 축

1. Metric-center bias
   - OOF와 5-fold에서 local up 방향 `+0.25mm` bias가 안정적으로 좋아 보였다.
   - 하지만 public은 `0.69120~0.69140`으로 떨어졌다.
   - 결론: train OOF residual 중심과 public residual 중심이 다르거나, 이 bias가 public subset에는 맞지 않는다.

2. Regime-MoE
   - 외부 AI가 제안한 unsupervised trajectory regime clustering + per-regime Ridge MoE.
   - 순수 MoE OOF는 최고 `0.5544`로 constant velocity보다도 낮았다.
   - winner에서 MoE 방향으로 3%만 움직인 진단 blend도 public `0.69100`으로 하락했다.
   - 결론: 이 형태의 unsupervised KMeans/Ridge regime decomposition은 현재 문제의 public hit를 설명하지 못한다.

3. Orientation-equivariant analog transport
   - trajectory를 local frame으로 정렬하고, 비슷한 train shape의 future local displacement를 가져오는 방식.
   - pure analog OOF는 `0.1806~0.2178`로 매우 낮았다.
   - winner-selective low-risk probe도 public `0.69140`으로 하락했다.
   - 결론: nearest analog future displacement는 public winner의 부족분을 보완하지 못한다.

## 핵심 판단

- 최근 3일간 `0.69200` 주변에서 보정/selector/retrieval/analog 축이 모두 막혔다.
- OOF가 좋아 보인 축도 public에서 반복적으로 하락했다.
- 현재 접근이 기존 best submission과 과거 실험 기억을 중심으로 “근처 보정”을 반복하면서 0.691~0.692 근방에 갇힌 가능성이 높다.
- 다음 실험은 기존 submission/candidate/cache를 anchor로 사용하지 않는 clean-room 연구가 필요하다.

## 내일 clean-room 원칙

- 사용 가능:
  - `data/raw/open (3)`의 raw train/test trajectory
  - `train_labels.csv`
  - `sample_submission.csv`
  - 대회 정의와 metric
- 사용 금지:
  - 기존 best submission을 anchor로 사용하는 보정
  - 기존 OOF cache
  - 기존 generated submissions의 blend/average/post-process
  - 이전 실험의 threshold/top-fraction/alpha를 그대로 가져오는 미세조정
- 목표:
  - raw 11-point trajectory에서 feature/target/objective를 새로 설계한다.
  - 최소한 하나의 완전 독립 OOF pipeline을 만들고, 그 결과가 약하면 과감히 버린다.

## 관련 산출물

| 파일 | 역할 |
|---|---|
| `scripts/run_metric_center_bias_20260529.py` | local/global/regime metric-center bias 실험 |
| `scripts/run_analog_transport_20260530.py` | orientation-equivariant analog transport 실험 |
| `reports/latest_metric_center_bias_20260529.md` | metric-center bias 리포트 |
| `reports/regime_moe_20260530.md` | 외부 AI Regime-MoE 실행 리포트 |
| `reports/latest_regime_moe_diagnostic_blend_20260530.md` | Regime-MoE winner 진단 blend 리포트 |
| `reports/analog_transport_20260530.md` | analog transport 리포트 |
| `reports/analog_transport_20260530_metrics.json` | analog transport 후보 메트릭 |

