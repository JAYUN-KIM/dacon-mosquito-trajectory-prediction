# 2026-05-15 새 축 재탐색 정리

## 요약

- 5/12 이후 `0.68440` 최고점 근처에서 selector soft 계열 미세 조정이 막혔다.
- 오늘은 기존 multiplier/selector 미세 조정 대신, 대회 정의와 규칙을 다시 확인하고 외부 논문 아이디어를 참고해 새 축을 시도했다.
- DACON 규칙상 외부 데이터 사용은 허용되지만 test 데이터는 학습에 사용할 수 없고, 원격 API 기반 모델도 사용할 수 없다.
- 따라서 외부 모기 궤적 데이터를 직접 섞지는 않고, 논문에서 자주 보이는 동역학/칼만/유사 궤적 아이디어만 train-only 방식으로 구현했다.
- 결과적으로 analog residual correction 계열은 public에서 `0.68300 ~ 0.68360`으로 약했고, 현재 best `0.68440`을 넘지 못했다.

## 제출 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `analogres_rank1_k64s010.csv` | 0.68300 | OOF 1위였지만 public에서는 약함 |
| `analogres_rank2_k96s015.csv` | 0.68300 | correction 강도 증가도 개선 없음 |
| `analogres_rank3_k128s010.csv` | 0.68360 | analog 계열 중 public 최고지만 best에는 못 미침 |

## 실험 축

1. Route gain binary model
   - selector soft가 틀릴 것 같은 샘플을 예측해 기존 current/conf45 후보로 fallback하는 구조를 테스트했다.
   - OOF에서는 current fallback이 selector soft보다 살짝 좋아 보였지만, public 제출 후보 2개는 모두 `0.68420`으로 best 아래였다.
   - 결론적으로 route gain은 “위험 샘플 감지” 자체가 public split에서 충분히 강하지 않았다.

2. Analog residual correction
   - 정규화된 최근 displacement, acceleration, speed, turn cosine, selector probability를 motion descriptor로 만들었다.
   - train OOF에서 selector-soft 잔차를 계산한 뒤, test와 비슷한 train 궤적의 잔차를 최근접 이웃 방식으로 작게 더했다.
   - OOF 평균은 `k64_s010`이 anchor 대비 `+0.001267` 개선 신호가 있었지만, public에서는 `0.68300`으로 하락했다.
   - correction 이동량이 평균 `0.0001m` 수준이라 너무 약했고, 더 키우면 train analog noise가 들어올 가능성이 커 보인다.

## 외부 자료 검토

- DACON 규칙은 외부 데이터 사용을 허용하지만, 제공 test 데이터는 어떤 형태로도 학습에 사용할 수 없다고 명시한다.
- 모기 비행/추적 문헌에서는 3D 좌표, 속도, 가속도, 칼만 필터, 유사 궤적 기반 예측 아이디어가 반복적으로 등장한다.
- 다만 공개 모기 궤적 데이터는 센서, 좌표계, 실험 환경이 대회 데이터와 다를 가능성이 크다.
- 현재 단계에서는 외부 데이터를 직접 합치기보다, 논문 아이디어를 대회 train 데이터 안에서 재현하는 편이 더 안전하다.

참고:

- https://dacon.io/competitions/official/236716/overview/rules
- https://dacon.io/competitions/official/236716/overview/evaluation
- https://arxiv.org/abs/2505.13615
- https://www.mdpi.com/2079-9292/14/7/1333
- https://arxiv.org/abs/2007.14216

## 인사이트

- selector soft 이후의 단순 후처리, route fallback, analog residual 보정은 모두 best를 넘지 못했다.
- OOF에서 보이는 미세 개선은 public에서 쉽게 깨지고 있어, 지금은 0.684 부근의 미세 조정보다 더 큰 구조 변화가 필요하다.
- analog 방식은 “비슷한 궤적은 비슷한 오차를 낸다”는 가설이었지만, 현재 feature/거리공간으로는 public 일반화가 약했다.
- 내일은 이 축을 더 키우기보다 완전히 다른 표현학습/시퀀스 모델/확률적 hit-region 최적화 쪽을 새로 파는 편이 좋다.

## 생성 코드와 리포트

| 파일 | 역할 |
|---|---|
| `scripts/run_route_gain_model_20260515.py` | selector soft 실패 샘플을 예측해 fallback하는 route gain 후보 생성 |
| `reports/latest_route_gain_model_20260515.md` | route gain OOF 진단과 후보 이동량 기록 |
| `scripts/run_analog_residual_correction_20260515.py` | 최근접 유사 궤적 잔차 보정 후보 생성 |
| `reports/latest_analog_residual_correction_20260515.md` | analog residual CV, 제출 후보, 외부 자료 검토 기록 |

## 다음 방향

1. 현재 best anchor는 계속 `direct_selector_rank2_selectorsoft.csv = 0.68440`으로 둔다.
2. route gain과 analog residual correction은 당분간 주력 축에서 내린다.
3. 다음 실험은 새 축 우선이다. 후보는 sequence representation learning, pseudo hit probability calibration, axis-wise uncertainty modeling, hidden regime clustering 순서로 본다.
4. 특히 평균 거리 개선보다 `1cm 안쪽으로 들어올 가능성이 있는 샘플`을 직접 분류/보정하는 목적 함수를 다시 설계한다.
