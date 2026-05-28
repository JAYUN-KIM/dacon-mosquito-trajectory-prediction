# 2026-05-28 새 축 재탐색 정리

## 요약

- 기존 최고 Public LB는 `0.69200`이며, 5/28 실험에서는 이를 넘지 못했다.
- recursive gate, winner residual, self-consistency, wide action selector 모두 `0.69200` 근처에서 막히는 흐름이 이어졌다.
- 오늘 가장 의미 있었던 점은 “강한 OOF 신호가 public으로 바로 이어지지 않는다”는 것을 다시 확인한 것이다.
- 내일은 같은 계열 미세조정보다 외부 관점의 조언을 받아 문제 재정의와 새 feature/target formulation을 다시 잡는 편이 낫다.

## Public 결과

| 제출 파일 | Public LB | 해석 |
|---|---:|---|
| `selfcons28_rank1_top05b016c0018t090.csv` | 0.69180 | 내부 과거 구간 self-consistency 물리 라우터. 안전하지만 breakthrough 없음 |
| `selfcons28_rank2_top025b030c0025t085.csv` | 0.69200 | self-consistency를 더 공격적으로 적용했지만 기존 최고점과 동률 |
| `wideact28_rank1_softblend08.csv` | 0.69180 | wide physics action selector. OOF는 강했지만 public에서는 소폭 하락 |
| `wideact28_refine_softblend06.csv` | 0.69140 | wide action 방향을 약하게 줄인 안전 버전도 하락 |
| `wide_action_final_probe_unconfirmed` | 0.69020 | 마지막 wide-action 계열 추가 제출. 정확한 파일명은 다음 대화에서 확인 필요 |

## 오늘 시도한 축

1. Local-axis scale calibrator
   - 현재 winner의 +80ms 변위를 마지막 속도 기준 local frame의 forward/side/up 축으로 나눠, 축별 multiplicative scale을 보정했다.
   - OOF상 winner 대비 `-0.0001` 수준이라 제출 우선순위에서 제외했다.

2. Self-consistency physics router
   - 각 trajectory 내부의 과거 prefix로 후보 물리식이 최근 관측을 얼마나 잘 맞추는지 평가했다.
   - self-consistent한 physics 후보 쪽으로 winner를 아주 작게 이동했다.
   - rank2가 `0.69200` 동률을 만들었지만 돌파 신호는 없었다.

3. Residual density mode
   - 평균 residual 대신 주변 train OOF residual의 density mode를 찾아 hit@1cm 확률질량을 키우는 시도를 했다.
   - OOF 개선폭이 `+0.0001` 정도로 약해 제출 우선순위에서 제외했다.

4. Wide physics action selector
   - winner를 후보 중 하나로 넣고, 넓은 물리/다항/constant-turn 후보군 중 샘플별 best action을 multiclass selector로 학습했다.
   - OOF proxy에서는 `soft_blend08`이 winner 대비 `+0.0016`으로 가장 강했다.
   - 하지만 public은 `0.69180`으로 하락해 OOF-public gap이 큰 축으로 판단한다.

## 핵심 인사이트

- 현재 public best 주변에서는 “winner를 조금 움직이는 post-process”가 대부분 0.6914~0.6920 사이에 갇힌다.
- OOF proxy에서 좋아 보이는 selector류가 public에서 과적합되는 패턴이 반복된다.
- self-consistency, residual density, wide action selector 모두 아이디어는 다르지만 최종적으로는 winner 주변 mm 단위 이동이어서 public 분산을 이기지 못했다.
- 내일은 후보 후처리보다 target 정의, data split 가정, feature representation, 혹은 hit@1cm 전용 학습 objective를 외부 시각으로 다시 검토해야 한다.

## 내일 방향

- 다른 AI에게 현재 최고점, 데이터 구조, 실패 축을 공유하고 “우리가 놓친 완전히 다른 접근”을 먼저 물어본다.
- 특히 아래 질문을 던진다.
  - +80ms target을 직접 예측하는 것 말고 중간 latent state나 physical parameter를 예측하는 방식이 있는가?
  - R-Hit@1cm에 맞는 surrogate loss/objective를 더 직접적으로 만들 수 있는가?
  - train/test trajectory 생성 방식에서 exploit 가능한 구조적 단서가 더 있는가?
  - public plateau가 sample subset 문제라면 subset을 식별하는 다른 unsupervised regime feature가 있는가?

## 관련 산출물

| 파일 | 역할 |
|---|---|
| `scripts/run_local_axis_scale_calibrator_20260528.py` | local 축별 scale 보정 실험 |
| `scripts/run_self_consistency_physics_router_20260528.py` | 내부 과거 self-consistency 기반 물리 라우터 |
| `scripts/run_residual_density_mode_20260528.py` | KNN residual density mode 실험 |
| `scripts/run_wide_physics_action_selector_20260528.py` | 넓은 analytic physics action selector |
| `scripts/make_wide_action_soft_refine_20260528.py` | wide action soft 방향 세기 조절 후보 생성 |
| `docs/external_ai_advice_request_2026-05-29.md` | 내일 다른 AI에게 전달할 조언 요청 문서 |
