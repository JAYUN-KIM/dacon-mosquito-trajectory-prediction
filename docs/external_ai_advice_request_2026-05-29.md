# 외부 AI 조언 요청 메모: DACON 모기 비행 궤적 예측

## 문제 정의

- 과거 11개 3D 좌표를 40ms 간격으로 제공받는다.
- 입력 구간은 `-400ms ~ 0ms`이다.
- 예측 목표는 마지막 관측 시점 기준 `+80ms`의 3D 좌표 `(x, y, z)`이다.
- 평가 지표는 R-Hit@1cm이다.
- 예측 좌표와 실제 좌표의 3D Euclidean distance가 `0.01m` 이하이면 hit로 계산한다.

## 현재 최고 성과

- 최고 Public LB: `0.69200`
- 최고 후보: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv`
- 핵심 아이디어: recursive one-step dynamics를 모든 샘플에 적용하지 않고, gain selector가 고른 top 9% 샘플에만 45% 정도 이동

## 지금까지 효과 있었던 축

| 단계 | Public LB | 핵심 |
|---|---:|---|
| 물리 baseline | 0.61539 | constant velocity/acceleration |
| local-frame residual | 0.65900 | 마지막 속도 기준 local frame residual |
| hit-boundary weighted local-frame | 0.67100 | 1cm 경계 샘플 가중 |
| direct-step local target | 0.67800 | residual 대신 +80ms displacement 직접 예측 |
| selector soft routing | 0.68440 | multiplier 후보를 probability-weighted average |
| temporal-backcast pseudo-supervision | 0.68780 | 궤적 내부 pseudo target 확장 |
| constant-turn curvature correction | 0.69000 | 최근 turn dynamics 보정 |
| curvature gate | 0.69120 | correction이 이득인 샘플만 gate |
| recursive one-step gain gate | 0.69200 | +40ms one-step model을 두 번 적용하되 top 9%만 이동 |

## 최근 막힌 지점

- recursive gate fraction/strength 조정은 `0.69200`에서 plateau.
- winner residual calibrator도 `0.69200` 동률.
- self-consistency physics router도 최고 `0.69200` 동률.
- wide physics action selector는 OOF에서 강했지만 public은 `0.69180`으로 하락.
- local-axis scale, residual density mode, KNN/manifold류는 public에서 안정적으로 개선하지 못했다.

## 2026-05-28 실패/보류 축

1. Local-axis scale calibrator
   - winner displacement를 local forward/side/up 축별 scale로 보정.
   - OOF 신호가 약했다.

2. Self-consistency physics router
   - trajectory 내부 prefix로 물리 후보의 최근 예측 성능을 평가.
   - public 최고 `0.69200` 동률.

3. Residual density mode
   - 유사 train OOF residual의 평균이 아니라 density mode로 이동.
   - OOF 개선폭이 너무 작았다.

4. Wide physics action selector
   - 넓은 analytic physics 후보군 중 샘플별 best action을 분류.
   - OOF는 winner 대비 `+0.0016`이었지만 public은 `0.69180`.

## 조언받고 싶은 질문

1. `+80ms 좌표`를 직접 예측하는 것 말고, 중간 latent state나 physical parameter를 예측하는 더 좋은 formulation이 있을까?
2. R-Hit@1cm에 맞는 differentiable surrogate objective나 sample weighting을 더 직접적으로 설계할 방법이 있을까?
3. public plateau가 특정 regime/subset 문제라면, 어떤 unsupervised feature로 그 subset을 더 잘 식별할 수 있을까?
4. past 11-point trajectory에서 아직 쓰지 않은 구조적 feature가 있을까?
5. train/test가 같은 생성 프로세스라면 trajectory distribution exploit, calibration, conformal prediction 관점에서 시도할 만한 것이 있을까?
6. 현재처럼 winner 주변을 mm 단위로 움직이는 post-process가 막혔다면, 어떤 완전히 다른 modeling branch를 우선 시도해야 할까?

## 제약 및 주의

- test label이나 외부 비공개 정보는 사용할 수 없다.
- 공개 데이터나 논문/일반 물리 지식은 대회 규정에 어긋나지 않는 선에서 참고 가능하다.
- 목표는 mean distance가 아니라 1cm hit rate를 올리는 것이다.
- OOF-public gap이 반복되므로, OOF만 좋아 보이는 복잡한 selector는 위험하다.
