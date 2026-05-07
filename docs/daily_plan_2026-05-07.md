# 2026-05-07 daily research plan

## 오늘의 목표

Public LB `0.6594`를 만든 local-frame residual 방향을 더 밀어붙이되, 제출 5회를 모두 같은 계열에 쓰지 않는다.  
오늘은 `0.66` 돌파를 1차 목표로 두고, exploit 후보 2개와 explore 후보 2개, 마지막 적응형 후보 1개를 운영한다.

## 핵심 가설

어제 가장 큰 개선은 전역 `x, y, z` residual이 아니라 마지막 속도 방향 기준의 local-frame residual에서 나왔다.  
즉, 모델이 맞춰야 하는 오차를 `진행 방향 / 좌우 방향 / 상하 방향`으로 분해하면 hit@1cm 근처의 보정이 더 안정된다.

현재 best 주변 CV는 아래 구간에 봉우리가 있다.

```text
forward_shrink = 0.48 근처
side_shrink    = 0.48 ~ 0.55 근처
up_shrink      = 0.62 ~ 0.80 근처
```

## 제출 5회 운영안

1. Fine axis shrink top1
   - 어제 best 주변을 촘촘히 재탐색한 가장 안전한 후보.

2. Fine axis shrink top2
   - top1과 거의 같지만 `up` 또는 `side` 축이 다른 후보로 public/private 흔들림을 확인한다.

3. Smoothed local-frame basis
   - 마지막 1-step 속도만 쓰지 않고 최근 여러 step의 가중 속도로 local basis를 만든다.
   - 마지막 관측 노이즈가 큰 샘플에서 개선 가능성이 있다.

4. Hit-aware local-frame residual
   - 평균 거리보다 1cm hit 경계 근처 샘플을 더 중요하게 학습한다.
   - metric과 직접 맞추는 실험이다.

5. Public feedback 기반 blend 또는 hedge
   - 앞 4개 제출 결과를 보고 가장 좋은 계열을 살짝 blend하거나, 반대로 private 리스크가 낮은 안정 후보를 제출한다.

## 의사결정 규칙

- fine axis 후보가 둘 다 오르면 오늘은 local-frame calibration을 계속 판다.
- fine axis가 정체되고 smoothed basis가 오르면 local basis 정의 자체를 바꾼다.
- hit-aware가 오르면 이후 실험은 모델 구조보다 학습 가중치와 threshold-aware objective 쪽을 강화한다.
- 전부 정체되면 residual ML을 더 키우기보다 후보 선택, cluster별 shrink, trajectory shape bucketing으로 넘어간다.

