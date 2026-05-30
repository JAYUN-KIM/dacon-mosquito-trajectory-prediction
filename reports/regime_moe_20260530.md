# Regime MoE 실험 리포트 (2026-05-30)

## 실험 개요

- **연구 축**: Unsupervised Trajectory Regime Clustering + Per-Regime Ridge MoE
- **목표 지표**: R-Hit@1cm (3D Euclidean distance ≤ 0.01m)
- **OOF 방식**: 5-fold KFold (shuffle=True, seed=42)

## 가설

> 모기 비행에는 물리적으로 구별 가능한 'regime'이 존재한다.
> 가속 직선 비행 / 급격 방향 전환 / 호버링 / 나선형 선회 등.
> Regime을 먼저 unsupervised로 분리하면, 각 레짐 내부의 동역학이
> 단순·일관되므로 레짐별 독립 Ridge 모델이 더 강한 외삽력을 갖는다.

## 기존 실패 축과의 차이

| 기존 축 | 이번 축 |
|---------|----------|
| Gain Gate / Selector: 샘플별 scalar strength 조정 | 공간 분할 후 **레짐별 다른 함수** fitting |
| pseudo-supervision, curvature gate 등: 물리 모델 1개 | 레짐 k개의 독립 모델 (MoE) |
| winner 주변 mm 단위 post-process | winner를 참조하지 않음 (완전 독립 예측) |

## 실험 구성

| # | 설정 | k | alpha |
|---|------|---|-------|
| - | k3_a1 | 3 | 1.0 |
| - | k5_a1 | 5 | 1.0 |
| - | k5_a05 | 5 | 0.5 |
| - | k8_a1 | 8 | 1.0 |

## 결과 요약

| 순위 | tag | OOF hit | mean dist (m) | move from winner (cm) | 파일 |
|------|-----|---------|---------------|----------------------|------|
| 1 | k5_a05 | 0.5544 | 0.01312 | 0.60 | `regime_moe_k5_a05_oof0.5544.csv` |
| 2 | k3_a1 | 0.5528 | 0.01310 | 0.60 | `regime_moe_k3_a1_oof0.5528.csv` |
| 3 | k5_a1 | 0.5499 | 0.01318 | 0.61 | `regime_moe_k5_a1_oof0.5499.csv` |
| 4 | noregime_ridge | 0.5488 | 0.01326 | 0.00 | `N/A (reference only)` |
| 5 | k8_a1 | 0.5437 | 0.01341 | 0.63 | `regime_moe_k8_a1_oof0.5437.csv` |

## 베이스라인 참고

- Constant velocity OOF hit: `0.5787`
- No-regime Ridge OOF hit: `0.5488`

## Public 실패 시 결론

1. **Regime 분포 이동**: train/test 간 비행 패턴 분포가 다름 → covariate shift.
   → 대책: test trajectory로 regime centroid를 online 업데이트하는
     domain adaptation 또는 test-time clustering alignment.

2. **Regime 수 과소**: k가 너무 작아서 실제로 중요한 레짐 내부 분산이 큼.
   → 대책: k를 15~30까지 확대하거나 HDBSCAN 등 density-based 클러스터링 사용.

3. **Ridge 모델 자체의 한계**: 레짐 내부에서도 비선형이 강함.
   → 대책: 레짐별 LightGBM 또는 Gaussian Process 교체.

4. **Feature 부족**: regime feature가 실제 물리 레짐을 포착 못 함.
   → 대책: 주파수 도메인 특징 (FFT of trajectory), 위상 공간 embedding 추가.

## 추천 제출 순서

**1순위**: `regime_moe_k5_a05_oof0.5544.csv` (OOF hit=0.5544)

**2순위**: `regime_moe_k3_a1_oof0.5528.csv` (OOF hit=0.5528)

**3순위**: `regime_moe_k5_a1_oof0.5499.csv` (OOF hit=0.5499)

**4순위**: `regime_moe_k8_a1_oof0.5437.csv` (OOF hit=0.5437)

