# 2026-05-31 Clean-Room 연구 계획

## 목적

최근 실험이 기존 winner 주변 보정과 과거 실험 기억에 끌려 `0.691~0.692` 근처에서 반복되고 있다.  
다음 실험은 기존 best submission이나 이전 후보를 anchor로 삼지 않고, raw 데이터에서 다시 문제를 정의한다.

## 사용 가능 데이터

- `data/raw/open (3)/train/*.csv`
- `data/raw/open (3)/test/*.csv`
- `data/raw/open (3)/train_labels.csv`
- `data/raw/open (3)/sample_submission.csv`

## 사용 금지

- 기존 submission 파일을 학습/anchor/blend에 사용하지 않는다.
- `outputs/cache/*.npz`를 사용하지 않는다.
- 기존 champion, recursive gate, curvature gate, selector soft의 top fraction/alpha를 재사용하지 않는다.
- 기존 public feedback을 직접 최적화하는 미세조정부터 시작하지 않는다.

## 권장 시작점

1. Raw trajectory geometry audit
   - 좌표 scale, speed/acceleration/turn distribution을 다시 계산한다.
   - train/test unsupervised distribution shift를 먼저 본다.

2. Target formulation 재설계
   - `y - last` 직접 예측 외에 velocity ratio, acceleration ratio, turn parameter, arc parameter 등 latent target을 검토한다.

3. Metric-aware validation
   - 평균 거리보다 `distance <= 0.01` 경계 주변을 직접 보는 validation table을 만든다.
   - OOF-public gap이 큰 selector류는 뒤로 미룬다.

4. 완전 독립 candidate 생성
   - 기존 best와 섞지 않은 순수 candidate를 먼저 만든다.
   - 순수 candidate가 너무 약하면 그 축은 winner blend로 끌고 가지 말고 폐기한다.

## 내일 첫 질문

- “11개 점만 보고 +80ms를 맞히는 문제를 물리적으로 어떤 latent variable 예측 문제로 바꿀 수 있는가?”
- “train/test의 raw trajectory distribution 차이가 실제로 존재하는가?”
- “1cm hit boundary 근처 샘플은 어떤 motion regime에서 생기는가?”

