# 2026-05-15 Route Gain Model

- created_at: `2026-05-15T21:26:00`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `direct_selector_rank2_selectorsoft.csv = 0.68440`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\routegain_rank1_currentblend25p046.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\routegain_rank2_currentblend50p046.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\routegain_rank3_currentblend25p050.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\routegain_rank4_conf45blend25p046.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\routegain_rank5_conf45blend50p046.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\routegain_rank6_currentreplacep042.csv']`

## Public 제출 결과

- 제출한 route-gain 후보 2개는 모두 Public LB `0.68420`으로 확인됐다.
- OOF에서는 fallback 감지 신호가 있었지만, public에서는 기존 selector-soft best `0.68440`보다 약했다.
- 결론: route gain binary fallback은 당분간 주력 축에서 내리고, 더 큰 새 축을 우선한다.

## OOF Route Diagnostics

| strategy | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | route_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| current_replace_p0.58 | 0.011603 | 0.007084 | 0.024460 | 0.039592 | 0.657900 | 0.413400 |
| current_replace_p0.54 | 0.011603 | 0.007084 | 0.024460 | 0.039592 | 0.657900 | 0.409100 |
| current_replace_p0.50 | 0.011603 | 0.007084 | 0.024460 | 0.039592 | 0.657900 | 0.403600 |
| current_replace_p0.46 | 0.011604 | 0.007084 | 0.024460 | 0.039592 | 0.657900 | 0.394600 |
| current_replace_p0.42 | 0.011605 | 0.007084 | 0.024460 | 0.039592 | 0.657900 | 0.377400 |
| current_blend50_p0.58 | 0.011614 | 0.007092 | 0.024487 | 0.039598 | 0.657000 | 0.413400 |
| current_blend50_p0.54 | 0.011614 | 0.007092 | 0.024487 | 0.039598 | 0.657000 | 0.409100 |
| current_blend50_p0.50 | 0.011615 | 0.007092 | 0.024487 | 0.039598 | 0.657000 | 0.403600 |
| current_blend50_p0.46 | 0.011615 | 0.007092 | 0.024487 | 0.039598 | 0.657000 | 0.394600 |
| current_blend50_p0.42 | 0.011615 | 0.007092 | 0.024487 | 0.039598 | 0.657000 | 0.377400 |
| current_blend25_p0.58 | 0.011621 | 0.007098 | 0.024488 | 0.039672 | 0.656900 | 0.413400 |
| current_blend25_p0.54 | 0.011621 | 0.007098 | 0.024488 | 0.039672 | 0.656900 | 0.409100 |
| current_blend25_p0.50 | 0.011621 | 0.007098 | 0.024488 | 0.039672 | 0.656900 | 0.403600 |
| current_blend25_p0.46 | 0.011621 | 0.007098 | 0.024488 | 0.039672 | 0.656900 | 0.394600 |
| current_blend25_p0.42 | 0.011621 | 0.007098 | 0.024488 | 0.039672 | 0.656900 | 0.377400 |
| conf45_replace_p0.54 | 0.011623 | 0.007098 | 0.024471 | 0.039725 | 0.656800 | 0.409100 |
| conf45_replace_p0.58 | 0.011623 | 0.007098 | 0.024471 | 0.039725 | 0.656800 | 0.413400 |
| conf45_replace_p0.50 | 0.011623 | 0.007098 | 0.024471 | 0.039725 | 0.656800 | 0.403600 |
| conf45_replace_p0.46 | 0.011623 | 0.007098 | 0.024471 | 0.039725 | 0.656800 | 0.394600 |
| conf45_replace_p0.42 | 0.011624 | 0.007098 | 0.024471 | 0.039725 | 0.656800 | 0.377400 |
| conf45_blend50_p0.54 | 0.011625 | 0.007100 | 0.024499 | 0.039729 | 0.656800 | 0.409100 |
| conf45_blend50_p0.58 | 0.011625 | 0.007100 | 0.024499 | 0.039729 | 0.656800 | 0.413400 |
| conf45_blend50_p0.50 | 0.011625 | 0.007100 | 0.024499 | 0.039729 | 0.656800 | 0.403600 |
| conf45_blend50_p0.46 | 0.011625 | 0.007100 | 0.024499 | 0.039729 | 0.656800 | 0.394600 |
| conf45_blend50_p0.42 | 0.011625 | 0.007100 | 0.024499 | 0.039729 | 0.656800 | 0.377400 |
| conf45_blend25_p0.54 | 0.011626 | 0.007101 | 0.024490 | 0.039730 | 0.656800 | 0.409100 |
| conf45_blend25_p0.58 | 0.011626 | 0.007101 | 0.024490 | 0.039730 | 0.656800 | 0.413400 |
| conf45_blend25_p0.50 | 0.011626 | 0.007101 | 0.024490 | 0.039730 | 0.656800 | 0.403600 |
| conf45_blend25_p0.46 | 0.011626 | 0.007101 | 0.024490 | 0.039730 | 0.656800 | 0.394600 |
| conf45_blend25_p0.42 | 0.011626 | 0.007101 | 0.024490 | 0.039730 | 0.656800 | 0.377400 |
| soft_anchor | 0.011627 | 0.007105 | 0.024503 | 0.039730 | 0.656500 | nan |
| conf45 | 0.011633 | 0.007103 | 0.024540 | 0.039624 | 0.656200 | nan |
| current | 0.011648 | 0.007144 | 0.024535 | 0.039701 | 0.655500 | nan |

## Outputs

| rank | submission | strategy | fallback | threshold | mode | route_fraction | mean_soft_win_proba | vs_soft_anchor_mean_delta | vs_soft_anchor_median_delta | vs_soft_anchor_p95_delta | vs_soft_anchor_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | routegain_rank1_currentblend25p046.csv | current_blend25_p046 | current | 0.460000 | blend25 | 0.128100 | 0.601947 | 0.000003 | 0.000000 | 0.000023 | 0.000268 |
| 2 | routegain_rank2_currentblend50p046.csv | current_blend50_p046 | current | 0.460000 | blend50 | 0.128100 | 0.601947 | 0.000007 | 0.000000 | 0.000046 | 0.000536 |
| 3 | routegain_rank3_currentblend25p050.csv | current_blend25_p050 | current | 0.500000 | blend25 | 0.213100 | 0.601947 | 0.000005 | 0.000000 | 0.000034 | 0.000268 |
| 4 | routegain_rank4_conf45blend25p046.csv | conf45_blend25_p046 | conf45 | 0.460000 | blend25 | 0.128100 | 0.601947 | 0.000004 | 0.000000 | 0.000026 | 0.000291 |
| 5 | routegain_rank5_conf45blend50p046.csv | conf45_blend50_p046 | conf45 | 0.460000 | blend50 | 0.128100 | 0.601947 | 0.000008 | 0.000000 | 0.000051 | 0.000583 |
| 6 | routegain_rank6_currentreplacep042.csv | current_replace_p042 | current | 0.420000 | replace | 0.067500 | 0.601947 | 0.000007 | 0.000000 | 0.000036 | 0.001072 |

## Notes

- The model predicts whether selector_soft is closer than the current multiplier route on OOF data.
- Candidates keep selector_soft as the anchor and only fall back for samples predicted as soft-loss risks.
- This tests the next documented axis: route gain binary model and boundary-aware soft routing.
