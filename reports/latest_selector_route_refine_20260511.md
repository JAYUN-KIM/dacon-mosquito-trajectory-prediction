# 2026-05-11 Selector Route Refine

- created_at: `2026-05-11T22:11:32`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `selector_adjust_rank4_conf45pull015.csv = 0.68360`
- route_grid_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\route_refine_rank1_conf45pull100.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\route_refine_rank2_conf45pull200.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\route_refine_rank3_conf45pull250.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\route_refine_rank4_conf45pull300.csv']`
- continuous_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\contmult_rank1_a20_blend15.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\contmult_rank2_a20_blend25.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\contmult_rank3_a35_blend15.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\contmult_rank4_a35_blend25.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\contmult_rank5_a50_blend15.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\contmult_rank6_a50_blend25.csv']`

## Route Pull Grid

| rank | submission | route_pull_weight | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta | vs_base_mean_delta | vs_base_median_delta | vs_base_p95_delta | vs_base_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | route_refine_rank1_conf45pull100.csv | 0.100000 | 0.000003 | 0.000000 | 0.000027 | 0.000095 | 0.000044 | 0.000000 | 0.000168 | 0.002723 |
| 2 | route_refine_rank2_conf45pull200.csv | 0.200000 | 0.000003 | 0.000000 | 0.000027 | 0.000095 | 0.000051 | 0.000000 | 0.000218 | 0.002723 |
| 3 | route_refine_rank3_conf45pull250.csv | 0.250000 | 0.000007 | 0.000000 | 0.000053 | 0.000190 | 0.000055 | 0.000000 | 0.000269 | 0.002723 |
| 4 | route_refine_rank4_conf45pull300.csv | 0.300000 | 0.000010 | 0.000000 | 0.000080 | 0.000286 | 0.000058 | 0.000000 | 0.000321 | 0.002723 |

## Public Results

| submission | public_score | note |
| --- | --- | --- |
| route_refine_rank2_conf45pull200.csv | 0.68360 | conf0.45 pull 0.20, previous best tie |
| route_refine_rank1_conf45pull100.csv | 0.68360 | conf0.45 pull 0.10, previous best tie |
| route_refine_rank3_conf45pull250.csv | 0.68360 | conf0.45 pull 0.25, previous best tie |
| direct_selector_rank2_selectorsoft.csv | 0.68440 | probability-weighted selector soft routing, new public best |
| direct_selector_rank4_selectorconf045.csv | 0.68420 | full conf0.45 route, improved over previous best |

## Continuous Multiplier CV

| strategy | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- |
| current_mult | 0.656667 | 0.021554 | 0.637500 | 0.011665 | 0.007278 | 0.651278 |
| cont_alpha0.20 | 0.656500 | 0.021604 | 0.637500 | 0.011660 | 0.007275 | 0.651099 |
| cont_alpha0.50 | 0.656333 | 0.021333 | 0.637500 | 0.011652 | 0.007260 | 0.651000 |
| cont_alpha0.35 | 0.656167 | 0.021061 | 0.637500 | 0.011655 | 0.007271 | 0.650901 |

## Continuous Multiplier Outputs

| rank | submission | alpha | blend_with_continuous | mean_mult_f | mean_mult_s | mean_mult_u | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | contmult_rank1_a20_blend15.csv | 0.200000 | 0.150000 | 1.019796 | 1.059447 | 0.938548 | 0.000010 | 0.000003 | 0.000034 | 0.000416 |
| 2 | contmult_rank2_a20_blend25.csv | 0.200000 | 0.250000 | 1.019796 | 1.059447 | 0.938548 | 0.000016 | 0.000006 | 0.000056 | 0.000693 |
| 3 | contmult_rank3_a35_blend15.csv | 0.350000 | 0.150000 | 1.019642 | 1.059033 | 0.937459 | 0.000012 | 0.000005 | 0.000048 | 0.000421 |
| 4 | contmult_rank4_a35_blend25.csv | 0.350000 | 0.250000 | 1.019642 | 1.059033 | 0.937459 | 0.000020 | 0.000009 | 0.000080 | 0.000702 |
| 5 | contmult_rank5_a50_blend15.csv | 0.500000 | 0.150000 | 1.019489 | 1.058618 | 0.936370 | 0.000015 | 0.000007 | 0.000053 | 0.000427 |
| 6 | contmult_rank6_a50_blend25.csv | 0.500000 | 0.250000 | 1.019489 | 1.058618 | 0.936370 | 0.000025 | 0.000012 | 0.000089 | 0.000711 |

## Notes

- Route grid keeps the 2026-05-10 selector_conf0.55 anchor and changes only the pull toward selector_conf0.45.
- Continuous multiplier regression predicts per-sample forward/side/up multipliers from OOF grid-oracle labels, then blends cautiously with the public-best anchor.
- If continuous CV is below current_mult, prioritize route grid submissions and treat continuous outputs as exploratory only.
