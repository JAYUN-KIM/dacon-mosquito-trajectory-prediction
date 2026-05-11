# 2026-05-12 Expanded Selector Pool

- created_at: `2026-05-12T01:39:08`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `direct_selector_rank2_selectorsoft.csv = 0.68440`
- candidate_count: `32`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\expanded_selector_rank1_expandedsoftblend015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\expanded_selector_rank2_expandedsoftblend025.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\expanded_selector_rank3_expandedsoftblend035.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\expanded_selector_rank4_expandedsoftfull.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\expanded_selector_rank5_expandedhardblend015.csv']`

## CV

| strategy | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- |
| current_mult | 0.656667 | 0.021554 | 0.637500 | 0.011665 | 0.007278 | 0.651278 |
| current_expsoft_blend0.35 | 0.656667 | 0.022047 | 0.637000 | 0.011650 | 0.007257 | 0.651155 |
| current_expsoft_blend0.25 | 0.656500 | 0.021284 | 0.637500 | 0.011654 | 0.007265 | 0.651179 |
| current_expsoft_blend0.15 | 0.656167 | 0.021239 | 0.637000 | 0.011658 | 0.007266 | 0.650857 |
| expanded_soft | 0.656167 | 0.022418 | 0.635500 | 0.011636 | 0.007253 | 0.650562 |
| expanded_hard | 0.652167 | 0.023186 | 0.632000 | 0.011658 | 0.007235 | 0.646370 |

## Outputs

| rank | submission | candidate | note | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | expanded_selector_rank1_expandedsoftblend015.csv | expanded_soft_blend015 | 0.85 * current public best + 0.15 * expanded candidate soft | 0.000038 | 0.000026 | 0.000123 | 0.000204 |
| 2 | expanded_selector_rank2_expandedsoftblend025.csv | expanded_soft_blend025 | 0.75 * current public best + 0.25 * expanded candidate soft | 0.000063 | 0.000043 | 0.000205 | 0.000340 |
| 3 | expanded_selector_rank3_expandedsoftblend035.csv | expanded_soft_blend035 | 0.65 * current public best + 0.35 * expanded candidate soft | 0.000089 | 0.000060 | 0.000286 | 0.000476 |
| 4 | expanded_selector_rank4_expandedsoftfull.csv | expanded_soft_full | full expanded candidate soft | 0.000253 | 0.000171 | 0.000819 | 0.001359 |
| 5 | expanded_selector_rank5_expandedhardblend015.csv | expanded_hard_blend015 | 0.85 * current public best + 0.15 * expanded hard route | 0.000119 | 0.000109 | 0.000239 | 0.000518 |

## Public Results

| submission | public_score | note |
| --- | --- | --- |
| expanded_selector_rank1_expandedsoftblend015.csv | 0.68400 | expanded candidate pool blend underperformed; candidate pool expansion is deprioritized |

## Candidate Label Distribution

| idx | candidate | forward | side | up | label_count | oof_hit | oof_mean_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | f1.00_s1.04_u0.96 | 1.000000 | 1.040000 | 0.960000 | 2221 | 0.651900 | 0.011648 |
| 10 | f1.00_s1.10_u0.90 | 1.000000 | 1.100000 | 0.900000 | 2146 | 0.651200 | 0.011654 |
| 19 | f1.04_s1.04_u0.96 | 1.040000 | 1.040000 | 0.960000 | 1700 | 0.651800 | 0.011793 |
| 22 | f1.04_s1.10_u0.90 | 1.040000 | 1.100000 | 0.900000 | 1449 | 0.649800 | 0.011799 |
| 0 | diag_f1.02_s1.00_u1.00 | 1.020000 | 1.000000 | 1.000000 | 1077 | 0.655700 | 0.011645 |
| 6 | diag_f1.02_s1.12_u0.88 | 1.020000 | 1.120000 | 0.880000 | 504 | 0.654000 | 0.011656 |
| 14 | f1.01_s1.10_u0.90 | 1.010000 | 1.100000 | 0.900000 | 192 | 0.653400 | 0.011635 |
| 27 | off_f1.02_s1.12_u0.90 | 1.020000 | 1.120000 | 0.900000 | 169 | 0.654200 | 0.011657 |
| 18 | f1.03_s1.10_u0.90 | 1.030000 | 1.100000 | 0.900000 | 124 | 0.655400 | 0.011708 |
| 11 | f1.01_s1.04_u0.96 | 1.010000 | 1.040000 | 0.960000 | 121 | 0.654100 | 0.011629 |
| 30 | off_f1.01_s1.06_u0.96 | 1.010000 | 1.060000 | 0.960000 | 54 | 0.654200 | 0.011632 |
| 15 | f1.03_s1.04_u0.96 | 1.030000 | 1.040000 | 0.960000 | 52 | 0.654800 | 0.011702 |
| 23 | off_f1.02_s1.04_u0.94 | 1.020000 | 1.040000 | 0.940000 | 51 | 0.655900 | 0.011645 |
| 28 | off_f1.03_s1.06_u0.96 | 1.030000 | 1.060000 | 0.960000 | 30 | 0.654300 | 0.011706 |
| 31 | off_f1.01_s1.08_u0.94 | 1.010000 | 1.080000 | 0.940000 | 27 | 0.653700 | 0.011634 |
| 8 | f1.00_s1.06_u0.94 | 1.000000 | 1.060000 | 0.940000 | 18 | 0.651700 | 0.011650 |
| 9 | f1.00_s1.08_u0.92 | 1.000000 | 1.080000 | 0.920000 | 17 | 0.651400 | 0.011652 |
| 21 | f1.04_s1.08_u0.92 | 1.040000 | 1.080000 | 0.920000 | 13 | 0.650400 | 0.011797 |
| 20 | f1.04_s1.06_u0.94 | 1.040000 | 1.060000 | 0.940000 | 11 | 0.651500 | 0.011795 |
| 29 | off_f1.03_s1.08_u0.94 | 1.030000 | 1.080000 | 0.940000 | 10 | 0.655000 | 0.011708 |
| 24 | off_f1.02_s1.06_u0.96 | 1.020000 | 1.060000 | 0.960000 | 7 | 0.654800 | 0.011651 |
| 12 | f1.01_s1.06_u0.94 | 1.010000 | 1.060000 | 0.940000 | 3 | 0.653800 | 0.011630 |
| 16 | f1.03_s1.06_u0.94 | 1.030000 | 1.060000 | 0.940000 | 2 | 0.654900 | 0.011704 |
| 17 | f1.03_s1.08_u0.92 | 1.030000 | 1.080000 | 0.920000 | 1 | 0.655200 | 0.011706 |
| 13 | f1.01_s1.08_u0.92 | 1.010000 | 1.080000 | 0.920000 | 1 | 0.653500 | 0.011632 |
| 2 | diag_f1.02_s1.04_u0.96 | 1.020000 | 1.040000 | 0.960000 | 0 | 0.655500 | 0.011647 |
| 3 | diag_f1.02_s1.06_u0.94 | 1.020000 | 1.060000 | 0.940000 | 0 | 0.655500 | 0.011648 |
| 1 | diag_f1.02_s1.02_u0.98 | 1.020000 | 1.020000 | 0.980000 | 0 | 0.655400 | 0.011646 |
| 4 | diag_f1.02_s1.08_u0.92 | 1.020000 | 1.080000 | 0.920000 | 0 | 0.655200 | 0.011650 |
| 25 | off_f1.02_s1.08_u0.94 | 1.020000 | 1.080000 | 0.940000 | 0 | 0.655000 | 0.011652 |
| 5 | diag_f1.02_s1.10_u0.90 | 1.020000 | 1.100000 | 0.900000 | 0 | 0.654600 | 0.011653 |
| 26 | off_f1.02_s1.10_u0.92 | 1.020000 | 1.100000 | 0.920000 | 0 | 0.654600 | 0.011655 |

## Notes

- Temperature and top-k truncation hurt public, so this probe changes the multiplier candidate pool itself.
- Output candidates blend the expanded-pool soft route back into the current public best to control public risk.
