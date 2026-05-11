# 2026-05-12 Selector Soft Temperature

- created_at: `2026-05-12T01:19:13`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `direct_selector_rank2_selectorsoft.csv = 0.68440`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank1_softt075.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank2_softt085.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank3_softt095.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank4_softt110.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank5_softt125.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank6_softt150.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank7_top2t100.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank8_top3t100.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank9_top2t085.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\softtemp_rank10_top3t085.csv']`

## CV

| strategy | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- |
| top3_t1.00 | 0.657500 | 0.021360 | 0.637500 | 0.011639 | 0.007243 | 0.652160 |
| soft_t0.75 | 0.657333 | 0.021197 | 0.638000 | 0.011641 | 0.007253 | 0.652034 |
| soft_t0.85 | 0.657167 | 0.021239 | 0.638000 | 0.011642 | 0.007255 | 0.651857 |
| top3_t0.85 | 0.657167 | 0.021624 | 0.637000 | 0.011638 | 0.007238 | 0.651761 |
| top3_t1.15 | 0.657167 | 0.021127 | 0.637500 | 0.011640 | 0.007240 | 0.651885 |
| soft_t0.95 | 0.656833 | 0.021014 | 0.638000 | 0.011643 | 0.007259 | 0.651580 |
| soft_t1.00 | 0.656833 | 0.021014 | 0.638000 | 0.011644 | 0.007260 | 0.651580 |
| soft_t1.10 | 0.656833 | 0.021014 | 0.638000 | 0.011645 | 0.007261 | 0.651580 |
| soft_t1.25 | 0.656833 | 0.021014 | 0.638000 | 0.011646 | 0.007263 | 0.651580 |
| top2_t0.85 | 0.656833 | 0.021595 | 0.636500 | 0.011636 | 0.007229 | 0.651435 |
| current_mult | 0.656667 | 0.021554 | 0.637500 | 0.011665 | 0.007278 | 0.651278 |
| top2_t1.00 | 0.656667 | 0.021624 | 0.636500 | 0.011637 | 0.007229 | 0.651261 |
| top2_t1.15 | 0.656667 | 0.021624 | 0.636500 | 0.011637 | 0.007234 | 0.651261 |
| soft_t1.50 | 0.656333 | 0.021014 | 0.637500 | 0.011648 | 0.007264 | 0.651080 |

## Outputs

| rank | submission | candidate | note | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | softtemp_rank1_softt075.csv | soft_t0.75 | selector probability temperature 0.75 | 0.000040 | 0.000031 | 0.000102 | 0.000392 |
| 2 | softtemp_rank2_softt085.csv | soft_t0.85 | selector probability temperature 0.85 | 0.000022 | 0.000017 | 0.000058 | 0.000218 |
| 3 | softtemp_rank3_softt095.csv | soft_t0.95 | selector probability temperature 0.95 | 0.000007 | 0.000005 | 0.000018 | 0.000067 |
| 4 | softtemp_rank4_softt110.csv | soft_t1.10 | selector probability temperature 1.10 | 0.000012 | 0.000009 | 0.000034 | 0.000119 |
| 5 | softtemp_rank5_softt125.csv | soft_t1.25 | selector probability temperature 1.25 | 0.000027 | 0.000020 | 0.000079 | 0.000266 |
| 6 | softtemp_rank6_softt150.csv | soft_t1.50 | selector probability temperature 1.50 | 0.000047 | 0.000033 | 0.000143 | 0.000448 |
| 7 | softtemp_rank7_top2t100.csv | top2_t1.00 | top-2 truncated soft selector, T=1.00 | 0.000168 | 0.000125 | 0.000462 | 0.001155 |
| 8 | softtemp_rank8_top3t100.csv | top3_t1.00 | top-3 truncated soft selector, T=1.00 | 0.000098 | 0.000080 | 0.000241 | 0.000634 |
| 9 | softtemp_rank9_top2t085.csv | top2_t0.85 | top-2 truncated soft selector, T=0.85 | 0.000173 | 0.000133 | 0.000463 | 0.001157 |
| 10 | softtemp_rank10_top3t085.csv | top3_t0.85 | top-3 truncated soft selector, T=0.85 | 0.000105 | 0.000089 | 0.000255 | 0.000674 |

## Public Results

| submission | public_score | note |
| --- | --- | --- |
| softtemp_rank8_top3t100.csv | 0.68420 | CV 1위였지만 public에서는 full soft anchor보다 하락 |
| softtemp_rank1_softt075.csv | 0.68420 | sharper temperature도 public에서는 하락 |

## Notes

- The 2026-05-11 public best came from probability-weighted selector soft routing.
- This probe changes only the probability shape via temperature or top-k truncation, keeping the same direct-step candidate pool.
- If CV ranks lower-temperature soft variants near soft_t1.00, submit the smallest movement candidates first.
