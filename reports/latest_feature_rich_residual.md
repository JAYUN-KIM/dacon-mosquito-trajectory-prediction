# Feature-Rich Residual

- Created at: `2026-05-06T01:11:16`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Feature count: `417`
- Candidate feature families: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Full ensemble seeds: `[42, 777, 2026, 3407, 10007]`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\feature_rich_lgbm_a0275_s0.32_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\feature_rich_lgbm_a0275_s0.36_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\feature_rich_lgbm_a0275_s0.40_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\feature_rich_lgbm_a0275_s0.44_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\feature_rich_lgbm_a0275_s0.48_5seed.csv']`

## Shrink CV

| shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- |
| 0.440000 | 0.615200 | 0.017470 | 0.592500 | 0.012457 | 0.007962 |
| 0.400000 | 0.614200 | 0.017024 | 0.594500 | 0.012490 | 0.007944 |
| 0.320000 | 0.613900 | 0.019562 | 0.591500 | 0.012575 | 0.007953 |
| 0.480000 | 0.613900 | 0.018174 | 0.589000 | 0.012432 | 0.007958 |
| 0.280000 | 0.613400 | 0.020632 | 0.588500 | 0.012628 | 0.007994 |
| 0.550000 | 0.613400 | 0.017250 | 0.587500 | 0.012403 | 0.007957 |
| 0.360000 | 0.613200 | 0.019172 | 0.590500 | 0.012529 | 0.007952 |

## Readout

- This tests whether richer physics/poly candidate features improve residual modeling beyond the current 0.6412 public family.
- If CV rises or stays flat while public improves, keep expanding candidate-derived features rather than switching to deep models.
- If CV drops, the current base feature set is already near the sweet spot and the next move should be better validation/bucketing.
