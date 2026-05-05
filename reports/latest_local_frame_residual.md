# Local-Frame Residual

- Created at: `2026-05-06T01:23:23`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Feature count: `511`
- Candidate feature families: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Full ensemble seeds: `[42, 777, 2026, 3407, 10007]`
- Target: residual projected into the final-velocity local frame
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_frame_lgbm_a0275_s0.32_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_frame_lgbm_a0275_s0.36_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_frame_lgbm_a0275_s0.40_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_frame_lgbm_a0275_s0.44_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_frame_lgbm_a0275_s0.48_5seed.csv']`

## Shrink CV

| shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- |
| 0.550000 | 0.632600 | 0.016311 | 0.613000 | 0.012135 | 0.007573 |
| 0.480000 | 0.631900 | 0.018700 | 0.609000 | 0.012186 | 0.007571 |
| 0.440000 | 0.630400 | 0.019517 | 0.605500 | 0.012227 | 0.007605 |
| 0.400000 | 0.628300 | 0.019623 | 0.604500 | 0.012276 | 0.007640 |
| 0.360000 | 0.627300 | 0.018877 | 0.603000 | 0.012333 | 0.007672 |
| 0.320000 | 0.623000 | 0.020408 | 0.597500 | 0.012398 | 0.007697 |
| 0.280000 | 0.621700 | 0.021449 | 0.594000 | 0.012471 | 0.007750 |

## Readout

- This is the first geometry-change experiment after the 0.6412/0.6434 residual family.
- If it improves, continue with local-frame features and possibly separate along-track/cross-track model capacity.
- If it underperforms, keep global residual targets and focus on candidate-derived features/bucketing.
