# Public Push Residual

- Created at: `2026-05-06T00:53:47`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Full ensemble seeds: `[42, 777, 2026, 3407, 10007]`
- Model: `lgbm_wide`, base `velocity_scale=1.0`, `accel_scale=0.275`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\public_push_lgbm_wide_a0275_s0.20_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\public_push_lgbm_wide_a0275_s0.22_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\public_push_lgbm_wide_a0275_s0.25_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\public_push_lgbm_wide_a0275_s0.28_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\public_push_lgbm_wide_a0275_s0.30_5seed.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\public_push_lgbm_wide_a0275_s0.35_5seed.csv']`

## Shrink CV

| shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- |
| 0.400000 | 0.612400 | 0.016353 | 0.592000 | 0.012555 | 0.008005 |
| 0.350000 | 0.612400 | 0.018345 | 0.589500 | 0.012597 | 0.008006 |
| 0.450000 | 0.611900 | 0.016603 | 0.591500 | 0.012524 | 0.007998 |
| 0.280000 | 0.611000 | 0.018524 | 0.587000 | 0.012673 | 0.008014 |
| 0.300000 | 0.611000 | 0.019710 | 0.585500 | 0.012649 | 0.007990 |
| 0.250000 | 0.610500 | 0.020050 | 0.585500 | 0.012712 | 0.008007 |
| 0.220000 | 0.610000 | 0.020335 | 0.585500 | 0.012754 | 0.008037 |
| 0.200000 | 0.609600 | 0.020945 | 0.584500 | 0.012785 | 0.008045 |
| 0.150000 | 0.607500 | 0.021642 | 0.582000 | 0.012869 | 0.008059 |

## Readout

- This is a public-oriented refinement around the family that improved the leaderboard to 0.6348.
- The 5-seed full ensemble should reduce model variance compared with the previous 3-seed submissions.
- Try nearby shrink variants one at a time; they differ by sub-millimeter to millimeter shifts but that matters for R-Hit@1cm.
