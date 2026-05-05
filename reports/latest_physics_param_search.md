# Physics Parameter Search

- Created at: `2026-05-05T23:38:01`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Validation size: `2000`
- Best family: `velocity_accel_grid`
- Best velocity scale: `1.000000`
- Best acceleration scale: `0.150000`
- Best acceleration blend: `0.000000`
- Best R-Hit@1cm: `0.631000`
- Best mean distance: `0.012670` m
- Submission: `C:\open\dacon-mosquito-trajectory-prediction\submissions\physics_param_search_best.csv`

## Top 20

| family | velocity_scale | accel_scale | blend_accel | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| velocity_accel_grid | 1.000000 | 0.150000 | 0.000000 | 0.012670 | 0.007877 | 0.028530 | 0.047260 | 0.631000 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.180000 | 0.012690 | 0.007859 | 0.028958 | 0.047621 | 0.631000 |
| velocity_accel_grid | 1.000000 | 0.200000 | 0.000000 | 0.012709 | 0.007890 | 0.029147 | 0.048215 | 0.631000 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.200000 | 0.012709 | 0.007890 | 0.029147 | 0.048215 | 0.631000 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.140000 | 0.012665 | 0.007873 | 0.028501 | 0.047193 | 0.630500 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.220000 | 0.012732 | 0.007871 | 0.029117 | 0.047715 | 0.630000 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.260000 | 0.012792 | 0.007933 | 0.029020 | 0.047971 | 0.629500 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.160000 | 0.012675 | 0.007861 | 0.028536 | 0.047335 | 0.629000 |
| velocity_accel_grid | 1.000000 | 0.250000 | 0.000000 | 0.012776 | 0.007921 | 0.029018 | 0.047511 | 0.628500 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.240000 | 0.012760 | 0.007913 | 0.029017 | 0.047826 | 0.628000 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.280000 | 0.012829 | 0.007936 | 0.029021 | 0.048235 | 0.627500 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.300000 | 0.012869 | 0.007958 | 0.029145 | 0.048190 | 0.626000 |
| velocity_accel_grid | 1.000000 | 0.300000 | 0.000000 | 0.012869 | 0.007958 | 0.029145 | 0.048190 | 0.626000 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.100000 | 0.012658 | 0.007912 | 0.028735 | 0.047252 | 0.625000 |
| velocity_accel_grid | 1.000000 | 0.100000 | 0.000000 | 0.012658 | 0.007912 | 0.028735 | 0.047252 | 0.625000 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.120000 | 0.012659 | 0.007881 | 0.028917 | 0.047186 | 0.625000 |
| velocity_accel_grid | 0.980000 | 0.200000 | 0.000000 | 0.012645 | 0.007924 | 0.028718 | 0.047449 | 0.624500 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.080000 | 0.012662 | 0.007912 | 0.028699 | 0.048159 | 0.624500 |
| velocity_accel_blend | 1.000000 | 1.000000 | 0.320000 | 0.012914 | 0.007988 | 0.029464 | 0.047753 | 0.624000 |
| velocity_accel_grid | 1.020000 | 0.200000 | 0.000000 | 0.012912 | 0.007965 | 0.029257 | 0.048502 | 0.623000 |

## Readout

- This search optimizes the threshold metric directly, not just average distance.
- If the best velocity scale is below 1.0, the validation set prefers conservative extrapolation to stay inside the 1cm ball more often.
- If acceleration terms do not win, acceleration is likely too noisy at the last two intervals and should be smoothed before reuse.
