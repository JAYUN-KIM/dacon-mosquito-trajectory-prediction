# Aggressive Experiments

- Created at: `2026-05-05T23:48:12`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Validation fraction per seed: `0.2`
- Best physics mean R-Hit@1cm: `0.597700`
- Best physics config: velocity_scale `1.000000`, accel_scale `0.275000`
- Physics submission: `C:\open\dacon-mosquito-trajectory-prediction\submissions\aggressive_physics_cv_best.csv`
- Best LGBM residual mean R-Hit@1cm: `0.607000`
- Best LGBM shrink: `0.500000`
- LGBM submission: `C:\open\dacon-mosquito-trajectory-prediction\submissions\aggressive_lgbm_residual.csv`

## Physics CV Top 20

| family | velocity_scale | accel_scale | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| physics_grid | 1.000000 | 0.275000 | 0.597700 | 0.021162 | 0.573000 | 0.013185 | 0.008210 |
| physics_grid | 1.000000 | 0.300000 | 0.596500 | 0.021325 | 0.570500 | 0.013232 | 0.008236 |
| physics_grid | 1.000000 | 0.250000 | 0.596300 | 0.021052 | 0.572500 | 0.013145 | 0.008225 |
| physics_grid | 1.000000 | 0.225000 | 0.596100 | 0.021498 | 0.571500 | 0.013111 | 0.008222 |
| physics_grid | 1.000000 | 0.200000 | 0.595900 | 0.022010 | 0.571500 | 0.013085 | 0.008231 |
| physics_grid | 1.000000 | 0.175000 | 0.595800 | 0.021449 | 0.572000 | 0.013067 | 0.008223 |
| physics_grid | 1.000000 | 0.150000 | 0.594900 | 0.022990 | 0.569000 | 0.013056 | 0.008237 |
| physics_grid | 1.000000 | 0.325000 | 0.594900 | 0.021755 | 0.567500 | 0.013287 | 0.008251 |
| physics_grid | 0.980000 | 0.200000 | 0.594700 | 0.021859 | 0.567500 | 0.013016 | 0.008304 |
| physics_grid | 0.980000 | 0.225000 | 0.594600 | 0.021758 | 0.566000 | 0.013037 | 0.008303 |
| physics_grid | 0.980000 | 0.175000 | 0.594200 | 0.023132 | 0.566000 | 0.013002 | 0.008328 |
| physics_grid | 1.000000 | 0.350000 | 0.593300 | 0.021052 | 0.567000 | 0.013348 | 0.008276 |
| physics_grid | 0.980000 | 0.275000 | 0.593200 | 0.018397 | 0.569500 | 0.013103 | 0.008278 |
| physics_grid | 0.980000 | 0.300000 | 0.592800 | 0.018992 | 0.568500 | 0.013146 | 0.008284 |
| physics_grid | 0.980000 | 0.250000 | 0.592800 | 0.020632 | 0.567000 | 0.013067 | 0.008283 |
| physics_grid | 1.000000 | 0.125000 | 0.592700 | 0.021510 | 0.568000 | 0.013053 | 0.008244 |
| physics_grid | 0.980000 | 0.150000 | 0.592700 | 0.021600 | 0.567000 | 0.012995 | 0.008321 |
| physics_grid | 1.000000 | 0.100000 | 0.592100 | 0.021226 | 0.568000 | 0.013057 | 0.008296 |
| physics_grid | 1.020000 | 0.225000 | 0.591900 | 0.022317 | 0.565500 | 0.013319 | 0.008325 |
| physics_grid | 0.980000 | 0.325000 | 0.591800 | 0.018969 | 0.566000 | 0.013196 | 0.008282 |

## LightGBM Residual CV

| family | shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- | --- |
| lgbm_residual | 0.500000 | 0.607000 | 0.020156 | 0.577000 | 0.012495 | 0.008056 |
| lgbm_residual | 0.400000 | 0.606900 | 0.019565 | 0.578500 | 0.012531 | 0.008016 |
| lgbm_residual | 0.300000 | 0.606400 | 0.021214 | 0.578500 | 0.012604 | 0.008026 |
| lgbm_residual | 0.200000 | 0.605200 | 0.021014 | 0.579500 | 0.012716 | 0.008093 |
| lgbm_residual | 0.650000 | 0.602500 | 0.019944 | 0.573000 | 0.012509 | 0.008110 |
| lgbm_residual | 0.100000 | 0.602300 | 0.022750 | 0.576500 | 0.012866 | 0.008156 |
| lgbm_residual | 0.000000 | 0.594900 | 0.022990 | 0.569000 | 0.013056 | 0.008237 |
| lgbm_residual | 0.800000 | 0.594800 | 0.020690 | 0.562500 | 0.012596 | 0.008257 |
| lgbm_residual | 1.000000 | 0.580700 | 0.017359 | 0.552500 | 0.012817 | 0.008423 |

## Readout

- Use the physics CV candidate as the safer next submission because it is deterministic and stable across seeds.
- Use the LGBM residual candidate only if CV improves both hit rate and mean distance; residual models can easily optimize distance while losing the 1cm threshold.
- If physics still dominates, the next aggressive path is sample-wise method selection using trajectory noise/curvature buckets.
