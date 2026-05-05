# Residual Model Zoo

- Created at: `2026-05-06T00:38:55`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- CV seeds: `[42, 777, 2026]`
- Full-train seeds: `[42, 777, 2026]`
- Shrinks: `[0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75]`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\residual_zoo_rank1_lgbm_wide_a0275_s0.25.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\residual_zoo_rank2_lgbm_wide_a0275_s0.45.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\residual_zoo_rank3_lgbm_wide_a0275_s0.35.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\residual_zoo_top2_blend.csv']`

## Top 20

| model | base | velocity_scale | accel_scale | shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_wide | a0275 | 1.000000 | 0.275000 | 0.250000 | 0.612167 | 0.026260 | 0.585500 | 0.012449 | 0.008100 |
| lgbm_wide | a0275 | 1.000000 | 0.275000 | 0.450000 | 0.611667 | 0.019502 | 0.592000 | 0.012267 | 0.008081 |
| lgbm_wide | a0275 | 1.000000 | 0.275000 | 0.350000 | 0.610667 | 0.024271 | 0.587000 | 0.012336 | 0.008032 |
| lgbm_wide | a015 | 1.000000 | 0.150000 | 0.350000 | 0.610167 | 0.022250 | 0.588000 | 0.012253 | 0.008016 |
| lgbm_wide | a015 | 1.000000 | 0.150000 | 0.450000 | 0.609500 | 0.022288 | 0.586500 | 0.012199 | 0.008077 |
| lgbm_base | a0275 | 1.000000 | 0.275000 | 0.250000 | 0.609500 | 0.027915 | 0.580000 | 0.012481 | 0.008117 |
| lgbm_wide | a015 | 1.000000 | 0.150000 | 0.250000 | 0.609333 | 0.024085 | 0.586500 | 0.012348 | 0.008061 |
| lgbm_wide | a0275 | 1.000000 | 0.275000 | 0.500000 | 0.609167 | 0.019264 | 0.589500 | 0.012250 | 0.008084 |
| lgbm_wide | a015 | 1.000000 | 0.150000 | 0.500000 | 0.609167 | 0.022002 | 0.587000 | 0.012188 | 0.008088 |
| lgbm_base | a0275 | 1.000000 | 0.275000 | 0.500000 | 0.609167 | 0.024871 | 0.583000 | 0.012291 | 0.008105 |
| lgbm_base | a0275 | 1.000000 | 0.275000 | 0.450000 | 0.608833 | 0.024659 | 0.582000 | 0.012309 | 0.008059 |
| lgbm_base | a015 | 1.000000 | 0.150000 | 0.450000 | 0.608667 | 0.021888 | 0.585500 | 0.012239 | 0.008089 |
| catboost_d5 | a0275 | 1.000000 | 0.275000 | 0.250000 | 0.608667 | 0.027574 | 0.580000 | 0.012556 | 0.008169 |
| lgbm_base | a015 | 1.000000 | 0.150000 | 0.250000 | 0.608167 | 0.026750 | 0.581500 | 0.012378 | 0.008120 |
| lgbm_wide | a0275 | 1.000000 | 0.275000 | 0.550000 | 0.608000 | 0.021570 | 0.587500 | 0.012243 | 0.008096 |
| lgbm_wide | a015 | 1.000000 | 0.150000 | 0.550000 | 0.607833 | 0.021548 | 0.585500 | 0.012187 | 0.008111 |
| catboost_d5 | a0275 | 1.000000 | 0.275000 | 0.350000 | 0.607833 | 0.029036 | 0.578000 | 0.012459 | 0.008158 |
| lgbm_smooth | a0275 | 1.000000 | 0.275000 | 0.250000 | 0.607667 | 0.027076 | 0.579500 | 0.012489 | 0.008155 |
| lgbm_base | a0275 | 1.000000 | 0.275000 | 0.350000 | 0.607667 | 0.028250 | 0.577500 | 0.012374 | 0.008058 |
| lgbm_smooth | a0275 | 1.000000 | 0.275000 | 0.350000 | 0.607500 | 0.025060 | 0.581500 | 0.012381 | 0.008109 |

## Readout

- Compare these against `aggressive_lgbm_residual.csv`; public already liked residual modeling, so nearby model/shrink variants are worth trying early.
- Prefer candidates that improve mean hit while keeping `min_r_hit` competitive across CV seeds.
- If CatBoost appears near the top, the top-2 blend is especially interesting because model errors should be less correlated.
