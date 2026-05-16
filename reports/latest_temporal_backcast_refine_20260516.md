# 2026-05-16 Temporal Backcast Refine

- created_at: `2026-05-16T13:52:10`
- public probe result: `35% = 0.6862`, `50% = 0.6878`, `100% = 0.6864`
- interpretation: temporal-backcast direction is valid, but optimal public strength is around 50%.
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r1f102s100u100_w42.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r1f102s100u100_w46.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r1f102s100u100_w48.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r1f102s100u100_w52.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r1f102s100u100_w55.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r1f102s100u100_w57.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r1f102s100u100_w62.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r2f102s104u096_w48.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r2f102s104u096_w52.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r2f102s104u096_w56.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r3f102s104u094_w48.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r3f102s104u094_w52.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r3f102s104u094_w56.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r4f102s106u094_w48.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r4f102s106u094_w52.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_r4f102s106u094_w56.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2_w48.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2_w52.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2_w56.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2r3_w48.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2r3_w52.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2r3_w56.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2r3r4_w48.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2r3r4_w52.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_refine_avgr1r2r3r4_w56.csv']`

## Recommended Next Public Probe

| submission | source | blend_weight | kind | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| temporalbc_refine_r1f102s100u100_w52.csv | r1_f102_s100_u100 | 0.520000 | rank1_strength_grid | 0.000918 | 0.000556 | 0.003010 | 0.029474 |
| temporalbc_refine_r1f102s100u100_w55.csv | r1_f102_s100_u100 | 0.550000 | rank1_strength_grid | 0.000971 | 0.000588 | 0.003184 | 0.031174 |
| temporalbc_refine_avgr1r2_w52.csv | avg_r1r2 | 0.520000 | temporal_direction_ensemble | 0.000917 | 0.000557 | 0.003006 | 0.029966 |
| temporalbc_refine_r2f102s104u096_w52.csv | r2_f102_s104_u096 | 0.520000 | multiplier_variant_grid | 0.000916 | 0.000555 | 0.002988 | 0.030459 |
| temporalbc_refine_avgr1r2r3_w52.csv | avg_r1r2r3 | 0.520000 | temporal_direction_ensemble | 0.000916 | 0.000555 | 0.002995 | 0.030129 |

## All Candidates

| submission | source | blend_weight | kind | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| temporalbc_refine_r2f102s104u096_w48.csv | r2_f102_s104_u096 | 0.480000 | multiplier_variant_grid | 0.000846 | 0.000512 | 0.002758 | 0.028116 |
| temporalbc_refine_r3f102s104u094_w48.csv | r3_f102_s104_u094 | 0.480000 | multiplier_variant_grid | 0.000844 | 0.000512 | 0.002751 | 0.028112 |
| temporalbc_refine_r4f102s106u094_w48.csv | r4_f102_s106_u094 | 0.480000 | multiplier_variant_grid | 0.000846 | 0.000514 | 0.002751 | 0.028571 |
| temporalbc_refine_r2f102s104u096_w52.csv | r2_f102_s104_u096 | 0.520000 | multiplier_variant_grid | 0.000916 | 0.000555 | 0.002988 | 0.030459 |
| temporalbc_refine_r3f102s104u094_w52.csv | r3_f102_s104_u094 | 0.520000 | multiplier_variant_grid | 0.000915 | 0.000555 | 0.002981 | 0.030454 |
| temporalbc_refine_r4f102s106u094_w52.csv | r4_f102_s106_u094 | 0.520000 | multiplier_variant_grid | 0.000916 | 0.000557 | 0.002981 | 0.030952 |
| temporalbc_refine_r2f102s104u096_w56.csv | r2_f102_s104_u096 | 0.560000 | multiplier_variant_grid | 0.000986 | 0.000598 | 0.003218 | 0.032802 |
| temporalbc_refine_r3f102s104u094_w56.csv | r3_f102_s104_u094 | 0.560000 | multiplier_variant_grid | 0.000985 | 0.000597 | 0.003210 | 0.032797 |
| temporalbc_refine_r4f102s106u094_w56.csv | r4_f102_s106_u094 | 0.560000 | multiplier_variant_grid | 0.000987 | 0.000599 | 0.003210 | 0.033333 |
| temporalbc_refine_r1f102s100u100_w42.csv | r1_f102_s100_u100 | 0.420000 | rank1_strength_grid | 0.000742 | 0.000449 | 0.002431 | 0.023806 |
| temporalbc_refine_r1f102s100u100_w46.csv | r1_f102_s100_u100 | 0.460000 | rank1_strength_grid | 0.000812 | 0.000492 | 0.002663 | 0.026073 |
| temporalbc_refine_r1f102s100u100_w48.csv | r1_f102_s100_u100 | 0.480000 | rank1_strength_grid | 0.000848 | 0.000513 | 0.002779 | 0.027207 |
| temporalbc_refine_r1f102s100u100_w52.csv | r1_f102_s100_u100 | 0.520000 | rank1_strength_grid | 0.000918 | 0.000556 | 0.003010 | 0.029474 |
| temporalbc_refine_r1f102s100u100_w55.csv | r1_f102_s100_u100 | 0.550000 | rank1_strength_grid | 0.000971 | 0.000588 | 0.003184 | 0.031174 |
| temporalbc_refine_r1f102s100u100_w57.csv | r1_f102_s100_u100 | 0.580000 | rank1_strength_grid | 0.001024 | 0.000620 | 0.003357 | 0.032875 |
| temporalbc_refine_r1f102s100u100_w62.csv | r1_f102_s100_u100 | 0.620000 | rank1_strength_grid | 0.001095 | 0.000663 | 0.003589 | 0.035142 |
| temporalbc_refine_avgr1r2_w48.csv | avg_r1r2 | 0.480000 | temporal_direction_ensemble | 0.000846 | 0.000514 | 0.002775 | 0.027661 |
| temporalbc_refine_avgr1r2r3_w48.csv | avg_r1r2r3 | 0.480000 | temporal_direction_ensemble | 0.000845 | 0.000512 | 0.002765 | 0.027811 |
| temporalbc_refine_avgr1r2r3r4_w48.csv | avg_r1r2r3r4 | 0.480000 | temporal_direction_ensemble | 0.000845 | 0.000513 | 0.002758 | 0.028001 |
| temporalbc_refine_avgr1r2_w52.csv | avg_r1r2 | 0.520000 | temporal_direction_ensemble | 0.000917 | 0.000557 | 0.003006 | 0.029966 |
| temporalbc_refine_avgr1r2r3_w52.csv | avg_r1r2r3 | 0.520000 | temporal_direction_ensemble | 0.000916 | 0.000555 | 0.002995 | 0.030129 |
| temporalbc_refine_avgr1r2r3r4_w52.csv | avg_r1r2r3r4 | 0.520000 | temporal_direction_ensemble | 0.000916 | 0.000556 | 0.002988 | 0.030335 |
| temporalbc_refine_avgr1r2_w56.csv | avg_r1r2 | 0.560000 | temporal_direction_ensemble | 0.000987 | 0.000599 | 0.003238 | 0.032271 |
| temporalbc_refine_avgr1r2r3_w56.csv | avg_r1r2r3 | 0.560000 | temporal_direction_ensemble | 0.000986 | 0.000598 | 0.003226 | 0.032446 |
| temporalbc_refine_avgr1r2r3r4_w56.csv | avg_r1r2r3r4 | 0.560000 | temporal_direction_ensemble | 0.000986 | 0.000598 | 0.003218 | 0.032668 |

## Notes

- The previous best public score was 0.68440; temporal-backcast 50% blend moved it to 0.68780.
- This script searches around the winning strength and tests whether nearby multiplier variants or temporal-direction ensembles add more gain.
