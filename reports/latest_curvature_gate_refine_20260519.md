# 2026-05-19 Curvature Gate Refine

- created_at: `2026-05-19T21:29:37`
- public feedback: `gate_t50_a105 = 0.691`, `gate_t38_a105 = 0.690`
- final public feedback: `gate_t52_a105 = 0.6912`, `gate_t50_a105_low025 = 0.6904`
- interpretation: broad curvature application is not enough; the useful region is around threshold 0.50.
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank1_gatet48a105.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank2_gatet52a105.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank3_gatet56a105.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank4_gatet50a100.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank5_gatet50a110.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank6_gatet50a105low025.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank7_gatet50a105low050.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\curvgate_refine_rank8_gatet52a110.csv']`

## Recommended Remaining Public Probes

| rank | submission | name | threshold | high_alpha | low_alpha | route_fraction | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta | vs_temporal_anchor_mean_delta | vs_temporal_anchor_median_delta | vs_temporal_anchor_p95_delta | vs_temporal_anchor_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | curvgate_refine_rank1_gatet48a105.csv | gate_t48_a105 | 0.480000 | 0.105000 | 0.000000 | 0.691800 | 0.000194 | 0.000071 | 0.000755 | 0.009186 | 0.000367 | 0.000203 | 0.001213 | 0.011363 |
| 2 | curvgate_refine_rank2_gatet52a105.csv | gate_t52_a105 | 0.520000 | 0.105000 | 0.000000 | 0.572600 | 0.000235 | 0.000092 | 0.000843 | 0.009186 | 0.000310 | 0.000124 | 0.001115 | 0.011363 |
| 5 | curvgate_refine_rank5_gatet50a110.csv | gate_t50_a110 | 0.500000 | 0.110000 | 0.000000 | 0.636200 | 0.000228 | 0.000100 | 0.000821 | 0.009186 | 0.000358 | 0.000178 | 0.001228 | 0.011904 |
| 6 | curvgate_refine_rank6_gatet50a105low025.csv | gate_t50_a105_low025 | 0.500000 | 0.105000 | 0.025000 | 0.636200 | 0.000167 | 0.000075 | 0.000597 | 0.006635 | 0.000387 | 0.000214 | 0.001181 | 0.011363 |

## All Refine Candidates

| rank | submission | name | threshold | high_alpha | low_alpha | route_fraction | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta | vs_temporal_anchor_mean_delta | vs_temporal_anchor_median_delta | vs_temporal_anchor_p95_delta | vs_temporal_anchor_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | curvgate_refine_rank1_gatet48a105.csv | gate_t48_a105 | 0.480000 | 0.105000 | 0.000000 | 0.691800 | 0.000194 | 0.000071 | 0.000755 | 0.009186 | 0.000367 | 0.000203 | 0.001213 | 0.011363 |
| 2 | curvgate_refine_rank2_gatet52a105.csv | gate_t52_a105 | 0.520000 | 0.105000 | 0.000000 | 0.572600 | 0.000235 | 0.000092 | 0.000843 | 0.009186 | 0.000310 | 0.000124 | 0.001115 | 0.011363 |
| 3 | curvgate_refine_rank3_gatet56a105.csv | gate_t56_a105 | 0.560000 | 0.105000 | 0.000000 | 0.442400 | 0.000281 | 0.000133 | 0.000951 | 0.009186 | 0.000245 | 0.000000 | 0.000996 | 0.011363 |
| 4 | curvgate_refine_rank4_gatet50a100.csv | gate_t50_a100 | 0.500000 | 0.100000 | 0.000000 | 0.636200 | 0.000196 | 0.000056 | 0.000781 | 0.009186 | 0.000326 | 0.000162 | 0.001117 | 0.010822 |
| 5 | curvgate_refine_rank5_gatet50a110.csv | gate_t50_a110 | 0.500000 | 0.110000 | 0.000000 | 0.636200 | 0.000228 | 0.000100 | 0.000821 | 0.009186 | 0.000358 | 0.000178 | 0.001228 | 0.011904 |
| 6 | curvgate_refine_rank6_gatet50a105low025.csv | gate_t50_a105_low025 | 0.500000 | 0.105000 | 0.025000 | 0.636200 | 0.000167 | 0.000075 | 0.000597 | 0.006635 | 0.000387 | 0.000214 | 0.001181 | 0.011363 |
| 7 | curvgate_refine_rank7_gatet50a105low050.csv | gate_t50_a105_low050 | 0.500000 | 0.105000 | 0.050000 | 0.636200 | 0.000121 | 0.000067 | 0.000415 | 0.004083 | 0.000432 | 0.000262 | 0.001260 | 0.011363 |
| 8 | curvgate_refine_rank8_gatet52a110.csv | gate_t52_a110 | 0.520000 | 0.110000 | 0.000000 | 0.572600 | 0.000249 | 0.000113 | 0.000875 | 0.009186 | 0.000325 | 0.000130 | 0.001168 | 0.011904 |
