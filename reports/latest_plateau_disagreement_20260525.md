# 2026-05-25 Plateau Disagreement Candidates

- created_at: `2026-05-25T13:41:06`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best: `0.69140`
- anchor: `champmicro_rank3_gatet520a1025.csv`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\plateaudis_rank1_stablemeanall.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\plateaudis_rank2_stablemeanplateau.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\plateaudis_rank3_towarda1005top08b50.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\plateaudis_rank4_towarda1005top15b50.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\plateaudis_rank5_towardt54top06b35.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\plateaudis_rank6_towardt54top12b25.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\plateaudis_rank7_awaya1075soft.csv']`

## Idea

- Alpha probing found a broad 0.69140 plateau.
- Fresh hit-mode retrieval dropped to 0.68980, so large new-coordinate movement is unsafe.
- This experiment only uses disagreement among public-stable plateau/champion submissions and keeps movement extremely small.

## Submission Distance Diagnostics

| name | vs_anchor_mean_delta | vs_anchor_median_delta | vs_anchor_p95_delta | vs_anchor_max_delta |
| --- | --- | --- | --- | --- |
| a1025 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| a1015 | 0.000003 | 0.000001 | 0.000011 | 0.000108 |
| a1010 | 0.000004 | 0.000002 | 0.000016 | 0.000162 |
| a1005 | 0.000006 | 0.000002 | 0.000021 | 0.000216 |
| t52_a105 | 0.000007 | 0.000003 | 0.000027 | 0.000271 |
| a1075 | 0.000015 | 0.000006 | 0.000053 | 0.000541 |
| cochamp_w50 | 0.000022 | 0.000003 | 0.000100 | 0.003907 |
| t54_a105 | 0.000037 | 0.000003 | 0.000166 | 0.008009 |

## Outputs

| rank | submission | name | mode | source | route_fraction | vs_anchor_mean_delta | vs_anchor_median_delta | vs_anchor_p95_delta | vs_anchor_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | plateaudis_rank3_towarda1005top08b50.csv | toward_a1005_top08_b50 | top_fraction | a1005 | 0.080000 | 0.000001 | 0.000000 | 0.000011 | 0.000108 |
| 7 | plateaudis_rank7_awaya1075soft.csv | away_a1075_soft | anti_overcorrect | mixed | 0.100000 | 0.000002 | 0.000000 | 0.000011 | 0.000108 |
| 4 | plateaudis_rank4_towarda1005top15b50.csv | toward_a1005_top15_b50 | top_fraction | a1005 | 0.150000 | 0.000002 | 0.000000 | 0.000011 | 0.000108 |
| 2 | plateaudis_rank2_stablemeanplateau.csv | stable_mean_plateau | mean_plateau | mixed | 1.000000 | 0.000003 | 0.000001 | 0.000012 | 0.000122 |
| 1 | plateaudis_rank1_stablemeanall.csv | stable_mean_all | mean_all | mixed | 1.000000 | 0.000008 | 0.000001 | 0.000035 | 0.001725 |
| 6 | plateaudis_rank6_towardt54top12b25.csv | toward_t54_top12_b25 | top_fraction | t54_a105 | 0.120000 | 0.000008 | 0.000000 | 0.000041 | 0.002002 |
| 5 | plateaudis_rank5_towardt54top06b35.csv | toward_t54_top06_b35 | top_fraction | t54_a105 | 0.060000 | 0.000011 | 0.000000 | 0.000058 | 0.002803 |

## Recommended Public Order

1. `plateaudis_rank2_stablemeanplateau.csv`
2. `plateaudis_rank4_towarda1005top15b50.csv`

## Notes

- If both stay at 0.69140, the plateau is saturated and today should stop.
- If either drops, do not submit the other high-movement variants.
