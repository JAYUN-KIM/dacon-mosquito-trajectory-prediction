# 2026-05-25 Champion Alpha Ultrafine

- created_at: `2026-05-25T13:34:25`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best: `0.69140`
- public_feedback: `{'hitmode_rank1_localshape_k32_s0008_clustermean_r30_b18_c00008.csv': 0.6898, 'champmicro_rank1_gatet520a1075.csv': 0.691, 'champmicro_rank3_gatet520a1025.csv': 0.6914, 'champalpha_rank1_t52a1015.csv': 0.6914}`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha2_rank1_t52a1010.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha2_rank2_t52a1012.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha2_rank3_t52a1018.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha2_rank4_t52a1022.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha2_rank5_t52a1005.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha2_rank6_t52a1028.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha2_rank7_t52a0995.csv']`

## Idea

- The fresh local hit-mode retrieval probe dropped to 0.68980, so that axis is cut immediately.
- The only recent public-positive signal is alpha-down around the t52 curvature gate.
- This run keeps threshold fixed at 0.52 and probes a very tight alpha band around 0.1015-0.1025.

## Outputs

| rank | submission | name | threshold | alpha | route_fraction | mean_alpha | vs_current_champion_mean_delta | vs_current_champion_median_delta | vs_current_champion_p95_delta | vs_current_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | champalpha2_rank1_t52a1010.csv | t52_a1010 | 0.520000 | 0.101000 | 0.572600 | 0.057833 | 0.000004 | 0.000002 | 0.000016 | 0.000162 |
| 2 | champalpha2_rank2_t52a1012.csv | t52_a1012 | 0.520000 | 0.101200 | 0.572600 | 0.057947 | 0.000004 | 0.000002 | 0.000014 | 0.000141 |
| 3 | champalpha2_rank3_t52a1018.csv | t52_a1018 | 0.520000 | 0.101800 | 0.572600 | 0.058291 | 0.000002 | 0.000001 | 0.000007 | 0.000076 |
| 4 | champalpha2_rank4_t52a1022.csv | t52_a1022 | 0.520000 | 0.102200 | 0.572600 | 0.058520 | 0.000001 | 0.000000 | 0.000003 | 0.000032 |
| 5 | champalpha2_rank5_t52a1005.csv | t52_a1005 | 0.520000 | 0.100500 | 0.572600 | 0.057546 | 0.000006 | 0.000002 | 0.000021 | 0.000216 |
| 6 | champalpha2_rank6_t52a1028.csv | t52_a1028 | 0.520000 | 0.102800 | 0.572600 | 0.058863 | 0.000001 | 0.000000 | 0.000003 | 0.000032 |
| 7 | champalpha2_rank7_t52a0995.csv | t52_a0995 | 0.520000 | 0.099500 | 0.572600 | 0.056974 | 0.000009 | 0.000004 | 0.000032 | 0.000325 |

## Recommended Public Order

1. `champalpha2_rank1_t52a1010.csv`
2. `champalpha2_rank2_t52a1012.csv`
3. `champalpha2_rank3_t52a1018.csv`

## Notes

- If rank1/rank2 beat 0.69140, continue lowering toward 0.1000.
- If rank3 beats 0.69140, the optimum is likely between 0.1015 and 0.1025.
- If all tie or drop, stop alpha probing and return to a genuinely new modeling axis.
