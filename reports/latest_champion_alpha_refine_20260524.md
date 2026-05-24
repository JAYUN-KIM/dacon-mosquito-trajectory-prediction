# 2026-05-24 Champion Alpha Refine

- created_at: `2026-05-24T15:38:06`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- public_feedback: `{'champmicro_rank1_gatet520a1075.csv': 0.691, 'champmicro_rank3_gatet520a1025.csv': 0.6914}`
- followup_public_feedback: `champalpha_rank1_t52a1015.csv = 0.6914`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha_rank1_t52a1015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha_rank2_t52a1020.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha_rank3_t52a1030.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha_rank4_t52a1000.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha_rank5_t52a1035.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha_rank6_t515a1025.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\champalpha_rank7_t525a1025.csv']`

## Idea

- Public feedback shows alpha-up failed: t52_a1075 scored 0.6910.
- Public feedback also shows alpha-down helped: t52_a1025 scored 0.6914.
- This run keeps the same robust t52 gate family and probes only a tight alpha band around 0.1025.

## Outputs

| rank | submission | name | threshold | alpha | route_fraction | mean_alpha | vs_champion_mean_delta | vs_champion_median_delta | vs_champion_p95_delta | vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | champalpha_rank1_t52a1015.csv | t52_a1015 | 0.520000 | 0.101500 | 0.572600 | 0.058119 | 0.000010 | 0.000004 | 0.000037 | 0.000379 |
| 2 | champalpha_rank2_t52a1020.csv | t52_a1020 | 0.520000 | 0.102000 | 0.572600 | 0.058405 | 0.000009 | 0.000004 | 0.000032 | 0.000325 |
| 3 | champalpha_rank3_t52a1030.csv | t52_a1030 | 0.520000 | 0.103000 | 0.572600 | 0.058978 | 0.000006 | 0.000002 | 0.000021 | 0.000216 |
| 4 | champalpha_rank4_t52a1000.csv | t52_a1000 | 0.520000 | 0.100000 | 0.572600 | 0.057260 | 0.000015 | 0.000006 | 0.000053 | 0.000541 |
| 5 | champalpha_rank5_t52a1035.csv | t52_a1035 | 0.520000 | 0.103500 | 0.572600 | 0.059264 | 0.000004 | 0.000002 | 0.000016 | 0.000162 |
| 6 | champalpha_rank6_t515a1025.csv | t515_a1025 | 0.515000 | 0.102500 | 0.589300 | 0.060403 | 0.000015 | 0.000003 | 0.000034 | 0.004583 |
| 7 | champalpha_rank7_t525a1025.csv | t525_a1025 | 0.525000 | 0.102500 | 0.556200 | 0.057010 | 0.000014 | 0.000003 | 0.000034 | 0.003274 |

## Recommended Public Order

1. `champalpha_rank1_t52a1015.csv`
2. `champalpha_rank2_t52a1020.csv`
3. `champalpha_rank3_t52a1030.csv`

## Notes

- If rank1/rank2 improve, continue lowering alpha toward 0.099-0.101.
- If rank3 improves, the optimum is likely between 0.1025 and 0.105.
- Threshold probes t515/t525 are secondary; use only after alpha neighborhood is mapped.
