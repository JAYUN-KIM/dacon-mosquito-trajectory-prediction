# 2026-05-16 Temporal Backcast Augmentation

- created_at: `2026-05-16T12:50:11`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `direct_selector_rank2_selectorsoft.csv = 0.68440`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank1_tbc678w020_f1.02_s1.00_u1.00.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank1_anchorblend20_tbc678w020_f1.02_s1.00_u1.00.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank1_anchorblend35_tbc678w020_f1.02_s1.00_u1.00.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank1_anchorblend50_tbc678w020_f1.02_s1.00_u1.00.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank2_tbc678w020_f1.02_s1.04_u0.96.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank2_anchorblend20_tbc678w020_f1.02_s1.04_u0.96.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank2_anchorblend35_tbc678w020_f1.02_s1.04_u0.96.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank2_anchorblend50_tbc678w020_f1.02_s1.04_u0.96.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank3_tbc678w020_f1.02_s1.04_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank3_anchorblend20_tbc678w020_f1.02_s1.04_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank3_anchorblend35_tbc678w020_f1.02_s1.04_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank3_anchorblend50_tbc678w020_f1.02_s1.04_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank4_tbc678w020_f1.02_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank4_anchorblend20_tbc678w020_f1.02_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank4_anchorblend35_tbc678w020_f1.02_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank4_anchorblend50_tbc678w020_f1.02_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank5_tbc678w020_f1.03_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank5_anchorblend20_tbc678w020_f1.03_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank5_anchorblend35_tbc678w020_f1.03_s1.06_u0.94.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\temporalbc_rank5_anchorblend50_tbc678w020_f1.03_s1.06_u0.94.csv']`

## Problem Redefinition

- Reset axis: instead of routing among existing candidates, use the internal time structure of each trajectory to create pseudo +80ms tasks.
- For cutoff c, the model sees points up to c and predicts c+2. Missing early history is linearly backcast so every pseudo sample still has 11 input points.
- Pseudo rows are down-weighted heavily, because the true competition target is still the provided +80ms label.

## CV Leaderboard

| spec | cutoffs | pseudo_weight | pseudo_boundary | forward_mult | side_mult | up_mult | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tb_c678_w020 | 6,7,8 | 0.200000 | True | 1.020000 | 1.000000 | 1.000000 | 0.671500 | 0.025456 | 0.653500 | 0.011343 | 0.006904 | 0.665136 |
| tb_c678_w020 | 6,7,8 | 0.200000 | True | 1.020000 | 1.040000 | 0.960000 | 0.671500 | 0.026163 | 0.653000 | 0.011345 | 0.006903 | 0.664959 |
| tb_c678_w020 | 6,7,8 | 0.200000 | True | 1.020000 | 1.040000 | 0.940000 | 0.671000 | 0.026163 | 0.652500 | 0.011344 | 0.006898 | 0.664459 |
| tb_c678_w020 | 6,7,8 | 0.200000 | True | 1.020000 | 1.060000 | 0.940000 | 0.671000 | 0.026163 | 0.652500 | 0.011346 | 0.006894 | 0.664459 |
| tb_c678_w020 | 6,7,8 | 0.200000 | True | 1.030000 | 1.060000 | 0.940000 | 0.669750 | 0.021567 | 0.654500 | 0.011413 | 0.006946 | 0.664358 |
| tb_c678_w020 | 6,7,8 | 0.200000 | True | 1.010000 | 1.040000 | 0.960000 | 0.669250 | 0.027931 | 0.649500 | 0.011318 | 0.006929 | 0.662267 |
| tb_c78_w030 | 7,8 | 0.300000 | True | 1.010000 | 1.040000 | 0.960000 | 0.669250 | 0.027931 | 0.649500 | 0.011320 | 0.006936 | 0.662267 |
| tb_c678_w020 | 6,7,8 | 0.200000 | True | 1.030000 | 1.080000 | 0.920000 | 0.669000 | 0.022627 | 0.653000 | 0.011415 | 0.006936 | 0.663343 |
| tb_c678_w010 | 6,7,8 | 0.100000 | True | 1.010000 | 1.040000 | 0.960000 | 0.669000 | 0.024042 | 0.652000 | 0.011386 | 0.007038 | 0.662990 |
| tb_c78_w030 | 7,8 | 0.300000 | True | 1.020000 | 1.040000 | 0.940000 | 0.667750 | 0.027931 | 0.648000 | 0.011349 | 0.006935 | 0.660767 |
| tb_c78_w030 | 7,8 | 0.300000 | True | 1.020000 | 1.000000 | 1.000000 | 0.667500 | 0.026870 | 0.648500 | 0.011346 | 0.006932 | 0.660782 |
| tb_c78_w030 | 7,8 | 0.300000 | True | 1.020000 | 1.060000 | 0.940000 | 0.667500 | 0.026870 | 0.648500 | 0.011351 | 0.006918 | 0.660782 |
| tb_c78_w030 | 7,8 | 0.300000 | True | 1.020000 | 1.040000 | 0.960000 | 0.667250 | 0.027224 | 0.648000 | 0.011349 | 0.006932 | 0.660444 |
| tb_c678_w010 | 6,7,8 | 0.100000 | True | 1.020000 | 1.000000 | 1.000000 | 0.666500 | 0.024749 | 0.649000 | 0.011408 | 0.007044 | 0.660313 |
| tb_c678_w010 | 6,7,8 | 0.100000 | True | 1.020000 | 1.060000 | 0.940000 | 0.666500 | 0.024749 | 0.649000 | 0.011414 | 0.007045 | 0.660313 |
| tb_c78_w015 | 7,8 | 0.150000 | True | 1.020000 | 1.040000 | 0.940000 | 0.666500 | 0.028991 | 0.646000 | 0.011410 | 0.007018 | 0.659252 |
| tb_c678_w010 | 6,7,8 | 0.100000 | True | 1.020000 | 1.040000 | 0.960000 | 0.666500 | 0.024042 | 0.649500 | 0.011412 | 0.007044 | 0.660490 |
| tb_c78_w015 | 7,8 | 0.150000 | True | 1.020000 | 1.000000 | 1.000000 | 0.666500 | 0.028284 | 0.646500 | 0.011408 | 0.007013 | 0.659429 |
| tb_c78_w015 | 7,8 | 0.150000 | True | 1.020000 | 1.060000 | 0.940000 | 0.666500 | 0.029698 | 0.645500 | 0.011413 | 0.007020 | 0.659075 |
| tb_c78_w015 | 7,8 | 0.150000 | True | 1.020000 | 1.040000 | 0.960000 | 0.666000 | 0.029698 | 0.645000 | 0.011411 | 0.007016 | 0.658575 |
| tb_c678_w010 | 6,7,8 | 0.100000 | True | 1.020000 | 1.040000 | 0.940000 | 0.666000 | 0.024749 | 0.648500 | 0.011411 | 0.007048 | 0.659813 |
| tb_c78_w015 | 7,8 | 0.150000 | True | 1.010000 | 1.040000 | 0.960000 | 0.665750 | 0.027931 | 0.646000 | 0.011385 | 0.007008 | 0.658767 |
| tb_c678_w010 | 6,7,8 | 0.100000 | True | 1.030000 | 1.060000 | 0.940000 | 0.665000 | 0.024042 | 0.648000 | 0.011478 | 0.007064 | 0.658990 |
| tb_c78_w030 | 7,8 | 0.300000 | True | 1.030000 | 1.080000 | 0.920000 | 0.664500 | 0.024749 | 0.647000 | 0.011423 | 0.006963 | 0.658313 |
| tb_c678_w010 | 6,7,8 | 0.100000 | True | 1.030000 | 1.080000 | 0.920000 | 0.664000 | 0.023335 | 0.647500 | 0.011481 | 0.007071 | 0.658166 |
| tb_c78_w030 | 7,8 | 0.300000 | True | 1.030000 | 1.060000 | 0.940000 | 0.663750 | 0.024395 | 0.646500 | 0.011421 | 0.006967 | 0.657651 |
| tb_c78_w015 | 7,8 | 0.150000 | True | 1.030000 | 1.060000 | 0.940000 | 0.662750 | 0.030759 | 0.641000 | 0.011478 | 0.007033 | 0.655060 |
| tb_c8_w030 | 8 | 0.300000 | True | 1.020000 | 1.060000 | 0.940000 | 0.662500 | 0.029698 | 0.641500 | 0.011448 | 0.006980 | 0.655075 |
| tb_c78_w015 | 7,8 | 0.150000 | True | 1.030000 | 1.080000 | 0.920000 | 0.662500 | 0.031113 | 0.640500 | 0.011481 | 0.007037 | 0.654722 |
| tb_c8_w030 | 8 | 0.300000 | True | 1.020000 | 1.040000 | 0.940000 | 0.662250 | 0.029345 | 0.641500 | 0.011445 | 0.006974 | 0.654914 |
| tb_c8_w030 | 8 | 0.300000 | True | 1.020000 | 1.040000 | 0.960000 | 0.662000 | 0.028991 | 0.641500 | 0.011446 | 0.006978 | 0.654752 |
| tb_c78_flatw020 | 7,8 | 0.200000 | False | 1.020000 | 1.040000 | 0.940000 | 0.661500 | 0.028284 | 0.641500 | 0.011393 | 0.007048 | 0.654429 |
| tb_c78_flatw020 | 7,8 | 0.200000 | False | 1.020000 | 1.000000 | 1.000000 | 0.661000 | 0.028991 | 0.640500 | 0.011390 | 0.007014 | 0.653752 |
| tb_c8_w030 | 8 | 0.300000 | True | 1.010000 | 1.040000 | 0.960000 | 0.661000 | 0.031113 | 0.639000 | 0.011422 | 0.006959 | 0.653222 |
| tb_c78_flatw020 | 7,8 | 0.200000 | False | 1.020000 | 1.040000 | 0.960000 | 0.660750 | 0.028638 | 0.640500 | 0.011394 | 0.007043 | 0.653591 |
| tb_c8_w030 | 8 | 0.300000 | True | 1.020000 | 1.000000 | 1.000000 | 0.660750 | 0.029345 | 0.640000 | 0.011443 | 0.006995 | 0.653414 |
| tb_c78_flatw020 | 7,8 | 0.200000 | False | 1.020000 | 1.060000 | 0.940000 | 0.660500 | 0.028284 | 0.640500 | 0.011397 | 0.007055 | 0.653429 |
| tb_c8_w015 | 8 | 0.150000 | True | 1.010000 | 1.040000 | 0.960000 | 0.659750 | 0.029345 | 0.639000 | 0.011459 | 0.007024 | 0.652414 |
| tb_c78_flatw020 | 7,8 | 0.200000 | False | 1.010000 | 1.040000 | 0.960000 | 0.659500 | 0.027577 | 0.640000 | 0.011377 | 0.007061 | 0.652606 |
| tb_c8_w015 | 8 | 0.150000 | True | 1.020000 | 1.040000 | 0.940000 | 0.659500 | 0.028284 | 0.639500 | 0.011480 | 0.007025 | 0.652429 |
| tb_c8_w015 | 8 | 0.150000 | True | 1.020000 | 1.040000 | 0.960000 | 0.659250 | 0.028638 | 0.639000 | 0.011481 | 0.007017 | 0.652091 |
| tb_c8_w015 | 8 | 0.150000 | True | 1.020000 | 1.000000 | 1.000000 | 0.659000 | 0.028991 | 0.638500 | 0.011478 | 0.007027 | 0.651752 |
| tb_c8_flatw030 | 8 | 0.300000 | False | 1.020000 | 1.040000 | 0.940000 | 0.658500 | 0.028284 | 0.638500 | 0.011470 | 0.007120 | 0.651429 |
| tb_c8_w030 | 8 | 0.300000 | True | 1.030000 | 1.080000 | 0.920000 | 0.658000 | 0.029698 | 0.637000 | 0.011514 | 0.007044 | 0.650575 |
| tb_c8_flatw030 | 8 | 0.300000 | False | 1.020000 | 1.040000 | 0.960000 | 0.658000 | 0.028991 | 0.637500 | 0.011471 | 0.007117 | 0.650752 |
| tb_c8_w015 | 8 | 0.150000 | True | 1.020000 | 1.060000 | 0.940000 | 0.657750 | 0.028638 | 0.637500 | 0.011483 | 0.007024 | 0.650591 |
| tb_c8_flatw030 | 8 | 0.300000 | False | 1.020000 | 1.060000 | 0.940000 | 0.657750 | 0.030052 | 0.636500 | 0.011474 | 0.007115 | 0.650237 |
| tb_c8_w030 | 8 | 0.300000 | True | 1.030000 | 1.060000 | 0.940000 | 0.657750 | 0.030052 | 0.636500 | 0.011511 | 0.007042 | 0.650237 |
| tb_c78_flatw020 | 7,8 | 0.200000 | False | 1.030000 | 1.060000 | 0.940000 | 0.657750 | 0.031466 | 0.635500 | 0.011453 | 0.007073 | 0.649883 |
| tb_c8_flatw030 | 8 | 0.300000 | False | 1.010000 | 1.040000 | 0.960000 | 0.657000 | 0.028991 | 0.636500 | 0.011457 | 0.007109 | 0.649752 |
| tb_c8_flatw030 | 8 | 0.300000 | False | 1.020000 | 1.000000 | 1.000000 | 0.657000 | 0.028991 | 0.636500 | 0.011466 | 0.007130 | 0.649752 |
| tb_c8_flatw030 | 8 | 0.300000 | False | 1.030000 | 1.060000 | 0.940000 | 0.656250 | 0.026517 | 0.637500 | 0.011527 | 0.007136 | 0.649621 |
| tb_c8_w015 | 8 | 0.150000 | True | 1.030000 | 1.060000 | 0.940000 | 0.656250 | 0.030052 | 0.635000 | 0.011544 | 0.007069 | 0.648737 |
| tb_c78_flatw020 | 7,8 | 0.200000 | False | 1.030000 | 1.080000 | 0.920000 | 0.656250 | 0.030759 | 0.634500 | 0.011456 | 0.007067 | 0.648560 |
| tb_c8_flatw030 | 8 | 0.300000 | False | 1.030000 | 1.080000 | 0.920000 | 0.655500 | 0.027577 | 0.636000 | 0.011530 | 0.007126 | 0.648606 |
| tb_c8_w015 | 8 | 0.150000 | True | 1.030000 | 1.080000 | 0.920000 | 0.655250 | 0.027931 | 0.635500 | 0.011547 | 0.007059 | 0.648267 |

## Outputs

| rank | submission | spec | forward_mult | side_mult | up_mult | cv_mean_r_hit | cv_mean_distance | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporalbc_rank1_tbc678w020_f1.02_s1.00_u1.00.csv | tb_c678_w020 | 1.020000 | 1.000000 | 1.000000 | 0.671500 | 0.011343 | 0.001766 | 0.001069 | 0.005789 | 0.056681 |
| 1b20 | temporalbc_rank1_anchorblend20_tbc678w020_f1.02_s1.00_u1.00.csv | tb_c678_w020 | 1.020000 | 1.000000 | 1.000000 | 0.671500 | 0.011343 | 0.000353 | 0.000214 | 0.001158 | 0.011336 |
| 1b35 | temporalbc_rank1_anchorblend35_tbc678w020_f1.02_s1.00_u1.00.csv | tb_c678_w020 | 1.020000 | 1.000000 | 1.000000 | 0.671500 | 0.011343 | 0.000618 | 0.000374 | 0.002026 | 0.019838 |
| 1b50 | temporalbc_rank1_anchorblend50_tbc678w020_f1.02_s1.00_u1.00.csv | tb_c678_w020 | 1.020000 | 1.000000 | 1.000000 | 0.671500 | 0.011343 | 0.000883 | 0.000534 | 0.002894 | 0.028340 |
| 2 | temporalbc_rank2_tbc678w020_f1.02_s1.04_u0.96.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.960000 | 0.671500 | 0.011345 | 0.001761 | 0.001067 | 0.005747 | 0.058575 |
| 2b20 | temporalbc_rank2_anchorblend20_tbc678w020_f1.02_s1.04_u0.96.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.960000 | 0.671500 | 0.011345 | 0.000352 | 0.000213 | 0.001149 | 0.011715 |
| 2b35 | temporalbc_rank2_anchorblend35_tbc678w020_f1.02_s1.04_u0.96.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.960000 | 0.671500 | 0.011345 | 0.000617 | 0.000373 | 0.002011 | 0.020501 |
| 2b50 | temporalbc_rank2_anchorblend50_tbc678w020_f1.02_s1.04_u0.96.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.960000 | 0.671500 | 0.011345 | 0.000881 | 0.000534 | 0.002873 | 0.029287 |
| 3 | temporalbc_rank3_tbc678w020_f1.02_s1.04_u0.94.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.940000 | 0.671000 | 0.011344 | 0.001759 | 0.001067 | 0.005732 | 0.058566 |
| 3b20 | temporalbc_rank3_anchorblend20_tbc678w020_f1.02_s1.04_u0.94.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.940000 | 0.671000 | 0.011344 | 0.000352 | 0.000213 | 0.001146 | 0.011713 |
| 3b35 | temporalbc_rank3_anchorblend35_tbc678w020_f1.02_s1.04_u0.94.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.940000 | 0.671000 | 0.011344 | 0.000616 | 0.000373 | 0.002006 | 0.020498 |
| 3b50 | temporalbc_rank3_anchorblend50_tbc678w020_f1.02_s1.04_u0.94.csv | tb_c678_w020 | 1.020000 | 1.040000 | 0.940000 | 0.671000 | 0.011344 | 0.000880 | 0.000533 | 0.002866 | 0.029283 |
| 4 | temporalbc_rank4_tbc678w020_f1.02_s1.06_u0.94.csv | tb_c678_w020 | 1.020000 | 1.060000 | 0.940000 | 0.671000 | 0.011346 | 0.001762 | 0.001070 | 0.005732 | 0.059523 |
| 4b20 | temporalbc_rank4_anchorblend20_tbc678w020_f1.02_s1.06_u0.94.csv | tb_c678_w020 | 1.020000 | 1.060000 | 0.940000 | 0.671000 | 0.011346 | 0.000352 | 0.000214 | 0.001146 | 0.011905 |
| 4b35 | temporalbc_rank4_anchorblend35_tbc678w020_f1.02_s1.06_u0.94.csv | tb_c678_w020 | 1.020000 | 1.060000 | 0.940000 | 0.671000 | 0.011346 | 0.000617 | 0.000375 | 0.002006 | 0.020833 |
| 4b50 | temporalbc_rank4_anchorblend50_tbc678w020_f1.02_s1.06_u0.94.csv | tb_c678_w020 | 1.020000 | 1.060000 | 0.940000 | 0.671000 | 0.011346 | 0.000881 | 0.000535 | 0.002866 | 0.029761 |
| 5 | temporalbc_rank5_tbc678w020_f1.03_s1.06_u0.94.csv | tb_c678_w020 | 1.030000 | 1.060000 | 0.940000 | 0.669750 | 0.011413 | 0.001949 | 0.001263 | 0.005981 | 0.059536 |
| 5b20 | temporalbc_rank5_anchorblend20_tbc678w020_f1.03_s1.06_u0.94.csv | tb_c678_w020 | 1.030000 | 1.060000 | 0.940000 | 0.669750 | 0.011413 | 0.000390 | 0.000253 | 0.001196 | 0.011907 |
| 5b35 | temporalbc_rank5_anchorblend35_tbc678w020_f1.03_s1.06_u0.94.csv | tb_c678_w020 | 1.030000 | 1.060000 | 0.940000 | 0.669750 | 0.011413 | 0.000682 | 0.000442 | 0.002093 | 0.020838 |
| 5b50 | temporalbc_rank5_anchorblend50_tbc678w020_f1.03_s1.06_u0.94.csv | tb_c678_w020 | 1.030000 | 1.060000 | 0.940000 | 0.669750 | 0.011413 | 0.000974 | 0.000631 | 0.002990 | 0.029768 |

## Notes

- This is intentionally experimental. If public improves, the next step is stronger pseudo curricula and regime-specific pseudo weights.
- If public drops, it suggests within-history dynamics are distribution-shifted from the +80ms target or the backcast padding injects noise.

## Recommended Public Probe

The first two hit-probability router submissions on 2026-05-16 both scored `0.68420`, so this experiment should use the remaining submissions as a stronger new-axis probe.

Recommended order:

1. `temporalbc_rank1_anchorblend35_tbc678w020_f1.02_s1.00_u1.00.csv`
2. `temporalbc_rank1_anchorblend50_tbc678w020_f1.02_s1.00_u1.00.csv`
3. `temporalbc_rank1_tbc678w020_f1.02_s1.00_u1.00.csv`

Reason: this tests the same temporal-backcast direction at 35%, 50%, and 100% strength. If the public score improves with strength, this becomes the next main axis. If it collapses, the pseudo-horizon distribution shift is probably too large.
