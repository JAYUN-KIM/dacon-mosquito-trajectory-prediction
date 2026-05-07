# Retrieval Blend Router

- Created at: `2026-05-07T09:57:28`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Local configs: `[('axis_best_048_055_062', (0.48, 0.55, 0.62)), ('axis_fine_046_055_066', (0.46, 0.55, 0.66)), ('axis_fine_048_052_070', (0.48, 0.52, 0.7))]`
- Retrieval configs: `[('retr_localmotion_localres_k50_inverse', 'local_motion', 'local_residual', 50, 'inverse'), ('retr_localmotion_localres_k50_softmax075', 'local_motion', 'local_residual', 50, 'softmax0.75'), ('retr_localmotion_localres_k32_softmax075', 'local_motion', 'local_residual', 32, 'softmax0.75')]`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Full local ensemble seeds: `[42, 777, 2026, 3407, 10007]`
- Local feature families: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_blend_rank1_confidentrouteblend_axisfine048052070_retrlocalmotionlocalresk50softmax075_w0.15_r0.20.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_blend_rank2_confidentrouteblend_axisfine048052070_retrlocalmotionlocalresk50inverse_w0.15_r0.20.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_blend_rank3_confidentrouteblend_axisfine048052070_retrlocalmotionlocalresk32softmax075_w0.30_r0.10.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_blend_rank4_confidentrouteblend_axisfine048052070_retrlocalmotionlocalresk32softmax075_w0.30_r0.05.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_blend_rank5_confidentrouteblend_axisfine048052070_retrlocalmotionlocalresk32softmax075_w0.50_r0.05.csv']`

## Top 40 Blend/Router Configs

| strategy | local_config | retrieval_config | weight | route_fraction | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_softmax075 | 0.150000 | 0.200000 | 0.635600 | 0.015793 | 0.617000 | 0.012160 | 0.007566 | 0.631652 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.150000 | 0.200000 | 0.635600 | 0.015793 | 0.617000 | 0.012160 | 0.007566 | 0.631652 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.300000 | 0.100000 | 0.635500 | 0.016183 | 0.616500 | 0.012161 | 0.007565 | 0.631454 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.300000 | 0.050000 | 0.635400 | 0.015522 | 0.617500 | 0.012158 | 0.007558 | 0.631520 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.500000 | 0.050000 | 0.635400 | 0.015614 | 0.617500 | 0.012161 | 0.007564 | 0.631496 |
| linear_blend | axis_best_048_055_062 | retr_localmotion_localres_k32_softmax075 | 0.000000 | 1.000000 | 0.635400 | 0.015225 | 0.617500 | 0.012162 | 0.007561 | 0.631594 |
| linear_blend | axis_best_048_055_062 | retr_localmotion_localres_k50_inverse | 0.000000 | 1.000000 | 0.635400 | 0.015225 | 0.617500 | 0.012162 | 0.007561 | 0.631594 |
| linear_blend | axis_best_048_055_062 | retr_localmotion_localres_k50_softmax075 | 0.000000 | 1.000000 | 0.635400 | 0.015225 | 0.617500 | 0.012162 | 0.007561 | 0.631594 |
| local_only | axis_best_048_055_062 | none | 0.000000 | 0.000000 | 0.635400 | 0.015225 | 0.617500 | 0.012162 | 0.007561 | 0.631594 |
| linear_blend | axis_fine_046_055_066 | retr_localmotion_localres_k32_softmax075 | 0.000000 | 1.000000 | 0.635400 | 0.015888 | 0.617500 | 0.012173 | 0.007560 | 0.631428 |
| linear_blend | axis_fine_046_055_066 | retr_localmotion_localres_k50_inverse | 0.000000 | 1.000000 | 0.635400 | 0.015888 | 0.617500 | 0.012173 | 0.007560 | 0.631428 |
| linear_blend | axis_fine_046_055_066 | retr_localmotion_localres_k50_softmax075 | 0.000000 | 1.000000 | 0.635400 | 0.015888 | 0.617500 | 0.012173 | 0.007560 | 0.631428 |
| local_only | axis_fine_046_055_066 | none | 0.000000 | 0.000000 | 0.635400 | 0.015888 | 0.617500 | 0.012173 | 0.007560 | 0.631428 |
| linear_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_softmax075 | 0.020000 | 1.000000 | 0.635400 | 0.015769 | 0.617000 | 0.012156 | 0.007573 | 0.631458 |
| linear_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.020000 | 1.000000 | 0.635400 | 0.015769 | 0.617000 | 0.012157 | 0.007572 | 0.631458 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_softmax075 | 0.150000 | 0.100000 | 0.635400 | 0.015833 | 0.617000 | 0.012158 | 0.007562 | 0.631442 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.150000 | 0.100000 | 0.635400 | 0.015833 | 0.617000 | 0.012158 | 0.007562 | 0.631442 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.150000 | 0.200000 | 0.635400 | 0.015758 | 0.617000 | 0.012160 | 0.007562 | 0.631461 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.300000 | 0.200000 | 0.635400 | 0.015714 | 0.617000 | 0.012167 | 0.007575 | 0.631472 |
| linear_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.020000 | 1.000000 | 0.635400 | 0.015872 | 0.616500 | 0.012156 | 0.007572 | 0.631432 |
| linear_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.060000 | 1.000000 | 0.635400 | 0.016931 | 0.615000 | 0.012155 | 0.007567 | 0.631167 |
| linear_blend | axis_fine_048_052_070 | retr_localmotion_localres_k32_softmax075 | 0.000000 | 1.000000 | 0.635300 | 0.015357 | 0.617500 | 0.012157 | 0.007552 | 0.631461 |
| linear_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.000000 | 1.000000 | 0.635300 | 0.015357 | 0.617500 | 0.012157 | 0.007552 | 0.631461 |
| linear_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_softmax075 | 0.000000 | 1.000000 | 0.635300 | 0.015357 | 0.617500 | 0.012157 | 0.007552 | 0.631461 |
| local_only | axis_fine_048_052_070 | none | 0.000000 | 0.000000 | 0.635300 | 0.015357 | 0.617500 | 0.012157 | 0.007552 | 0.631461 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_softmax075 | 0.150000 | 0.050000 | 0.635300 | 0.015357 | 0.617500 | 0.012158 | 0.007553 | 0.631461 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.150000 | 0.050000 | 0.635300 | 0.015357 | 0.617500 | 0.012158 | 0.007553 | 0.631461 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_softmax075 | 0.300000 | 0.050000 | 0.635300 | 0.015357 | 0.617500 | 0.012159 | 0.007558 | 0.631461 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.300000 | 0.050000 | 0.635300 | 0.015357 | 0.617500 | 0.012159 | 0.007558 | 0.631461 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.300000 | 0.200000 | 0.635300 | 0.015357 | 0.617500 | 0.012166 | 0.007571 | 0.631461 |
| linear_blend | axis_fine_046_055_066 | retr_localmotion_localres_k50_inverse | 0.020000 | 1.000000 | 0.635300 | 0.015924 | 0.617500 | 0.012172 | 0.007565 | 0.631319 |
| confident_route_blend | axis_best_048_055_062 | retr_localmotion_localres_k50_softmax075 | 0.150000 | 0.150000 | 0.635300 | 0.015304 | 0.617000 | 0.012165 | 0.007576 | 0.631474 |
| confident_route_blend | axis_fine_046_055_066 | retr_localmotion_localres_k32_softmax075 | 0.300000 | 0.050000 | 0.635300 | 0.016254 | 0.617000 | 0.012174 | 0.007560 | 0.631236 |
| confident_route_blend | axis_fine_046_055_066 | retr_localmotion_localres_k50_softmax075 | 0.150000 | 0.200000 | 0.635300 | 0.015959 | 0.617000 | 0.012176 | 0.007567 | 0.631310 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_softmax075 | 0.150000 | 0.150000 | 0.635300 | 0.015904 | 0.616500 | 0.012159 | 0.007563 | 0.631324 |
| confident_route_blend | axis_fine_048_052_070 | retr_localmotion_localres_k50_inverse | 0.150000 | 0.150000 | 0.635300 | 0.015904 | 0.616500 | 0.012159 | 0.007562 | 0.631324 |
| confident_route_blend | axis_fine_046_055_066 | retr_localmotion_localres_k32_softmax075 | 0.300000 | 0.100000 | 0.635300 | 0.016423 | 0.616500 | 0.012177 | 0.007569 | 0.631194 |
| confident_route_blend | axis_fine_046_055_066 | retr_localmotion_localres_k50_inverse | 0.300000 | 0.200000 | 0.635300 | 0.016266 | 0.616500 | 0.012181 | 0.007579 | 0.631234 |
| confident_route_blend | axis_fine_046_055_066 | retr_localmotion_localres_k32_softmax075 | 0.300000 | 0.200000 | 0.635300 | 0.016281 | 0.616500 | 0.012182 | 0.007580 | 0.631230 |
| confident_route_blend | axis_best_048_055_062 | retr_localmotion_localres_k32_softmax075 | 0.150000 | 0.200000 | 0.635300 | 0.015853 | 0.616000 | 0.012166 | 0.007581 | 0.631337 |

## Readout

- This checks whether the new trajectory retrieval axis complements the local-frame residual anchor.
- If the best row is still `local_only`, retrieval is not worth a public submission yet.
- If a small blend or confident route wins CV, submit that candidate before spending more attempts on retrieval-only files.
