# Retrieval Route Refine

- Created at: `2026-05-07T10:17:44`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Public anchor: `retr_blend_rank1...w0.15_r0.20.csv = 0.6604`
- Local configs: `[('axis_fine_048_052_070', (0.48, 0.52, 0.7)), ('axis_best_048_055_062', (0.48, 0.55, 0.62))]`
- Retrieval configs: `[('retr_k50_softmax075', 'local_motion', 'local_residual', 50, 'softmax0.75'), ('retr_k50_inverse', 'local_motion', 'local_residual', 50, 'inverse'), ('retr_k32_softmax075', 'local_motion', 'local_residual', 32, 'softmax0.75')]`
- Confidence modes: `['neighbor_mean', 'neighbor_min', 'neighbor_k', 'agreement', 'combo_mean_agreement']`
- Blend weights: `[0.08, 0.1, 0.12, 0.15, 0.18, 0.22, 0.26, 0.3, 0.36]`
- Route fractions: `[0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.32]`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Full local ensemble seeds: `[42, 777, 2026, 3407, 10007]`
- Local feature families: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_route_refine_rank1_axisfine048052070_retrk32softmax075_combomeanagreement_w0.36_r0.25.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_route_refine_rank2_axisfine048052070_retrk32softmax075_combomeanagreement_w0.36_r0.32.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_route_refine_rank3_axisfine048052070_retrk50softmax075_neighbormin_w0.10_r0.32.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_route_refine_rank4_axisfine048052070_retrk50inverse_neighbormin_w0.10_r0.32.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retr_route_refine_rank5_axisfine048052070_retrk32softmax075_combomeanagreement_w0.30_r0.25.csv']`

## Top 50 Route Refine Configs

| strategy | local_config | retrieval_config | confidence_mode | weight | route_fraction | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | combo_mean_agreement | 0.360000 | 0.250000 | 0.636100 | 0.015864 | 0.618500 | 0.012164 | 0.007572 | 0.632134 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | combo_mean_agreement | 0.360000 | 0.320000 | 0.636000 | 0.017037 | 0.617500 | 0.012171 | 0.007590 | 0.631741 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_min | 0.100000 | 0.320000 | 0.635900 | 0.015765 | 0.618000 | 0.012161 | 0.007556 | 0.631959 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.100000 | 0.320000 | 0.635900 | 0.015765 | 0.618000 | 0.012161 | 0.007556 | 0.631959 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | combo_mean_agreement | 0.300000 | 0.250000 | 0.635900 | 0.015833 | 0.618000 | 0.012162 | 0.007574 | 0.631942 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | combo_mean_agreement | 0.300000 | 0.320000 | 0.635900 | 0.016712 | 0.617000 | 0.012167 | 0.007586 | 0.631722 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_mean | 0.360000 | 0.080000 | 0.635800 | 0.016196 | 0.617500 | 0.012161 | 0.007565 | 0.631751 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | combo_mean_agreement | 0.360000 | 0.220000 | 0.635800 | 0.016037 | 0.618500 | 0.012162 | 0.007568 | 0.631791 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | combo_mean_agreement | 0.260000 | 0.250000 | 0.635800 | 0.015583 | 0.618000 | 0.012161 | 0.007573 | 0.631904 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_min | 0.120000 | 0.320000 | 0.635800 | 0.016072 | 0.618000 | 0.012162 | 0.007555 | 0.631782 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.120000 | 0.320000 | 0.635800 | 0.016072 | 0.618000 | 0.012162 | 0.007557 | 0.631782 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | combo_mean_agreement | 0.300000 | 0.250000 | 0.635800 | 0.015583 | 0.618000 | 0.012162 | 0.007560 | 0.631904 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | combo_mean_agreement | 0.300000 | 0.250000 | 0.635800 | 0.015583 | 0.618000 | 0.012162 | 0.007560 | 0.631904 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | combo_mean_agreement | 0.360000 | 0.250000 | 0.635800 | 0.015583 | 0.618000 | 0.012164 | 0.007568 | 0.631904 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_min | 0.360000 | 0.180000 | 0.635800 | 0.015897 | 0.618000 | 0.012171 | 0.007568 | 0.631826 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_min | 0.360000 | 0.200000 | 0.635800 | 0.015679 | 0.618000 | 0.012172 | 0.007577 | 0.631880 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_min | 0.360000 | 0.220000 | 0.635800 | 0.015897 | 0.618000 | 0.012173 | 0.007579 | 0.631826 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_min | 0.100000 | 0.220000 | 0.635800 | 0.015523 | 0.617500 | 0.012159 | 0.007557 | 0.631919 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.100000 | 0.220000 | 0.635800 | 0.015523 | 0.617500 | 0.012159 | 0.007556 | 0.631919 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_min | 0.100000 | 0.250000 | 0.635800 | 0.015667 | 0.617500 | 0.012159 | 0.007556 | 0.631883 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.100000 | 0.250000 | 0.635800 | 0.015667 | 0.617500 | 0.012159 | 0.007555 | 0.631883 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_min | 0.120000 | 0.220000 | 0.635800 | 0.015523 | 0.617500 | 0.012160 | 0.007560 | 0.631919 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.120000 | 0.220000 | 0.635800 | 0.015523 | 0.617500 | 0.012160 | 0.007559 | 0.631919 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_min | 0.120000 | 0.250000 | 0.635800 | 0.015667 | 0.617500 | 0.012160 | 0.007556 | 0.631883 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.120000 | 0.250000 | 0.635800 | 0.015667 | 0.617500 | 0.012160 | 0.007555 | 0.631883 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | combo_mean_agreement | 0.360000 | 0.280000 | 0.635800 | 0.016445 | 0.617500 | 0.012166 | 0.007580 | 0.631689 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_mean | 0.180000 | 0.180000 | 0.635800 | 0.015687 | 0.617000 | 0.012160 | 0.007566 | 0.631878 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_mean | 0.180000 | 0.180000 | 0.635800 | 0.015687 | 0.617000 | 0.012160 | 0.007566 | 0.631878 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.150000 | 0.220000 | 0.635800 | 0.015758 | 0.617000 | 0.012161 | 0.007564 | 0.631860 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.150000 | 0.250000 | 0.635800 | 0.015900 | 0.617000 | 0.012161 | 0.007562 | 0.631825 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | combo_mean_agreement | 0.300000 | 0.200000 | 0.635700 | 0.015418 | 0.618000 | 0.012159 | 0.007555 | 0.631846 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | combo_mean_agreement | 0.300000 | 0.200000 | 0.635700 | 0.015418 | 0.618000 | 0.012159 | 0.007552 | 0.631846 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | combo_mean_agreement | 0.360000 | 0.200000 | 0.635700 | 0.015418 | 0.618000 | 0.012160 | 0.007560 | 0.631846 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | combo_mean_agreement | 0.360000 | 0.250000 | 0.635700 | 0.015635 | 0.618000 | 0.012164 | 0.007569 | 0.631791 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_k | 0.180000 | 0.100000 | 0.635700 | 0.015719 | 0.617500 | 0.012158 | 0.007558 | 0.631770 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_k | 0.180000 | 0.100000 | 0.635700 | 0.015719 | 0.617500 | 0.012158 | 0.007558 | 0.631770 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_mean | 0.100000 | 0.180000 | 0.635700 | 0.015502 | 0.617500 | 0.012158 | 0.007560 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_mean | 0.100000 | 0.180000 | 0.635700 | 0.015502 | 0.617500 | 0.012159 | 0.007559 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_min | 0.080000 | 0.220000 | 0.635700 | 0.015575 | 0.617500 | 0.012159 | 0.007552 | 0.631806 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_mean | 0.100000 | 0.200000 | 0.635700 | 0.015502 | 0.617500 | 0.012159 | 0.007560 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_mean | 0.100000 | 0.200000 | 0.635700 | 0.015502 | 0.617500 | 0.012159 | 0.007559 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_mean | 0.120000 | 0.180000 | 0.635700 | 0.015502 | 0.617500 | 0.012159 | 0.007562 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_mean | 0.120000 | 0.180000 | 0.635700 | 0.015502 | 0.617500 | 0.012159 | 0.007561 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_min | 0.080000 | 0.250000 | 0.635700 | 0.015719 | 0.617500 | 0.012159 | 0.007553 | 0.631770 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_mean | 0.120000 | 0.200000 | 0.635700 | 0.015502 | 0.617500 | 0.012159 | 0.007562 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_mean | 0.120000 | 0.200000 | 0.635700 | 0.015502 | 0.617500 | 0.012159 | 0.007561 | 0.631824 |
| route_blend | axis_fine_048_052_070 | retr_k50_softmax075 | neighbor_min | 0.100000 | 0.280000 | 0.635700 | 0.015727 | 0.617500 | 0.012160 | 0.007556 | 0.631768 |
| route_blend | axis_fine_048_052_070 | retr_k50_inverse | neighbor_min | 0.100000 | 0.280000 | 0.635700 | 0.015727 | 0.617500 | 0.012160 | 0.007556 | 0.631768 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_k | 0.360000 | 0.080000 | 0.635700 | 0.016250 | 0.617500 | 0.012161 | 0.007566 | 0.631637 |
| route_blend | axis_fine_048_052_070 | retr_k32_softmax075 | neighbor_mean | 0.180000 | 0.200000 | 0.635700 | 0.015727 | 0.617500 | 0.012161 | 0.007571 | 0.631768 |

## Readout

- This refines the public-improving selective retrieval route instead of trying retrieval as a standalone model.
- The safest follow-up is a neighbor-distance route near the 0.15 weight / 0.20 fraction anchor.
- Agreement-based confidence is included as an exploratory hedge; use it only if it beats the anchor in CV.
