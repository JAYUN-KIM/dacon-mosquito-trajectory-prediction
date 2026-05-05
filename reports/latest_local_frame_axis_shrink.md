# Local-Frame Axis Shrink

- Created at: `2026-05-06T01:35:40`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Feature count: `511`
- Candidate feature families: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Full ensemble seeds: `[42, 777, 2026, 3407, 10007]`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_axis_rank1_f0.48_s0.55_u0.62.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_axis_rank2_f0.48_s0.55_u0.70.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_axis_rank3_f0.48_s0.55_u0.80.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_axis_rank4_f0.48_s0.48_u0.80.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\local_axis_rank5_f0.48_s0.48_u0.62.csv']`

## Top 20 Axis Shrinks

| forward_shrink | side_shrink | up_shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.480000 | 0.550000 | 0.620000 | 0.635400 | 0.015225 | 0.617500 | 0.012162 | 0.007561 |
| 0.480000 | 0.550000 | 0.700000 | 0.635200 | 0.015655 | 0.617000 | 0.012157 | 0.007556 |
| 0.480000 | 0.550000 | 0.800000 | 0.634700 | 0.015340 | 0.617500 | 0.012156 | 0.007536 |
| 0.480000 | 0.480000 | 0.800000 | 0.634700 | 0.015454 | 0.617500 | 0.012158 | 0.007523 |
| 0.480000 | 0.480000 | 0.620000 | 0.634700 | 0.016940 | 0.615000 | 0.012164 | 0.007557 |
| 0.480000 | 0.480000 | 0.700000 | 0.634400 | 0.016475 | 0.614500 | 0.012159 | 0.007560 |
| 0.550000 | 0.550000 | 0.800000 | 0.634400 | 0.015262 | 0.617000 | 0.012120 | 0.007517 |
| 0.550000 | 0.620000 | 0.700000 | 0.634300 | 0.015611 | 0.618000 | 0.012122 | 0.007559 |
| 0.480000 | 0.620000 | 0.800000 | 0.634300 | 0.016158 | 0.617000 | 0.012158 | 0.007561 |
| 0.550000 | 0.620000 | 0.800000 | 0.634200 | 0.016096 | 0.618500 | 0.012121 | 0.007539 |
| 0.480000 | 0.400000 | 0.620000 | 0.634200 | 0.018370 | 0.614500 | 0.012172 | 0.007561 |
| 0.480000 | 0.620000 | 0.700000 | 0.634100 | 0.016161 | 0.616500 | 0.012158 | 0.007574 |
| 0.550000 | 0.700000 | 0.700000 | 0.634000 | 0.016105 | 0.618000 | 0.012128 | 0.007574 |
| 0.550000 | 0.550000 | 0.700000 | 0.634000 | 0.015616 | 0.614000 | 0.012120 | 0.007535 |
| 0.480000 | 0.400000 | 0.700000 | 0.633900 | 0.017064 | 0.615500 | 0.012166 | 0.007565 |
| 0.550000 | 0.480000 | 0.800000 | 0.633800 | 0.014750 | 0.617000 | 0.012122 | 0.007534 |
| 0.620000 | 0.400000 | 0.800000 | 0.633600 | 0.016126 | 0.614500 | 0.012112 | 0.007536 |
| 0.550000 | 0.400000 | 0.700000 | 0.633600 | 0.016345 | 0.614500 | 0.012130 | 0.007558 |
| 0.550000 | 0.620000 | 0.620000 | 0.633500 | 0.016140 | 0.617500 | 0.012128 | 0.007568 |
| 0.400000 | 0.620000 | 0.800000 | 0.633500 | 0.015752 | 0.616500 | 0.012222 | 0.007545 |

## Readout

- Public 0.659 confirmed the local-frame target is a strong direction.
- This experiment calibrates forward, side, and up residual corrections separately instead of using one scalar shrink.
- If one axis consistently wants a larger shrink, the next step is separate model capacity and features by local axis.
