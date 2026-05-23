# 2026-05-23 Aggressive New Axes

- created_at: `2026-05-23T14:20:19`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- champion: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\newaxis_smooth_rank1_smoothewpolyw11d2r55blend18.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\newaxis_action_rank1_actiondistilltop28.csv']`

## Idea

- Previous temporal curriculum probes all scored 0.69060, so this run intentionally tests two different assumptions.
- Axis A: exp-weighted polynomial smoothing/denoising as a fresh physics bias.
- Axis B: train OOF oracle action distillation, switching some samples to the candidate family predicted to rescue hits.

## Outputs

| axis | submission | selected | oof_r_hit | test_vs_champion_mean_delta | test_vs_champion_p95_delta |
| --- | --- | --- | --- | --- | --- |
| smooth_physics | newaxis_smooth_rank1_smoothewpolyw11d2r55blend18.csv | smooth_ewpoly_w11_d2_r55_blend18 | 0.656200 | 0.002647 | 0.008565 |
| action_distillation | newaxis_action_rank1_actiondistilltop28.csv | actiondistill_top28 | nan | 0.001190 | 0.007178 |

## Smooth Axis OOF

| name | smooth_candidate | blend_weight | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | test_vs_champion_mean_delta | test_vs_champion_median_delta | test_vs_champion_p95_delta | test_vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smooth_ewpoly_w11_d2_r55_blend18 | ewpoly_w11_d2_r55 | 0.180000 | 0.011838 | 0.007180 | 0.024872 | 0.042062 | 0.656200 | 0.002647 | 0.001720 | 0.008565 | 0.027413 |
| smooth_ewpoly_w9_d2_r55_blend18 | ewpoly_w9_d2_r55 | 0.180000 | 0.011822 | 0.007160 | 0.024639 | 0.042245 | 0.655300 | 0.002521 | 0.001589 | 0.008332 | 0.026274 |
| smooth_ewpoly_w7_d2_r55_blend18 | ewpoly_w7_d2_r55 | 0.180000 | 0.011794 | 0.007146 | 0.024496 | 0.041882 | 0.653600 | 0.002301 | 0.001381 | 0.008010 | 0.026706 |
| smooth_ewpoly_w7_d2_r70_blend18 | ewpoly_w7_d2_r70 | 0.180000 | 0.011904 | 0.007198 | 0.024592 | 0.042370 | 0.649600 | 0.002631 | 0.001582 | 0.009080 | 0.030157 |
| smooth_ewpoly_w9_d2_r70_blend18 | ewpoly_w9_d2_r70 | 0.180000 | 0.012022 | 0.007323 | 0.025238 | 0.042512 | 0.646600 | 0.003078 | 0.001940 | 0.010208 | 0.030029 |
| smooth_ewpoly_w7_d2_r55_blend28 | ewpoly_w7_d2_r55 | 0.280000 | 0.012368 | 0.007466 | 0.026050 | 0.043857 | 0.631200 | 0.003580 | 0.002148 | 0.012460 | 0.041543 |
| smooth_ewpoly_w9_d2_r55_blend28 | ewpoly_w9_d2_r55 | 0.280000 | 0.012446 | 0.007626 | 0.026033 | 0.044132 | 0.626700 | 0.003922 | 0.002471 | 0.012961 | 0.040870 |
| smooth_ewpoly_w11_d3_r60_blend18 | ewpoly_w11_d3_r60 | 0.180000 | 0.012373 | 0.007598 | 0.026244 | 0.042811 | 0.625100 | 0.003475 | 0.002121 | 0.011817 | 0.039345 |
| smooth_ewpoly_w7_d2_r70_blend28 | ewpoly_w7_d2_r70 | 0.280000 | 0.012606 | 0.007681 | 0.026537 | 0.044373 | 0.623100 | 0.004093 | 0.002462 | 0.014125 | 0.046911 |
| smooth_ewpoly_w11_d2_r55_blend28 | ewpoly_w11_d2_r55 | 0.280000 | 0.012486 | 0.007653 | 0.025938 | 0.044469 | 0.622400 | 0.004117 | 0.002676 | 0.013323 | 0.042642 |
| smooth_ewpoly_w9_d2_r70_blend28 | ewpoly_w9_d2_r70 | 0.280000 | 0.012871 | 0.007901 | 0.026994 | 0.045351 | 0.607700 | 0.004787 | 0.003018 | 0.015879 | 0.046712 |
| smooth_ewpoly_w7_d2_r55_blend40 | ewpoly_w7_d2_r55 | 0.400000 | 0.013254 | 0.008078 | 0.028511 | 0.046614 | 0.601500 | 0.005114 | 0.003068 | 0.017800 | 0.059347 |
| smooth_ewpoly_w9_d2_r55_blend40 | ewpoly_w9_d2_r55 | 0.400000 | 0.013421 | 0.008229 | 0.028309 | 0.046653 | 0.586800 | 0.005603 | 0.003531 | 0.018515 | 0.058386 |
| smooth_ewpoly_w7_d2_r70_blend40 | ewpoly_w7_d2_r70 | 0.400000 | 0.013678 | 0.008342 | 0.029796 | 0.047211 | 0.584400 | 0.005847 | 0.003516 | 0.020179 | 0.067016 |
| smooth_ewpoly_w11_d3_r60_blend28 | ewpoly_w11_d3_r60 | 0.280000 | 0.013509 | 0.008330 | 0.030340 | 0.045643 | 0.583800 | 0.005405 | 0.003299 | 0.018382 | 0.061203 |
| smooth_ewpoly_w11_d2_r55_blend40 | ewpoly_w11_d2_r55 | 0.400000 | 0.013505 | 0.008399 | 0.028250 | 0.046940 | 0.583400 | 0.005882 | 0.003823 | 0.019033 | 0.060917 |
| smooth_ewpoly_w7_d2_r55_blend55 | ewpoly_w7_d2_r55 | 0.550000 | 0.014571 | 0.008851 | 0.032603 | 0.050132 | 0.560000 | 0.007032 | 0.004218 | 0.024475 | 0.081603 |
| smooth_ewpoly_w9_d2_r70_blend40 | ewpoly_w9_d2_r70 | 0.400000 | 0.014165 | 0.008813 | 0.030258 | 0.048656 | 0.559100 | 0.006839 | 0.004312 | 0.022685 | 0.066731 |
| smooth_ewpoly_w9_d2_r55_blend55 | ewpoly_w9_d2_r55 | 0.550000 | 0.014878 | 0.009133 | 0.032449 | 0.050973 | 0.540600 | 0.007704 | 0.004855 | 0.025459 | 0.080281 |
| smooth_ewpoly_w7_d2_r70_blend55 | ewpoly_w7_d2_r70 | 0.550000 | 0.015253 | 0.009219 | 0.034467 | 0.052539 | 0.536300 | 0.008040 | 0.004835 | 0.027745 | 0.092147 |
| smooth_ewpoly_w11_d3_r60_blend40 | ewpoly_w11_d3_r60 | 0.400000 | 0.015139 | 0.009296 | 0.034985 | 0.051800 | 0.532200 | 0.007721 | 0.004712 | 0.026260 | 0.087434 |
| smooth_ewpoly_w11_d2_r55_blend55 | ewpoly_w11_d2_r55 | 0.550000 | 0.015033 | 0.009455 | 0.032152 | 0.050779 | 0.527100 | 0.008088 | 0.005257 | 0.026171 | 0.083761 |
| smooth_ewpoly_w9_d2_r70_blend55 | ewpoly_w9_d2_r70 | 0.550000 | 0.016061 | 0.010009 | 0.035094 | 0.054110 | 0.499800 | 0.009404 | 0.005929 | 0.031192 | 0.091755 |
| smooth_ewpoly_w11_d3_r60_blend55 | ewpoly_w11_d3_r60 | 0.550000 | 0.017437 | 0.010648 | 0.041801 | 0.061411 | 0.471700 | 0.010617 | 0.006479 | 0.036108 | 0.120221 |

## Action Switch Diagnostics

| name | fraction | actual_switch_fraction | top_candidate_mode | mean_non_champion_conf | min_routed_conf | vs_champion_mean_delta | vs_champion_median_delta | vs_champion_p95_delta | vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actiondistill_top18 | 0.180000 | 0.180000 | poly_w3_d2 | 0.998001 | 0.999541 | 0.000733 | 0.000000 | 0.004888 | 0.106481 |
| actiondistill_top28 | 0.280000 | 0.280000 | poly_w3_d2 | 0.998001 | 0.999440 | 0.001190 | 0.000000 | 0.007178 | 0.118320 |
| actiondistill_top40 | 0.400000 | 0.400000 | poly_w3_d2 | 0.998001 | 0.999302 | 0.001701 | 0.000000 | 0.009210 | 0.118320 |
| actiondistill_top55 | 0.550000 | 0.550000 | poly_w3_d2 | 0.998001 | 0.999034 | 0.002352 | 0.000118 | 0.011130 | 0.118320 |

## Candidate Oracle Diagnostics

| candidate | hit_rate | rescue_rate_vs_champion | harm_rate_vs_champion | mean_distance |
| --- | --- | --- | --- | --- |
| poly_w3_d1 | 0.512400 | 0.034500 | 0.192000 | 0.015763 |
| phys_a400 | 0.591600 | 0.033400 | 0.111700 | 0.013222 |
| phys_a275 | 0.600300 | 0.032800 | 0.102400 | 0.012933 |
| phys_v098_a275 | 0.598400 | 0.032700 | 0.104200 | 0.012859 |
| wdiff_w5_d050 | 0.511600 | 0.032600 | 0.190900 | 0.015396 |
| wdiff_w5_d075 | 0.460800 | 0.032000 | 0.241100 | 0.017326 |
| poly_w4_d2 | 0.456900 | 0.032000 | 0.245000 | 0.019086 |
| poly_w4_d1 | 0.435400 | 0.031600 | 0.266100 | 0.019011 |
| poly_w5_d2 | 0.435500 | 0.030700 | 0.265100 | 0.020139 |
| phys_v102_a275 | 0.593800 | 0.030100 | 0.106200 | 0.013143 |
| poly_w3_d2 | 0.423700 | 0.030100 | 0.276300 | 0.020174 |
| wdiff_w7_d060 | 0.464600 | 0.029800 | 0.235100 | 0.016792 |
| phys_a150 | 0.596200 | 0.028800 | 0.102500 | 0.012818 |
| ewpoly_w7_d2_r55 | 0.432400 | 0.028700 | 0.266200 | 0.019316 |
| poly_w5_d1 | 0.363100 | 0.027600 | 0.334400 | 0.022362 |
| ewpoly_w7_d2_r70 | 0.402600 | 0.027000 | 0.294300 | 0.020843 |
| wdiff_w11_d070 | 0.399800 | 0.026400 | 0.296500 | 0.018742 |
| phys_a000 | 0.578800 | 0.026000 | 0.117100 | 0.012941 |
| ewpoly_w9_d2_r55 | 0.405900 | 0.025800 | 0.289800 | 0.020123 |
| ewpoly_w11_d2_r55 | 0.389600 | 0.025500 | 0.305800 | 0.020536 |
| ewpoly_w11_d3_r60 | 0.320900 | 0.024400 | 0.373400 | 0.025232 |
| ewpoly_w9_d2_r70 | 0.357100 | 0.024200 | 0.337000 | 0.022701 |
| poly_w7_d2 | 0.345500 | 0.023300 | 0.347700 | 0.024217 |
| poly_w7_d1 | 0.245400 | 0.017700 | 0.442200 | 0.029387 |
| poly_w11_d3 | 0.206900 | 0.014800 | 0.477800 | 0.034896 |
| temporal55 | 0.668800 | 0.014100 | 0.015200 | 0.011383 |
| poly_w11_d2 | 0.212300 | 0.013900 | 0.471500 | 0.033061 |
| selector_soft | 0.656700 | 0.010600 | 0.023800 | 0.011626 |
| poly_w11_d1 | 0.127300 | 0.009200 | 0.551800 | 0.042703 |
| fixed_a120 | 0.670100 | 0.003400 | 0.003200 | 0.011362 |
| fixed_a060 | 0.670300 | 0.002900 | 0.002500 | 0.011365 |
| fixed_a090 | 0.669100 | 0.002100 | 0.002900 | 0.011362 |
| gate_t50 | 0.670000 | 0.000600 | 0.000500 | 0.011361 |
| cochamp | 0.670000 | 0.000200 | 0.000100 | 0.011363 |
| gate_t54 | 0.669700 | 0.000200 | 0.000400 | 0.011364 |
| champion | 0.669900 | 0.000000 | 0.000000 | 0.011362 |
