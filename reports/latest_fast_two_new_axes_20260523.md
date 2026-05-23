# 2026-05-23 Fast Two New Axes

- created_at: `2026-05-23T14:04:32`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- champion: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\fastnew_rank1_smoothewpolyw11d2r55blend18.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\fastnew_rank2_snapsnapv102a035jm0p2d096blend18.csv']`

## Idea

- The temporal curriculum probes scored 0.69060, so this is a fast aggressive pivot.
- Axis 1 uses exponentially weighted polynomial smoothing as a new denoising physics bias.
- Axis 2 uses jerk/snap rebound extrapolation, which is intentionally different from the existing constant-turn correction.

## Outputs

| rank | submission | name | candidate | blend | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | test_vs_champion_mean_delta | test_vs_champion_median_delta | test_vs_champion_p95_delta | test_vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fastnew_rank1_smoothewpolyw11d2r55blend18.csv | smooth_ewpoly_w11_d2_r55_blend18 | ewpoly_w11_d2_r55 | 0.180000 | 0.011832 | 0.007161 | 0.024874 | 0.042136 | 0.656400 | 0.002647 | 0.001720 | 0.008565 | 0.027413 |
| 2 | fastnew_rank2_snapsnapv102a035jm0p2d096blend18.csv | snap_snap_v102_a035_jm0p2_d096_blend18 | snap_v102_a035_jm0p2_d096 | 0.180000 | 0.011379 | 0.006852 | 0.023957 | 0.039650 | 0.670600 | 0.000961 | 0.000697 | 0.002658 | 0.015144 |

## Smooth Axis Leaderboard

| name | candidate | blend | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | test_vs_champion_mean_delta | test_vs_champion_median_delta | test_vs_champion_p95_delta | test_vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smooth_ewpoly_w11_d2_r55_blend18 | ewpoly_w11_d2_r55 | 0.180000 | 0.011832 | 0.007161 | 0.024874 | 0.042136 | 0.656400 | 0.002647 | 0.001720 | 0.008565 | 0.027413 |
| smooth_ewpoly_w9_d2_r55_blend18 | ewpoly_w9_d2_r55 | 0.180000 | 0.011816 | 0.007151 | 0.024737 | 0.042203 | 0.655100 | 0.002521 | 0.001589 | 0.008332 | 0.026274 |
| smooth_ewpoly_w7_d2_r55_blend18 | ewpoly_w7_d2_r55 | 0.180000 | 0.011786 | 0.007115 | 0.024496 | 0.041640 | 0.654000 | 0.002301 | 0.001381 | 0.008010 | 0.026706 |
| smooth_ewpoly_w7_d2_r70_blend18 | ewpoly_w7_d2_r70 | 0.180000 | 0.011898 | 0.007163 | 0.024638 | 0.042152 | 0.650100 | 0.002631 | 0.001582 | 0.009080 | 0.030157 |
| smooth_ewpoly_w9_d2_r70_blend18 | ewpoly_w9_d2_r70 | 0.180000 | 0.012018 | 0.007298 | 0.025236 | 0.042717 | 0.647400 | 0.003078 | 0.001940 | 0.010208 | 0.030029 |
| smooth_ewpoly_w7_d2_r55_blend28 | ewpoly_w7_d2_r55 | 0.280000 | 0.012359 | 0.007454 | 0.025947 | 0.043807 | 0.632500 | 0.003580 | 0.002148 | 0.012460 | 0.041543 |
| smooth_ewpoly_w9_d2_r55_blend28 | ewpoly_w9_d2_r55 | 0.280000 | 0.012440 | 0.007623 | 0.025943 | 0.044132 | 0.628400 | 0.003922 | 0.002471 | 0.012961 | 0.040870 |
| smooth_ewpoly_w11_d3_r60_blend18 | ewpoly_w11_d3_r60 | 0.180000 | 0.012363 | 0.007592 | 0.026202 | 0.042509 | 0.625700 | 0.003475 | 0.002121 | 0.011817 | 0.039345 |
| smooth_ewpoly_w7_d2_r70_blend28 | ewpoly_w7_d2_r70 | 0.280000 | 0.012601 | 0.007669 | 0.026399 | 0.044468 | 0.623500 | 0.004093 | 0.002462 | 0.014125 | 0.046911 |
| smooth_ewpoly_w11_d2_r55_blend28 | ewpoly_w11_d2_r55 | 0.280000 | 0.012480 | 0.007628 | 0.026036 | 0.044513 | 0.623300 | 0.004117 | 0.002676 | 0.013323 | 0.042642 |
| smooth_ewpoly_w9_d2_r70_blend28 | ewpoly_w9_d2_r70 | 0.280000 | 0.012868 | 0.007896 | 0.027021 | 0.045482 | 0.607700 | 0.004787 | 0.003018 | 0.015879 | 0.046712 |
| smooth_ewpoly_w7_d2_r55_blend40 | ewpoly_w7_d2_r55 | 0.400000 | 0.013246 | 0.008044 | 0.028505 | 0.046607 | 0.601000 | 0.005114 | 0.003068 | 0.017800 | 0.059347 |
| smooth_ewpoly_w9_d2_r55_blend40 | ewpoly_w9_d2_r55 | 0.400000 | 0.013416 | 0.008224 | 0.028373 | 0.046655 | 0.586900 | 0.005603 | 0.003531 | 0.018515 | 0.058386 |
| smooth_ewpoly_w11_d3_r60_blend28 | ewpoly_w11_d3_r60 | 0.280000 | 0.013499 | 0.008328 | 0.030265 | 0.045543 | 0.584800 | 0.005405 | 0.003299 | 0.018382 | 0.061203 |
| smooth_ewpoly_w7_d2_r70_blend40 | ewpoly_w7_d2_r70 | 0.400000 | 0.013674 | 0.008332 | 0.029914 | 0.047148 | 0.584500 | 0.005847 | 0.003516 | 0.020179 | 0.067016 |
| smooth_ewpoly_w11_d2_r55_blend40 | ewpoly_w11_d2_r55 | 0.400000 | 0.013500 | 0.008393 | 0.028157 | 0.047032 | 0.582900 | 0.005882 | 0.003823 | 0.019033 | 0.060917 |
| smooth_ewpoly_w7_d2_r55_blend55 | ewpoly_w7_d2_r55 | 0.550000 | 0.014564 | 0.008853 | 0.032594 | 0.050334 | 0.560800 | 0.007032 | 0.004218 | 0.024475 | 0.081603 |
| smooth_ewpoly_w9_d2_r70_blend40 | ewpoly_w9_d2_r70 | 0.400000 | 0.014163 | 0.008799 | 0.030253 | 0.048735 | 0.559400 | 0.006839 | 0.004312 | 0.022685 | 0.066731 |
| smooth_ewpoly_w9_d2_r55_blend55 | ewpoly_w9_d2_r55 | 0.550000 | 0.014875 | 0.009132 | 0.032424 | 0.050970 | 0.540200 | 0.007704 | 0.004855 | 0.025459 | 0.080281 |
| smooth_ewpoly_w7_d2_r70_blend55 | ewpoly_w7_d2_r70 | 0.550000 | 0.015251 | 0.009225 | 0.034473 | 0.052569 | 0.536300 | 0.008040 | 0.004835 | 0.027745 | 0.092147 |

## Snap Axis Leaderboard

| name | candidate | blend | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm | test_vs_champion_mean_delta | test_vs_champion_median_delta | test_vs_champion_p95_delta | test_vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snap_snap_v102_a035_jm0p2_d096_blend18 | snap_v102_a035_jm0p2_d096 | 0.180000 | 0.011379 | 0.006852 | 0.023957 | 0.039650 | 0.670600 | 0.000961 | 0.000697 | 0.002658 | 0.015144 |
| snap_snap_v100_a030_jm0p25_d098_blend18 | snap_v100_a030_jm0p25_d098 | 0.180000 | 0.011380 | 0.006853 | 0.023889 | 0.039490 | 0.670500 | 0.000955 | 0.000686 | 0.002703 | 0.015496 |
| snap_snap_v100_a045_j0p15_d096_blend18 | snap_v100_a045_j0p15_d096 | 0.180000 | 0.011389 | 0.006864 | 0.024086 | 0.038964 | 0.669400 | 0.001147 | 0.000877 | 0.002851 | 0.016484 |
| snap_snap_v098_a020_jm0p35_d098_blend18 | snap_v098_a020_jm0p35_d098 | 0.180000 | 0.011386 | 0.006845 | 0.023831 | 0.039827 | 0.668500 | 0.001043 | 0.000758 | 0.002830 | 0.016032 |
| snap_snap_v104_a015_jm0p45_d100_blend18 | snap_v104_a015_jm0p45_d100 | 0.180000 | 0.011456 | 0.006911 | 0.023845 | 0.040292 | 0.667500 | 0.001190 | 0.000787 | 0.003739 | 0.017402 |
| snap_snap_v102_a035_jm0p2_d096_blend28 | snap_v102_a035_jm0p2_d096 | 0.280000 | 0.011448 | 0.006888 | 0.024085 | 0.040304 | 0.666700 | 0.001495 | 0.001084 | 0.004135 | 0.023557 |
| snap_snap_v100_a030_jm0p25_d098_blend28 | snap_v100_a030_jm0p25_d098 | 0.280000 | 0.011451 | 0.006897 | 0.024081 | 0.040280 | 0.664900 | 0.001486 | 0.001066 | 0.004204 | 0.024105 |
| snap_snap_v096_a055_j0p25_d094_blend18 | snap_v096_a055_j0p25_d094 | 0.180000 | 0.011424 | 0.006953 | 0.024025 | 0.039091 | 0.664800 | 0.001490 | 0.001190 | 0.003308 | 0.019358 |
| snap_snap_v098_a020_jm0p35_d098_blend28 | snap_v098_a020_jm0p35_d098 | 0.280000 | 0.011469 | 0.006932 | 0.023896 | 0.040207 | 0.664700 | 0.001622 | 0.001179 | 0.004402 | 0.024939 |
| snap_snap_v100_a045_j0p15_d096_blend28 | snap_v100_a045_j0p15_d096 | 0.280000 | 0.011483 | 0.006946 | 0.024543 | 0.039759 | 0.663500 | 0.001784 | 0.001365 | 0.004435 | 0.025642 |
| snap_snap_v104_a015_jm0p45_d100_blend28 | snap_v104_a015_jm0p45_d100 | 0.280000 | 0.011602 | 0.006984 | 0.024102 | 0.040939 | 0.661400 | 0.001851 | 0.001224 | 0.005816 | 0.027070 |
| snap_snap_v102_a035_jm0p2_d096_blend40 | snap_v102_a035_jm0p2_d096 | 0.400000 | 0.011585 | 0.006985 | 0.024446 | 0.040616 | 0.660400 | 0.002135 | 0.001548 | 0.005908 | 0.033652 |
| snap_snap_v100_a030_jm0p25_d098_blend40 | snap_v100_a030_jm0p25_d098 | 0.400000 | 0.011589 | 0.007024 | 0.024290 | 0.040680 | 0.658700 | 0.002123 | 0.001523 | 0.006006 | 0.034436 |
| snap_snap_v100_a045_j0p15_d096_blend40 | snap_v100_a045_j0p15_d096 | 0.400000 | 0.011664 | 0.007088 | 0.024740 | 0.040814 | 0.655900 | 0.002549 | 0.001949 | 0.006336 | 0.036631 |
| snap_snap_v098_a020_jm0p35_d098_blend40 | snap_v098_a020_jm0p35_d098 | 0.400000 | 0.011630 | 0.007117 | 0.024025 | 0.040855 | 0.655600 | 0.002317 | 0.001684 | 0.006288 | 0.035627 |
| snap_snap_v096_a055_j0p25_d094_blend28 | snap_v096_a055_j0p25_d094 | 0.280000 | 0.011588 | 0.007164 | 0.024463 | 0.039331 | 0.655400 | 0.002318 | 0.001851 | 0.005145 | 0.030113 |
| snap_snap_v104_a015_jm0p45_d100_blend40 | snap_v104_a015_jm0p45_d100 | 0.400000 | 0.011855 | 0.007198 | 0.024597 | 0.042066 | 0.649900 | 0.002644 | 0.001749 | 0.008309 | 0.038671 |
| snap_snap_v096_a055_j0p25_d094_blend40 | snap_v096_a055_j0p25_d094 | 0.400000 | 0.011890 | 0.007500 | 0.025133 | 0.040692 | 0.637300 | 0.003311 | 0.002644 | 0.007350 | 0.043018 |
