# 2026-05-26 Recursive One-Step Gate Refine

- created_at: `2026-05-26T22:15:06`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_before_today: `0.69140`
- public_feedback: `{'recstep_rank1_global_osc89b005late_f100s104u096_w18.csv': 'miss', 'recstep_rank2_global_osc89b005late_f100s100u100_w18.csv': 'miss', 'recstep_rank3_global_osc789b006recent_f104s100u100_w04.csv': 'miss', 'recstep_rank4_gate_osc89b005late_f100s100u100_top080_b40.csv': 0.6918, 'recstep_rank5_global_osc89b005late_f104s100u100_w04.csv': 'miss'}`
- successful_axis: `recursive one-step gain gate, os_c89_b005_late, f100s100u100, top080_b40 = 0.69180`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank1_top08_b45_f100s100u100_top080_b45.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank2_top06_b40_f100s100u100_top060_b40.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank3_top10_b40_f100s100u100_top100_b40.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank4_top08_b35_f100s100u100_top080_b35.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank5_top08_b50_f100s100u100_top080_b50.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank6_top12_b40_f100s100u100_top120_b40.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank7_tilt_top08_b40_f100s104u096_top080_b40.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate_refine_rank8_top05_b45_f100s100u100_top050_b45.csv']`

## Idea

- Global recursive blends failed, but the narrow gain-gated candidate improved to 0.69180.
- Keep the same late one-step dynamics model and refine only the public-positive gate region.
- Primary knobs: selected fraction around top 8% and blend strength around 40%.

## Outputs

| rank | submission | config | mult | frac | weight | oof_hit | oof_delta_hit_vs_champion | oof_vs_champion_mean_delta | oof_vs_champion_median_delta | oof_vs_champion_p95_delta | oof_vs_champion_max_delta | test_vs_champion_mean_delta | test_vs_champion_median_delta | test_vs_champion_p95_delta | test_vs_champion_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | recstepgate_refine_rank1_top08_b45_f100s100u100_top080_b45.csv | top08_b45 | f100s100u100 | 0.080000 | 0.450000 | 0.671100 | 0.001200 | 0.000154 | 0.000000 | 0.001034 | 0.013620 | 0.000144 | 0.000000 | 0.000952 | 0.014929 |
| 2 | recstepgate_refine_rank2_top06_b40_f100s100u100_top060_b40.csv | top06_b40 | f100s100u100 | 0.060000 | 0.400000 | 0.670900 | 0.001000 | 0.000108 | 0.000000 | 0.000602 | 0.012106 | 0.000101 | 0.000000 | 0.000570 | 0.013270 |
| 3 | recstepgate_refine_rank3_top10_b40_f100s100u100_top100_b40.csv | top10_b40 | f100s100u100 | 0.100000 | 0.400000 | 0.671100 | 0.001200 | 0.000163 | 0.000000 | 0.001106 | 0.012106 | 0.000153 | 0.000000 | 0.001014 | 0.013270 |
| 4 | recstepgate_refine_rank4_top08_b35_f100s100u100_top080_b35.csv | top08_b35 | f100s100u100 | 0.080000 | 0.350000 | 0.670600 | 0.000700 | 0.000119 | 0.000000 | 0.000804 | 0.010593 | 0.000112 | 0.000000 | 0.000740 | 0.011611 |
| 5 | recstepgate_refine_rank5_top08_b50_f100s100u100_top080_b50.csv | top08_b50 | f100s100u100 | 0.080000 | 0.500000 | 0.671100 | 0.001200 | 0.000171 | 0.000000 | 0.001149 | 0.015133 | 0.000160 | 0.000000 | 0.001058 | 0.016588 |
| 6 | recstepgate_refine_rank6_top12_b40_f100s100u100_top120_b40.csv | top12_b40 | f100s100u100 | 0.120000 | 0.400000 | 0.671000 | 0.001100 | 0.000191 | 0.000000 | 0.001307 | 0.012106 | 0.000177 | 0.000000 | 0.001188 | 0.013270 |
| 7 | recstepgate_refine_rank7_tilt_top08_b40_f100s104u096_top080_b40.csv | tilt_top08_b40 | f100s104u096 | 0.080000 | 0.400000 | 0.670100 | 0.000200 | 0.000150 | 0.000000 | 0.001004 | 0.019940 | 0.000135 | 0.000000 | 0.000870 | 0.013274 |
| 8 | recstepgate_refine_rank8_top05_b45_f100s100u100_top050_b45.csv | top05_b45 | f100s100u100 | 0.050000 | 0.450000 | 0.670900 | 0.001000 | 0.000105 | 0.000000 | 0.000008 | 0.013620 | 0.000097 | 0.000000 | 0.000009 | 0.014929 |

## Recommended Public Order

1. `recstepgate_refine_rank1_top08_b45_f100s100u100_top080_b45.csv`
2. `recstepgate_refine_rank2_top06_b40_f100s100u100_top060_b40.csv`
3. `recstepgate_refine_rank3_top10_b40_f100s100u100_top100_b40.csv`
4. `recstepgate_refine_rank4_top08_b35_f100s100u100_top080_b35.csv`
5. `recstepgate_refine_rank5_top08_b50_f100s100u100_top080_b50.csv`

## Decision Rule

- If rank1 improves, continue strength search upward around 0.45-0.55.
- If rank2 improves, the selected region should be narrowed below 8%.
- If rank3 improves, the selected region should be widened above 8%.
- If all tie or drop, keep `top080_b40` as the new champion and search a different selector feature set.
