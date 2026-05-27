# 2026-05-27 Recursive One-Step Gate Jitter

- created_at: `2026-05-27T22:36:34`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_winner: `recstep_rank4_gate_osc89b005late_f100s100u100_top080_b40.csv = 0.69180`
- public_feedback_after_submit: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`, `recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv = 0.69160`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank3_top080_b450_f100s100u100_top080_b450.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank4_top080_b475_f100s100u100_top080_b475.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank5_top095_b425_f100s100u100_top095_b425.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank6_top080_b425_f100s100u100_top080_b425.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank7_top075_b425_f100s100u100_top075_b425.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27_rank8_top085_b425_f100s100u100_top085_b425.csv']`

## Idea

- Keep yesterday's public-positive recursive one-step gain gate.
- Do not retry global recursive blends.
- Jitter only around `top080_b40`: selected fraction 6.5%-9.5% and strength 37.5%-47.5%.
- Rank candidates by OOF hit plus proximity to the known public winner.

## Candidate Leaderboard

| config | frac | weight | oof_hit | oof_delta_vs_base_champion | oof_delta_vs_public_winner_proxy | oof_vs_winner_mean_delta | oof_vs_winner_median_delta | oof_vs_winner_p95_delta | oof_vs_winner_max_delta | vs_public_winner_mean_delta | vs_public_winner_median_delta | vs_public_winner_p95_delta | vs_public_winner_max_delta | changed_vs_public_winner | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top090_b450 | 0.090000 | 0.450000 | 0.671200 | 0.001300 | 0.000200 | 0.000031 | 0.000000 | 0.000150 | 0.011920 | 0.000030 | 0.000000 | 0.000134 | 0.010416 | 900 | 0.000915 |
| top090_b400 | 0.090000 | 0.400000 | 0.671100 | 0.001200 | 0.000100 | 0.000013 | 0.000000 | 0.000000 | 0.010595 | 0.000013 | 0.000000 | 0.000000 | 0.009259 | 100 | 0.000815 |
| top080_b450 | 0.080000 | 0.450000 | 0.671100 | 0.001200 | 0.000100 | 0.000017 | 0.000000 | 0.000115 | 0.001513 | 0.000016 | 0.000000 | 0.000106 | 0.001659 | 800 | 0.000815 |
| top080_b475 | 0.080000 | 0.475000 | 0.671100 | 0.001200 | 0.000100 | 0.000026 | 0.000000 | 0.000172 | 0.002270 | 0.000024 | 0.000000 | 0.000159 | 0.002488 | 800 | 0.000815 |
| top095_b425 | 0.095000 | 0.425000 | 0.671100 | 0.001200 | 0.000100 | 0.000028 | 0.000000 | 0.000086 | 0.011257 | 0.000028 | 0.000000 | 0.000078 | 0.009838 | 950 | 0.000775 |
| top080_b425 | 0.080000 | 0.425000 | 0.671000 | 0.001100 | 0.000000 | 0.000009 | 0.000000 | 0.000057 | 0.000757 | 0.000008 | 0.000000 | 0.000053 | 0.000829 | 800 | 0.000715 |
| top075_b425 | 0.075000 | 0.425000 | 0.671000 | 0.001100 | 0.000000 | 0.000013 | 0.000000 | 0.000061 | 0.002742 | 0.000014 | 0.000000 | 0.000056 | 0.006562 | 800 | 0.000715 |
| top085_b425 | 0.085000 | 0.425000 | 0.671000 | 0.001100 | 0.000000 | 0.000015 | 0.000000 | 0.000064 | 0.011257 | 0.000014 | 0.000000 | 0.000059 | 0.007199 | 850 | 0.000715 |
| top065_b425 | 0.065000 | 0.425000 | 0.670900 | 0.001000 | -0.000100 | 0.000029 | 0.000000 | 0.000073 | 0.011340 | 0.000027 | 0.000000 | 0.000063 | 0.007446 | 800 | 0.000615 |
| top070_b450 | 0.070000 | 0.450000 | 0.670800 | 0.000900 | -0.000200 | 0.000029 | 0.000000 | 0.000134 | 0.011340 | 0.000028 | 0.000000 | 0.000117 | 0.007446 | 800 | 0.000515 |
| top070_b400 | 0.070000 | 0.400000 | 0.670700 | 0.000800 | -0.000300 | 0.000014 | 0.000000 | 0.000000 | 0.011340 | 0.000013 | 0.000000 | 0.000000 | 0.007446 | 100 | 0.000415 |
| top080_b375 | 0.080000 | 0.375000 | 0.670600 | 0.000700 | -0.000400 | 0.000009 | 0.000000 | 0.000057 | 0.000757 | 0.000008 | 0.000000 | 0.000053 | 0.000829 | 800 | 0.000315 |

## Outputs

| rank | submission | config | frac | weight | oof_delta_vs_base_champion | oof_delta_vs_public_winner_proxy | selection_score | changed_vs_public_winner | vs_public_winner_mean_delta | vs_public_winner_p95_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv | top090_b450 | 0.090000 | 0.450000 | 0.001300 | 0.000200 | 0.000915 | 900 | 0.000030 | 0.000134 |
| 2 | recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv | top090_b400 | 0.090000 | 0.400000 | 0.001200 | 0.000100 | 0.000815 | 100 | 0.000013 | 0.000000 |
| 3 | recstepgate27_rank3_top080_b450_f100s100u100_top080_b450.csv | top080_b450 | 0.080000 | 0.450000 | 0.001200 | 0.000100 | 0.000815 | 800 | 0.000016 | 0.000106 |
| 4 | recstepgate27_rank4_top080_b475_f100s100u100_top080_b475.csv | top080_b475 | 0.080000 | 0.475000 | 0.001200 | 0.000100 | 0.000815 | 800 | 0.000024 | 0.000159 |
| 5 | recstepgate27_rank5_top095_b425_f100s100u100_top095_b425.csv | top095_b425 | 0.095000 | 0.425000 | 0.001200 | 0.000100 | 0.000775 | 950 | 0.000028 | 0.000078 |
| 6 | recstepgate27_rank6_top080_b425_f100s100u100_top080_b425.csv | top080_b425 | 0.080000 | 0.425000 | 0.001100 | 0.000000 | 0.000715 | 800 | 0.000008 | 0.000053 |
| 7 | recstepgate27_rank7_top075_b425_f100s100u100_top075_b425.csv | top075_b425 | 0.075000 | 0.425000 | 0.001100 | 0.000000 | 0.000715 | 800 | 0.000014 | 0.000056 |
| 8 | recstepgate27_rank8_top085_b425_f100s100u100_top085_b425.csv | top085_b425 | 0.085000 | 0.425000 | 0.001100 | 0.000000 | 0.000715 | 850 | 0.000014 | 0.000059 |

## Recommended Public Order

1. `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv`
2. `recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv`
3. `recstepgate27_rank3_top080_b450_f100s100u100_top080_b450.csv`
4. `recstepgate27_rank4_top080_b475_f100s100u100_top080_b475.csv`
5. `recstepgate27_rank5_top095_b425_f100s100u100_top095_b425.csv`

## Decision Rule

- If rank1 improves over 0.69180, continue around that exact fraction/strength.
- If all candidates tie or drop, keep `top080_b40` as champion and move to gain selector feature engineering.
- If a narrower fraction wins, search 5%-7.5%; if a wider fraction wins, search 8.5%-11%.
