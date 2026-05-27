# 2026-05-27 Recursive One-Step Gate Peak

- created_at: `2026-05-27T23:13:38`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_winner: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- public_feedback: `{'recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv': 0.692, 'recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv': 0.6916, 'recstepgate27b_rank1_top090_b475_f100s100u100_top090_b475.csv': 0.692}`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank1_top090_b475_f100s100u100_top090_b475.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank2_top095_b475_f100s100u100_top095_b475.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank3_top100_b475_f100s100u100_top100_b475.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank4_top095_b450_f100s100u100_top095_b450.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank5_top100_b450_f100s100u100_top100_b450.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank6_top090_b500_f100s100u100_top090_b500.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank7_top090_b525_f100s100u100_top090_b525.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\recstepgate27b_rank8_top085_b450_f100s100u100_top085_b450.csv']`

## Idea

- Public feedback says `top090_b450` improved to 0.6920, while `top090_b400` dropped to 0.6916.
- Therefore the selected fraction around 9% is plausible, and strength should be >= 45%.
- Search 47.5%-52.5% at top 9%, plus 9.5%-10% region checks.

## Candidate Leaderboard

| config | frac | weight | oof_hit | oof_delta_vs_base_champion | oof_delta_vs_public_proxy | oof_vs_public_proxy_mean_delta | oof_vs_public_proxy_median_delta | oof_vs_public_proxy_p95_delta | oof_vs_public_proxy_max_delta | vs_public_winner_mean_delta | vs_public_winner_median_delta | vs_public_winner_p95_delta | vs_public_winner_max_delta | changed_vs_public_winner | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top090_b475 | 0.090000 | 0.475000 | 0.671300 | 0.001400 | 0.000100 | 0.000009 | 0.000000 | 0.000063 | 0.000757 | 0.000009 | 0.000000 | 0.000059 | 0.000829 | 900 | 0.000815 |
| top095_b475 | 0.095000 | 0.475000 | 0.671300 | 0.001400 | 0.000100 | 0.000016 | 0.000000 | 0.000072 | 0.007851 | 0.000016 | 0.000000 | 0.000064 | 0.004864 | 950 | 0.000815 |
| top100_b475 | 0.100000 | 0.475000 | 0.671300 | 0.001400 | 0.000100 | 0.000026 | 0.000000 | 0.000083 | 0.013354 | 0.000023 | 0.000000 | 0.000074 | 0.005452 | 1000 | 0.000815 |
| top095_b450 | 0.095000 | 0.450000 | 0.671200 | 0.001300 | 0.000000 | 0.000007 | 0.000000 | 0.000000 | 0.007438 | 0.000007 | 0.000000 | 0.000000 | 0.004608 | 50 | 0.000715 |
| top100_b450 | 0.100000 | 0.450000 | 0.671200 | 0.001300 | 0.000000 | 0.000015 | 0.000000 | 0.000000 | 0.012651 | 0.000013 | 0.000000 | 0.000000 | 0.005165 | 100 | 0.000715 |
| top090_b500 | 0.090000 | 0.500000 | 0.671200 | 0.001300 | 0.000000 | 0.000019 | 0.000000 | 0.000127 | 0.001513 | 0.000018 | 0.000000 | 0.000117 | 0.001659 | 900 | 0.000715 |
| top090_b525 | 0.090000 | 0.525000 | 0.671200 | 0.001300 | 0.000000 | 0.000028 | 0.000000 | 0.000190 | 0.002270 | 0.000026 | 0.000000 | 0.000176 | 0.002488 | 900 | 0.000715 |
| top085_b450 | 0.085000 | 0.450000 | 0.671100 | 0.001200 | -0.000100 | 0.000007 | 0.000000 | 0.000000 | 0.005175 | 0.000008 | 0.000000 | 0.000000 | 0.010416 | 50 | 0.000615 |
| top090_b425 | 0.090000 | 0.425000 | 0.671100 | 0.001200 | -0.000100 | 0.000009 | 0.000000 | 0.000063 | 0.000757 | 0.000009 | 0.000000 | 0.000059 | 0.000829 | 900 | 0.000615 |
| top085_b475 | 0.085000 | 0.475000 | 0.671100 | 0.001200 | -0.000100 | 0.000016 | 0.000000 | 0.000069 | 0.005175 | 0.000016 | 0.000000 | 0.000062 | 0.010416 | 900 | 0.000615 |

## Outputs

| rank | submission | config | frac | weight | oof_delta_vs_base_champion | oof_delta_vs_public_proxy | selection_score | changed_vs_public_winner | vs_public_winner_mean_delta | vs_public_winner_p95_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | recstepgate27b_rank1_top090_b475_f100s100u100_top090_b475.csv | top090_b475 | 0.090000 | 0.475000 | 0.001400 | 0.000100 | 0.000815 | 900 | 0.000009 | 0.000059 |
| 2 | recstepgate27b_rank2_top095_b475_f100s100u100_top095_b475.csv | top095_b475 | 0.095000 | 0.475000 | 0.001400 | 0.000100 | 0.000815 | 950 | 0.000016 | 0.000064 |
| 3 | recstepgate27b_rank3_top100_b475_f100s100u100_top100_b475.csv | top100_b475 | 0.100000 | 0.475000 | 0.001400 | 0.000100 | 0.000815 | 1000 | 0.000023 | 0.000074 |
| 4 | recstepgate27b_rank4_top095_b450_f100s100u100_top095_b450.csv | top095_b450 | 0.095000 | 0.450000 | 0.001300 | 0.000000 | 0.000715 | 50 | 0.000007 | 0.000000 |
| 5 | recstepgate27b_rank5_top100_b450_f100s100u100_top100_b450.csv | top100_b450 | 0.100000 | 0.450000 | 0.001300 | 0.000000 | 0.000715 | 100 | 0.000013 | 0.000000 |
| 6 | recstepgate27b_rank6_top090_b500_f100s100u100_top090_b500.csv | top090_b500 | 0.090000 | 0.500000 | 0.001300 | 0.000000 | 0.000715 | 900 | 0.000018 | 0.000117 |
| 7 | recstepgate27b_rank7_top090_b525_f100s100u100_top090_b525.csv | top090_b525 | 0.090000 | 0.525000 | 0.001300 | 0.000000 | 0.000715 | 900 | 0.000026 | 0.000176 |
| 8 | recstepgate27b_rank8_top085_b450_f100s100u100_top085_b450.csv | top085_b450 | 0.085000 | 0.450000 | 0.001200 | -0.000100 | 0.000615 | 50 | 0.000008 | 0.000000 |

## Recommended Public Order

1. `recstepgate27b_rank1_top090_b475_f100s100u100_top090_b475.csv`
2. `recstepgate27b_rank2_top095_b475_f100s100u100_top095_b475.csv`
3. `recstepgate27b_rank3_top100_b475_f100s100u100_top100_b475.csv`
4. `recstepgate27b_rank4_top095_b450_f100s100u100_top095_b450.csv`
5. `recstepgate27b_rank5_top100_b450_f100s100u100_top100_b450.csv`

## Decision Rule

- If rank1 improves, continue strength search above 0.475 at top 9%.
- If rank2 or rank3 wins, the selected fraction should widen toward 9.5%-10%.
- If all drop, keep `top090_b450` as champion and switch to gain selector feature engineering.
