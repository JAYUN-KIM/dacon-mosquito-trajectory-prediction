# 2026-05-28 Local-Axis Scale Calibrator

- created_at: `2026-05-28T22:55:53`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- oof_winner_hit_proxy: `0.671200`

## Idea

- Stop recursive gate fraction/strength tuning around the 0.692 plateau.
- Express the winner endpoint displacement in the final-velocity local frame.
- Learn axis-wise multiplicative scale deltas instead of a direct global residual.
- Apply only to predicted miss-risk samples with sub-millimeter to 1.2mm capped corrections.

## Leaderboard

| config | mode | top_frac | shrink | cap | oof_hit | oof_delta_vs_winner | oof_vs_winner_mean_delta | oof_vs_winner_median_delta | oof_vs_winner_p95_delta | oof_vs_winner_max_delta | test_vs_winner_mean_delta | test_vs_winner_median_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_top03_s015_c0009 | all | 0.030000 | 0.150000 | 0.000900 | 0.671100 | -0.000100 | 0.000001 | 0.000000 | 0.000000 | 0.000135 | 0.000001 | 0.000000 | 0.000000 | 0.000135 | -0.000100 |
| sideup_top04_s018_c0010 | sideup | 0.040000 | 0.180000 | 0.001000 | 0.671100 | -0.000100 | 0.000002 | 0.000000 | 0.000000 | 0.000178 | 0.000002 | 0.000000 | 0.000000 | 0.000180 | -0.000100 |
| fwd_top04_s020_c0010 | forward | 0.040000 | 0.200000 | 0.001000 | 0.671100 | -0.000100 | 0.000003 | 0.000000 | 0.000000 | 0.000198 | 0.000002 | 0.000000 | 0.000000 | 0.000183 | -0.000100 |
| sideup_top03_s025_c0012 | sideup | 0.030000 | 0.250000 | 0.001200 | 0.671100 | -0.000100 | 0.000003 | 0.000000 | 0.000000 | 0.000296 | 0.000003 | 0.000000 | 0.000000 | 0.000300 | -0.000100 |
| fwd_top04_s025_c0012 | forward | 0.040000 | 0.250000 | 0.001200 | 0.671100 | -0.000100 | 0.000004 | 0.000000 | 0.000000 | 0.000297 | 0.000003 | 0.000000 | 0.000000 | 0.000275 | -0.000100 |
| forwardside_top05_s018_c0011 | forwardside | 0.050000 | 0.180000 | 0.001100 | 0.671100 | -0.000100 | 0.000003 | 0.000000 | 0.000000 | 0.000196 | 0.000003 | 0.000000 | 0.000000 | 0.000198 | -0.000100 |
| all_top05_s018_c0011 | all | 0.050000 | 0.180000 | 0.001100 | 0.671100 | -0.000100 | 0.000004 | 0.000000 | 0.000000 | 0.000196 | 0.000004 | 0.000000 | 0.000000 | 0.000198 | -0.000100 |
| fwd_top06_s020_c0012 | forward | 0.060000 | 0.200000 | 0.001200 | 0.671100 | -0.000100 | 0.000004 | 0.000000 | 0.000013 | 0.000238 | 0.000004 | 0.000000 | 0.000010 | 0.000225 | -0.000100 |
| sideup_top06_s020_c0012 | sideup | 0.060000 | 0.200000 | 0.001200 | 0.671100 | -0.000100 | 0.000004 | 0.000000 | 0.000017 | 0.000238 | 0.000004 | 0.000000 | 0.000013 | 0.000240 | -0.000100 |
| all_top08_s015_c0012 | all | 0.080000 | 0.150000 | 0.001200 | 0.671100 | -0.000100 | 0.000005 | 0.000000 | 0.000040 | 0.000179 | 0.000005 | 0.000000 | 0.000037 | 0.000180 | -0.000100 |

## Outputs

| rank | submission | config | mode | top_frac | shrink | cap | oof_delta_vs_winner | selection_score | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | locaxis28_rank1_alltop03s015c0009.csv | all_top03_s015_c0009 | all | 0.030000 | 0.150000 | 0.000900 | -0.000100 | -0.000100 | 0.000001 | 0.000000 | 0.000135 |
| 2 | locaxis28_rank2_sideuptop04s018c0010.csv | sideup_top04_s018_c0010 | sideup | 0.040000 | 0.180000 | 0.001000 | -0.000100 | -0.000100 | 0.000002 | 0.000000 | 0.000180 |
| 3 | locaxis28_rank3_fwdtop04s020c0010.csv | fwd_top04_s020_c0010 | forward | 0.040000 | 0.200000 | 0.001000 | -0.000100 | -0.000100 | 0.000002 | 0.000000 | 0.000183 |
| 4 | locaxis28_rank4_sideuptop03s025c0012.csv | sideup_top03_s025_c0012 | sideup | 0.030000 | 0.250000 | 0.001200 | -0.000100 | -0.000100 | 0.000003 | 0.000000 | 0.000300 |
| 5 | locaxis28_rank5_fwdtop04s025c0012.csv | fwd_top04_s025_c0012 | forward | 0.040000 | 0.250000 | 0.001200 | -0.000100 | -0.000100 | 0.000003 | 0.000000 | 0.000275 |
| 6 | locaxis28_rank6_forwardsidetop05s018c0011.csv | forwardside_top05_s018_c0011 | forwardside | 0.050000 | 0.180000 | 0.001100 | -0.000100 | -0.000100 | 0.000003 | 0.000000 | 0.000198 |

## Recommended Public Order

1. `locaxis28_rank1_alltop03s015c0009.csv`
2. `locaxis28_rank2_sideuptop04s018c0010.csv`
3. `locaxis28_rank3_fwdtop04s020c0010.csv`
4. `locaxis28_rank4_sideuptop03s025c0012.csv`

## Decision Rule

- If rank1 is below 0.6920, demote this axis quickly.
- If rank1 ties 0.6920 and rank2 differs by axis mode, try rank2 only if submissions remain.
- If any candidate improves, refine by axis mode rather than recursive gate strength.
