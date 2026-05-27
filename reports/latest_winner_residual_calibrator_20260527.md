# 2026-05-27 Winner Residual Calibrator

- created_at: `2026-05-27T23:20:59`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- public_feedback_after_submit: `wincal27_rank1_top03s010c0008.csv = 0.69200`
- oof_winner_hit_proxy: `0.671200`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\wincal27_rank1_top03s010c0008.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\wincal27_rank2_top04s020c0010.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\wincal27_rank3_top06s020c0012.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\wincal27_rank4_top03s025c0015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\wincal27_rank5_top05s010c0010.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\wincal27_rank6_top05s015c0010.csv']`

## Idea

- Stop fraction/strength tuning around the recursive gate plateau.
- Treat the 0.692 winner as a fixed anchor.
- Train a tiny residual calibrator on OOF winner errors.
- Apply only to predicted miss-risk top samples with sub-2mm capped corrections.

## Leaderboard

| config | top_frac | shrink | cap | oof_hit | oof_delta_vs_winner | oof_vs_winner_mean_delta | oof_vs_winner_median_delta | oof_vs_winner_p95_delta | oof_vs_winner_max_delta | test_vs_winner_mean_delta | test_vs_winner_median_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top03_s010_c0008 | 0.030000 | 0.100000 | 0.000800 | 0.671200 | 0.000000 | 0.000002 | 0.000000 | 0.000000 | 0.000080 | 0.000002 | 0.000000 | 0.000000 | 0.000080 | 0.000000 |
| top04_s020_c0010 | 0.040000 | 0.200000 | 0.001000 | 0.671200 | 0.000000 | 0.000008 | 0.000000 | 0.000000 | 0.000200 | 0.000008 | 0.000000 | 0.000000 | 0.000200 | -0.000010 |
| top06_s020_c0012 | 0.060000 | 0.200000 | 0.001200 | 0.671200 | 0.000000 | 0.000014 | 0.000000 | 0.000240 | 0.000240 | 0.000014 | 0.000000 | 0.000240 | 0.000240 | -0.000010 |
| top03_s025_c0015 | 0.030000 | 0.250000 | 0.001500 | 0.671200 | 0.000000 | 0.000011 | 0.000000 | 0.000000 | 0.000375 | 0.000011 | 0.000000 | 0.000000 | 0.000375 | -0.000020 |
| top05_s010_c0010 | 0.050000 | 0.100000 | 0.001000 | 0.671100 | -0.000100 | 0.000005 | 0.000000 | 0.000002 | 0.000100 | 0.000005 | 0.000000 | 0.000002 | 0.000100 | -0.000100 |
| top05_s015_c0010 | 0.050000 | 0.150000 | 0.001000 | 0.671100 | -0.000100 | 0.000007 | 0.000000 | 0.000002 | 0.000150 | 0.000007 | 0.000000 | 0.000003 | 0.000150 | -0.000100 |
| top08_s015_c0012 | 0.080000 | 0.150000 | 0.001200 | 0.671100 | -0.000100 | 0.000014 | 0.000000 | 0.000180 | 0.000180 | 0.000014 | 0.000000 | 0.000180 | 0.000180 | -0.000100 |
| top08_s010_c0012 | 0.080000 | 0.100000 | 0.001200 | 0.671000 | -0.000200 | 0.000010 | 0.000000 | 0.000120 | 0.000120 | 0.000009 | 0.000000 | 0.000120 | 0.000120 | -0.000200 |
| top10_s010_c0015 | 0.100000 | 0.100000 | 0.001500 | 0.671000 | -0.000200 | 0.000015 | 0.000000 | 0.000150 | 0.000150 | 0.000015 | 0.000000 | 0.000150 | 0.000150 | -0.000200 |
| top10_s015_c0015 | 0.100000 | 0.150000 | 0.001500 | 0.670900 | -0.000300 | 0.000022 | 0.000000 | 0.000225 | 0.000225 | 0.000022 | 0.000000 | 0.000225 | 0.000225 | -0.000300 |

## Outputs

| rank | submission | config | top_frac | shrink | cap | oof_delta_vs_winner | selection_score | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wincal27_rank1_top03s010c0008.csv | top03_s010_c0008 | 0.030000 | 0.100000 | 0.000800 | 0.000000 | 0.000000 | 0.000002 | 0.000000 | 0.000080 |
| 2 | wincal27_rank2_top04s020c0010.csv | top04_s020_c0010 | 0.040000 | 0.200000 | 0.001000 | 0.000000 | -0.000010 | 0.000008 | 0.000000 | 0.000200 |
| 3 | wincal27_rank3_top06s020c0012.csv | top06_s020_c0012 | 0.060000 | 0.200000 | 0.001200 | 0.000000 | -0.000010 | 0.000014 | 0.000240 | 0.000240 |
| 4 | wincal27_rank4_top03s025c0015.csv | top03_s025_c0015 | 0.030000 | 0.250000 | 0.001500 | 0.000000 | -0.000020 | 0.000011 | 0.000000 | 0.000375 |
| 5 | wincal27_rank5_top05s010c0010.csv | top05_s010_c0010 | 0.050000 | 0.100000 | 0.001000 | -0.000100 | -0.000100 | 0.000005 | 0.000002 | 0.000100 |
| 6 | wincal27_rank6_top05s015c0010.csv | top05_s015_c0010 | 0.050000 | 0.150000 | 0.001000 | -0.000100 | -0.000100 | 0.000007 | 0.000003 | 0.000150 |

## Recommended Public Order

1. `wincal27_rank1_top03s010c0008.csv`
2. `wincal27_rank2_top04s020c0010.csv`
3. `wincal27_rank3_top06s020c0012.csv`
4. `wincal27_rank4_top03s025c0015.csv`

## Decision Rule

- If rank1 is below 0.6920, stop residual calibrator immediately.
- If rank1 ties, try rank2 only if it has lower movement or different top fraction.
- If any candidate improves, continue with miss-risk feature engineering rather than recursive gate strength tuning.
