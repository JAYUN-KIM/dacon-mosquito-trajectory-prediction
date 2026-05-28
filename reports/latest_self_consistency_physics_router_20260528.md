# 2026-05-28 Self-Consistency Physics Router

- created_at: `2026-05-28T23:02:39`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- oof_winner_hit_proxy: `0.671200`

## Idea

- Stop tuning the recursive gate itself.
- Build many physics/poly/smoothed-difference candidates.
- For each trajectory, score candidates by how well they predict recent observed endpoints from earlier prefixes.
- Move the 0.692 winner slightly toward this self-consistent physics estimate only on learned gain-positive samples.

## Leaderboard

| config | top_frac | blend | cap | temperature | self_oof_hit | gain_positive_rate | oof_hit | oof_delta_vs_winner | oof_vs_winner_mean_delta | oof_vs_winner_median_delta | oof_vs_winner_p95_delta | oof_vs_winner_max_delta | test_vs_winner_mean_delta | test_vs_winner_median_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top05_b016_c0018_t090 | 0.050000 | 0.160000 | 0.001800 | 0.900000 | 0.596900 | 0.321700 | 0.671400 | 0.000200 | 0.000003 | 0.000000 | 0.000000 | 0.000285 | 0.000003 | 0.000000 | 0.000000 | 0.000288 | 0.000200 |
| top025_b030_c0025_t085 | 0.025000 | 0.300000 | 0.002500 | 0.850000 | 0.597100 | 0.322600 | 0.671400 | 0.000200 | 0.000005 | 0.000000 | 0.000000 | 0.000750 | 0.000003 | 0.000000 | 0.000000 | 0.000750 | 0.000200 |
| top04_b024_c0022_t090 | 0.040000 | 0.240000 | 0.002200 | 0.900000 | 0.596900 | 0.321700 | 0.671400 | 0.000200 | 0.000004 | 0.000000 | 0.000000 | 0.000518 | 0.000004 | 0.000000 | 0.000000 | 0.000528 | 0.000200 |
| top06_b026_c0030_t100 | 0.060000 | 0.260000 | 0.003000 | 1.000000 | 0.595500 | 0.320300 | 0.671400 | 0.000200 | 0.000009 | 0.000000 | 0.000027 | 0.000780 | 0.000006 | 0.000000 | 0.000021 | 0.000780 | 0.000200 |
| top03_b018_c0015_t090 | 0.030000 | 0.180000 | 0.001500 | 0.900000 | 0.596900 | 0.321700 | 0.671300 | 0.000100 | 0.000002 | 0.000000 | 0.000000 | 0.000267 | 0.000002 | 0.000000 | 0.000000 | 0.000270 | 0.000100 |
| top06_b018_c0020_t100 | 0.060000 | 0.180000 | 0.002000 | 1.000000 | 0.595500 | 0.320300 | 0.671300 | 0.000100 | 0.000005 | 0.000000 | 0.000016 | 0.000360 | 0.000003 | 0.000000 | 0.000012 | 0.000360 | 0.000100 |
| top10_b012_c0018_t120 | 0.100000 | 0.120000 | 0.001800 | 1.200000 | 0.594200 | 0.319300 | 0.671200 | 0.000000 | 0.000004 | 0.000000 | 0.000029 | 0.000216 | 0.000004 | 0.000000 | 0.000027 | 0.000216 | 0.000000 |
| top08_b014_c0020_t110 | 0.080000 | 0.140000 | 0.002000 | 1.100000 | 0.594800 | 0.319600 | 0.671200 | 0.000000 | 0.000005 | 0.000000 | 0.000029 | 0.000280 | 0.000004 | 0.000000 | 0.000025 | 0.000280 | 0.000000 |

## Candidate Notes

| config | candidate_count | best_internal_candidate_top1 |
| --- | --- | --- |
| top03_b018_c0015_t090 | 56 | phys_v0.94_a0.30 |
| top05_b016_c0018_t090 | 56 | phys_v0.94_a0.30 |
| top06_b018_c0020_t100 | 56 | phys_v0.94_a0.30 |
| top08_b014_c0020_t110 | 56 | phys_v0.94_a0.30 |
| top04_b024_c0022_t090 | 56 | phys_v0.94_a0.30 |
| top10_b012_c0018_t120 | 56 | phys_v0.94_a0.30 |
| top025_b030_c0025_t085 | 56 | phys_v0.94_a0.30 |
| top06_b026_c0030_t100 | 56 | phys_v0.94_a0.30 |

## Outputs

| rank | submission | config | oof_delta_vs_winner | selection_score | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | selfcons28_rank1_top05b016c0018t090.csv | top05_b016_c0018_t090 | 0.000200 | 0.000200 | 0.000003 | 0.000000 | 0.000288 |
| 2 | selfcons28_rank2_top025b030c0025t085.csv | top025_b030_c0025_t085 | 0.000200 | 0.000200 | 0.000003 | 0.000000 | 0.000750 |
| 3 | selfcons28_rank3_top04b024c0022t090.csv | top04_b024_c0022_t090 | 0.000200 | 0.000200 | 0.000004 | 0.000000 | 0.000528 |
| 4 | selfcons28_rank4_top06b026c0030t100.csv | top06_b026_c0030_t100 | 0.000200 | 0.000200 | 0.000006 | 0.000021 | 0.000780 |
| 5 | selfcons28_rank5_top03b018c0015t090.csv | top03_b018_c0015_t090 | 0.000100 | 0.000100 | 0.000002 | 0.000000 | 0.000270 |
| 6 | selfcons28_rank6_top06b018c0020t100.csv | top06_b018_c0020_t100 | 0.000100 | 0.000100 | 0.000003 | 0.000012 | 0.000360 |

## Recommended Public Order

1. `selfcons28_rank1_top05b016c0018t090.csv`
2. `selfcons28_rank2_top025b030c0025t085.csv`
3. `selfcons28_rank3_top04b024c0022t090.csv`
4. `selfcons28_rank4_top06b026c0030t100.csv`

## Decision Rule

- If rank1 is below 0.6920, self-consistency physics is not complementary enough.
- If rank1 improves, refine candidate families and internal backtest cutoffs before touching recursive gate strength again.
