# 2026-05-28 Wide Physics Action Selector

- created_at: `2026-05-28T23:25:47`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- oof_winner_hit_proxy: `0.671200`
- selected_candidate_count: `48`

## Idea

- Restart from a broad analytic physics candidate pool.
- Include the current 0.692 winner as just one action among many.
- Train a multiclass action selector to choose the candidate that best optimizes hit@1cm per sample.
- Submit both soft and confidence-gated moves from the winner toward selected actions.

## Leaderboard

| config | mode | threshold | blend | cap | top_frac | oof_hit | oof_delta_vs_winner | oof_vs_winner_mean_delta | oof_vs_winner_median_delta | oof_vs_winner_p95_delta | oof_vs_winner_max_delta | test_vs_winner_mean_delta | test_vs_winner_median_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soft_blend08 | soft_blend | 0.000000 | 0.080000 | 0.004000 | 1.000000 | 0.672800 | 0.001600 | 0.000205 | 0.000203 | 0.000320 | 0.000320 | 0.000204 | 0.000200 | 0.000320 | 0.000320 | 0.001600 |
| soft_blend12 | soft_blend | 0.000000 | 0.120000 | 0.005000 | 1.000000 | 0.672100 | 0.000900 | 0.000336 | 0.000305 | 0.000600 | 0.000600 | 0.000332 | 0.000300 | 0.000600 | 0.000600 | 0.000893 |
| top08_conf18_b40_cap007 | top_gate | 0.180000 | 0.400000 | 0.007000 | 0.080000 | 0.670700 | -0.000500 | 0.000149 | 0.000000 | 0.001448 | 0.002800 | 0.000151 | 0.000000 | 0.001519 | 0.002800 | -0.000500 |
| top05_conf20_b55_cap008 | top_gate | 0.200000 | 0.550000 | 0.008000 | 0.050000 | 0.670400 | -0.000800 | 0.000138 | 0.000000 | 0.000010 | 0.004400 | 0.000142 | 0.000000 | 0.000006 | 0.004400 | -0.000800 |
| top03_conf25_b70_cap010 | top_gate | 0.250000 | 0.700000 | 0.010000 | 0.030000 | 0.669700 | -0.001500 | 0.000117 | 0.000000 | 0.000000 | 0.007000 | 0.000127 | 0.000000 | 0.000000 | 0.007000 | -0.001500 |
| hard_conf22_b35_cap006 | hard_gate | 0.220000 | 0.350000 | 0.006000 | 1.000000 | 0.668600 | -0.002600 | 0.000653 | 0.000000 | 0.002100 | 0.002100 | 0.000542 | 0.000000 | 0.002100 | 0.002100 | -0.002619 |
| hard_conf35_b60_cap008 | hard_gate | 0.350000 | 0.600000 | 0.008000 | 1.000000 | 0.667000 | -0.004200 | 0.000388 | 0.000000 | 0.003904 | 0.004800 | 0.000224 | 0.000000 | 0.002022 | 0.004800 | -0.004202 |
| hard_conf28_b45_cap007 | hard_gate | 0.280000 | 0.450000 | 0.007000 | 1.000000 | 0.666700 | -0.004500 | 0.000548 | 0.000000 | 0.003150 | 0.003150 | 0.000378 | 0.000000 | 0.003150 | 0.003150 | -0.004532 |

## Candidate Pool

| candidate | oof_hit | oracle_count |
| --- | --- | --- |
| winner_recgate09_b45 | 0.671200 | 1028 |
| base_champion | 0.669900 | 100 |
| recursive_onestep | 0.651500 | 809 |
| turn_tm025s05d098 | 0.628300 | 178 |
| turn_tm020s04d098 | 0.625800 | 95 |
| turn_tm015s03d100 | 0.622900 | 200 |
| turn_tp020s04d098 | 0.619100 | 125 |
| turn_tm030s06d096 | 0.617100 | 490 |
| phys_v1.00_a0.30 | 0.600800 | 32 |
| phys_v1.00_a0.22 | 0.598400 | 24 |
| phys_v1.00_a0.15 | 0.596200 | 37 |
| phys_v0.97_a0.30 | 0.594900 | 40 |
| phys_v0.97_a0.22 | 0.593100 | 37 |
| phys_v1.00_a0.08 | 0.592100 | 39 |
| phys_v1.00_a0.40 | 0.591600 | 55 |
| phys_v0.97_a0.15 | 0.589900 | 44 |
| phys_v1.03_a0.22 | 0.589200 | 39 |
| phys_v0.97_a0.40 | 0.588700 | 44 |
| phys_v1.03_a0.15 | 0.587300 | 37 |
| phys_v1.03_a0.08 | 0.586300 | 21 |
| phys_v0.97_a0.08 | 0.584500 | 68 |
| phys_v1.03_a0.30 | 0.584300 | 52 |
| phys_v1.00_a0.52 | 0.580900 | 203 |
| phys_v0.97_a0.52 | 0.579600 | 144 |
| phys_v1.03_a0.00 | 0.579100 | 83 |
| phys_v1.00_a0.00 | 0.578800 | 54 |
| phys_v1.00_a-0.06 | 0.578000 | 111 |
| phys_v1.03_a0.40 | 0.577900 | 122 |
| phys_v0.94_a0.52 | 0.557500 | 134 |
| poly_w3_d1 | 0.512400 | 312 |

## Outputs

| rank | submission | config | oof_delta_vs_winner | selection_score | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wideact28_rank1_softblend08.csv | soft_blend08 | 0.001600 | 0.001600 | 0.000204 | 0.000320 | 0.000320 |
| 2 | wideact28_rank2_softblend12.csv | soft_blend12 | 0.000900 | 0.000893 | 0.000332 | 0.000600 | 0.000600 |
| 3 | wideact28_rank3_top08conf18b40cap007.csv | top08_conf18_b40_cap007 | -0.000500 | -0.000500 | 0.000151 | 0.001519 | 0.002800 |
| 4 | wideact28_rank4_top05conf20b55cap008.csv | top05_conf20_b55_cap008 | -0.000800 | -0.000800 | 0.000142 | 0.000006 | 0.004400 |
| 5 | wideact28_rank5_top03conf25b70cap010.csv | top03_conf25_b70_cap010 | -0.001500 | -0.001500 | 0.000127 | 0.000000 | 0.007000 |
| 6 | wideact28_rank6_hardconf22b35cap006.csv | hard_conf22_b35_cap006 | -0.002600 | -0.002619 | 0.000542 | 0.002100 | 0.002100 |

## Recommended Public Order

1. `wideact28_rank1_softblend08.csv`
2. `wideact28_rank2_softblend12.csv`
3. `wideact28_rank3_top08conf18b40cap007.csv`
4. `wideact28_rank4_top05conf20b55cap008.csv`

## Decision Rule

- If rank1 collapses, wide action selection is overfitting OOF and should be abandoned.
- If rank1 ties or improves, inspect which analytic actions dominate the selected pool and refine that family.
