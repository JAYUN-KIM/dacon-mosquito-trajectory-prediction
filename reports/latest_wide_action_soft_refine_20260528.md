# 2026-05-28 Wide Action Soft Refine

- created_at: `2026-05-28T23:26:43`
- base_winner: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv`
- soft08: `wideact28_rank1_softblend08.csv`
- soft12: `wideact28_rank2_softblend12.csv`

## Outputs

| submission | rule | test_vs_winner_mean_delta | test_vs_winner_median_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | test_vs_soft08_mean_delta | test_vs_soft08_median_delta | test_vs_soft08_p95_delta | test_vs_soft08_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideact28_refine_softblend06.csv | soft08_direction_x075 | 0.000153 | 0.000150 | 0.000240 | 0.000240 | 0.000051 | 0.000050 | 0.000080 | 0.000080 |
| wideact28_refine_softblend10.csv | soft08_direction_x125 | 0.000254 | 0.000250 | 0.000400 | 0.000400 | 0.000051 | 0.000050 | 0.000080 | 0.000080 |
| wideact28_refine_softblend08_12avg.csv | avg_soft08_soft12 | 0.000268 | 0.000250 | 0.000460 | 0.000460 | 0.000064 | 0.000050 | 0.000140 | 0.000140 |

## Recommended Use

1. Submit `wideact28_rank1_softblend08.csv` first.
2. If it improves, try `wideact28_refine_softblend10.csv` or `wideact28_rank2_softblend12.csv`.
3. If it drops slightly, try `wideact28_refine_softblend06.csv` as the safer same-direction version.
