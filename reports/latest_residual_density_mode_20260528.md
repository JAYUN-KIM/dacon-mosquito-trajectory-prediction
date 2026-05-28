# 2026-05-28 Residual Density Mode

- created_at: `2026-05-28T23:12:39`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- oof_winner_hit_proxy: `0.671200`

## Idea

- Treat the 0.692 recursive winner as the origin.
- Retrieve similar train trajectories in feature space.
- Estimate the residual density mode that maximizes local probability mass inside the 1cm hit ball.
- Move only the highest density-gain test samples by a capped fraction of that residual mode.

## Leaderboard

| config | k | centers | sigma | top_frac | shrink | cap | oof_hit | oof_delta_vs_winner | oof_gain_p95 | test_gain_p95 | test_score_p95 | oof_vs_winner_mean_delta | oof_vs_winner_median_delta | oof_vs_winner_p95_delta | oof_vs_winner_max_delta | test_vs_winner_mean_delta | test_vs_winner_median_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k192_c80_s006_top08_b025_cap004 | 192 | 80 | 0.006000 | 0.080000 | 0.250000 | 0.004000 | 0.671300 | 0.000100 | 0.002301 | 0.002407 | 0.802087 | 0.000003 | 0.000000 | 0.000012 | 0.000506 | 0.000004 | 0.000000 | 0.000013 | 0.000765 | 0.000100 |
| k128_c64_s006_top10_b020_cap003 | 128 | 64 | 0.006000 | 0.100000 | 0.200000 | 0.003000 | 0.671300 | 0.000100 | 0.003742 | 0.003880 | 0.809153 | 0.000004 | 0.000000 | 0.000017 | 0.000578 | 0.000004 | 0.000000 | 0.000020 | 0.000504 | 0.000100 |
| k256_c96_s006_top06_b035_cap005 | 256 | 96 | 0.006000 | 0.060000 | 0.350000 | 0.005000 | 0.671200 | 0.000000 | 0.001728 | 0.001802 | 0.794779 | 0.000003 | 0.000000 | 0.000006 | 0.000500 | 0.000003 | 0.000000 | 0.000005 | 0.000680 | 0.000000 |
| k224_c80_s005_top04_b040_cap004 | 224 | 80 | 0.005000 | 0.040000 | 0.400000 | 0.004000 | 0.671200 | 0.000000 | 0.002593 | 0.002624 | 0.746300 | 0.000003 | 0.000000 | 0.000000 | 0.000651 | 0.000003 | 0.000000 | 0.000000 | 0.001224 | 0.000000 |
| k160_c64_s005_top05_b030_cap003 | 160 | 64 | 0.005000 | 0.050000 | 0.300000 | 0.003000 | 0.671200 | 0.000000 | 0.003527 | 0.003700 | 0.754146 | 0.000004 | 0.000000 | 0.000000 | 0.000900 | 0.000004 | 0.000000 | 0.000000 | 0.000900 | 0.000000 |
| k128_c48_s004_top03_b035_cap003 | 128 | 48 | 0.004000 | 0.030000 | 0.350000 | 0.003000 | 0.671000 | -0.000200 | 0.006005 | 0.006159 | 0.687616 | 0.000003 | 0.000000 | 0.000000 | 0.001050 | 0.000005 | 0.000000 | 0.000000 | 0.001050 | -0.000200 |
| k096_c32_s0035_top02_b055_cap004 | 96 | 32 | 0.003500 | 0.020000 | 0.550000 | 0.004000 | 0.671000 | -0.000200 | 0.009230 | 0.009592 | 0.647757 | 0.000005 | 0.000000 | 0.000000 | 0.001589 | 0.000007 | 0.000000 | 0.000000 | 0.001917 | -0.000200 |
| k096_c48_s004_top025_b045_cap004 | 96 | 48 | 0.004000 | 0.025000 | 0.450000 | 0.004000 | 0.670900 | -0.000300 | 0.008546 | 0.008777 | 0.695144 | 0.000005 | 0.000000 | 0.000000 | 0.001449 | 0.000007 | 0.000000 | 0.000000 | 0.001647 | -0.000300 |

## Outputs

| rank | submission | config | oof_delta_vs_winner | selection_score | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | densmode28_rank1_k192c80s006top08b025cap004.csv | k192_c80_s006_top08_b025_cap004 | 0.000100 | 0.000100 | 0.000004 | 0.000013 | 0.000765 |
| 2 | densmode28_rank2_k128c64s006top10b020cap003.csv | k128_c64_s006_top10_b020_cap003 | 0.000100 | 0.000100 | 0.000004 | 0.000020 | 0.000504 |
| 3 | densmode28_rank3_k256c96s006top06b035cap005.csv | k256_c96_s006_top06_b035_cap005 | 0.000000 | 0.000000 | 0.000003 | 0.000005 | 0.000680 |
| 4 | densmode28_rank4_k224c80s005top04b040cap004.csv | k224_c80_s005_top04_b040_cap004 | 0.000000 | 0.000000 | 0.000003 | 0.000000 | 0.001224 |
| 5 | densmode28_rank5_k160c64s005top05b030cap003.csv | k160_c64_s005_top05_b030_cap003 | 0.000000 | 0.000000 | 0.000004 | 0.000000 | 0.000900 |
| 6 | densmode28_rank6_k128c48s004top03b035cap003.csv | k128_c48_s004_top03_b035_cap003 | -0.000200 | -0.000200 | 0.000005 | 0.000000 | 0.001050 |

## Recommended Public Order

1. `densmode28_rank1_k192c80s006top08b025cap004.csv`
2. `densmode28_rank2_k128c64s006top10b020cap003.csv`
3. `densmode28_rank3_k256c96s006top06b035cap005.csv`
4. `densmode28_rank4_k224c80s005top04b040cap004.csv`

## Decision Rule

- If rank1 is below 0.6920, density-mode retrieval is still not public-stable.
- If rank1 ties but rank2 improves movement diversity, try rank2 only if submissions remain.
- If any candidate improves, refine sigma/K and residual-mode candidate centers instead of returning to recursive gate tuning.
