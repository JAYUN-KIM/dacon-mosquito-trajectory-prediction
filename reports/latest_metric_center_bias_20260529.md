# 2026-05-29 Metric-Center Bias

- created_at: `2026-05-30T00:04:06`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_anchor: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- oof_winner_hit_proxy: `0.671200`
- global_full_hit: `0.672100`
- local_full_hit: `0.672900`
- global_cv_hit: `0.669000`
- local_cv_hit: `0.672900`

## Idea

- Stop complex selector post-processing after the 0.692 plateau.
- Search a low-dimensional constant bias that directly maximizes OOF R-Hit@1cm.
- Test both global x/y/z bias and final-velocity local forward/side/up bias.
- Add a simple unsupervised regime-local bias as a slightly stronger variant.

## Leaderboard

| candidate | family | shrink | oof_hit | oof_delta_vs_winner | cv_hit_proxy | cv_delta_vs_winner | oof_vs_winner_mean_delta | oof_vs_winner_median_delta | oof_vs_winner_p95_delta | oof_vs_winner_max_delta | test_vs_winner_mean_delta | test_vs_winner_median_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_full | local | 1.000000 | 0.672900 | 0.001700 | 0.672900 | 0.001700 | 0.000250 | 0.000250 | 0.000250 | 0.000250 | 0.000250 | 0.000250 | 0.000250 | 0.000250 | 0.002125 |
| local_shrink075 | local | 0.750000 | 0.672500 | 0.001300 | 0.672500 | 0.001300 | 0.000188 | 0.000188 | 0.000188 | 0.000188 | 0.000188 | 0.000188 | 0.000188 | 0.000188 | 0.001625 |
| local_shrink050 | local | 0.500000 | 0.671900 | 0.000700 | 0.671900 | 0.000700 | 0.000125 | 0.000125 | 0.000125 | 0.000125 | 0.000125 | 0.000125 | 0.000125 | 0.000125 | 0.000875 |
| regimelocal_shrink065 | regime_local | 0.650000 | 0.675700 | 0.004500 | 0.670300 | -0.000900 | 0.000886 | 0.000829 | 0.001567 | 0.001657 | 0.000893 | 0.000829 | 0.001567 | 0.001657 | 0.000225 |
| regimelocal_shrink035 | regime_local | 0.350000 | 0.673200 | 0.002000 | 0.670900 | -0.000300 | 0.000477 | 0.000446 | 0.000844 | 0.000892 | 0.000481 | 0.000446 | 0.000844 | 0.000892 | 0.000200 |
| regimelocal_shrink050 | regime_local | 0.500000 | 0.673700 | 0.002500 | 0.670700 | -0.000500 | 0.000682 | 0.000637 | 0.001205 | 0.001275 | 0.000687 | 0.000637 | 0.001205 | 0.001275 | 0.000125 |
| global_shrink075 | global | 0.750000 | 0.672500 | 0.001300 | 0.669900 | -0.001300 | 0.000265 | 0.000265 | 0.000265 | 0.000265 | 0.000265 | 0.000265 | 0.000265 | 0.000265 | -0.000975 |
| global_full | global | 1.000000 | 0.672100 | 0.000900 | 0.669000 | -0.002200 | 0.000354 | 0.000354 | 0.000354 | 0.000354 | 0.000354 | 0.000354 | 0.000354 | 0.000354 | -0.001975 |

## Global Bias Top

| rank | bias_x | bias_y | bias_z | oof_hit |
| --- | --- | --- | --- | --- |
| 1.000000 | 0.000000 | -0.000250 | 0.000250 | 0.672100 |
| 2.000000 | 0.000000 | 0.000000 | 0.000250 | 0.672100 |
| 3.000000 | 0.000000 | 0.000250 | 0.000000 | 0.671800 |
| 4.000000 | 0.000250 | 0.000000 | 0.000000 | 0.671600 |
| 5.000000 | 0.000000 | 0.000250 | 0.000250 | 0.671600 |
| 6.000000 | 0.000250 | 0.000000 | 0.000250 | 0.671500 |
| 7.000000 | -0.000250 | -0.000250 | 0.000000 | 0.671400 |
| 8.000000 | -0.000250 | 0.000250 | 0.000250 | 0.671400 |
| 9.000000 | -0.000250 | 0.000250 | 0.000000 | 0.671300 |
| 10.000000 | 0.000000 | 0.000250 | -0.000250 | 0.671200 |
| 11.000000 | 0.000000 | 0.000000 | 0.000000 | 0.671200 |
| 12.000000 | 0.000000 | -0.000250 | 0.000000 | 0.671200 |

## Local Bias Top

| rank | bias_x | bias_y | bias_z | oof_hit |
| --- | --- | --- | --- | --- |
| 1.000000 | 0.000000 | 0.000000 | 0.000250 | 0.672900 |
| 2.000000 | 0.000250 | 0.000000 | 0.000250 | 0.672000 |
| 3.000000 | 0.000250 | 0.000000 | 0.000000 | 0.672000 |
| 4.000000 | 0.000000 | 0.000250 | 0.000250 | 0.671700 |
| 5.000000 | 0.000000 | -0.000250 | 0.000500 | 0.671700 |
| 6.000000 | 0.000250 | -0.000250 | 0.000250 | 0.671500 |
| 7.000000 | 0.000000 | 0.000250 | 0.000000 | 0.671500 |
| 8.000000 | 0.000500 | 0.000000 | 0.000250 | 0.671400 |
| 9.000000 | 0.000000 | -0.000250 | 0.000250 | 0.671300 |
| 10.000000 | 0.000250 | -0.000250 | 0.000000 | 0.671300 |
| 11.000000 | 0.000250 | -0.000250 | 0.000500 | 0.671200 |
| 12.000000 | 0.000250 | 0.000250 | 0.000250 | 0.671200 |

## Fold Bias Stability

| fold | global_bias | local_bias |
| --- | --- | --- |
| 1 | (0.0, -0.00025, 0.0002500000000000002) | (0.0, 0.0, 0.0002500000000000002) |
| 2 | (0.0, 0.0002500000000000002, 0.0) | (0.0, 0.0, 0.0002500000000000002) |
| 3 | (0.0, -0.00025, 0.0002500000000000002) | (0.0, 0.0, 0.0002500000000000002) |
| 4 | (-0.00025, 0.0002500000000000002, -0.0005) | (0.0, 0.0, 0.0002500000000000002) |
| 5 | (0.0, 0.0, 0.0002500000000000002) | (0.0, 0.0, 0.0002500000000000002) |

## Regime CV

| shrink | cv_hit |
| --- | --- |
| 0.350000 | 0.670900 |
| 0.500000 | 0.670700 |
| 0.650000 | 0.670300 |
| 0.800000 | 0.670000 |
| 1.000000 | 0.669100 |

## Outputs

| rank | submission | candidate | family | oof_delta_vs_winner | cv_delta_vs_winner | selection_score | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | metricbias29_rank1_localfull.csv | local_full | local | 0.001700 | 0.001700 | 0.002125 | 0.000250 | 0.000250 | 0.000250 |
| 2 | metricbias29_rank2_localshrink075.csv | local_shrink075 | local | 0.001300 | 0.001300 | 0.001625 | 0.000188 | 0.000188 | 0.000188 |
| 3 | metricbias29_rank3_localshrink050.csv | local_shrink050 | local | 0.000700 | 0.000700 | 0.000875 | 0.000125 | 0.000125 | 0.000125 |
| 4 | metricbias29_rank4_regimelocalshrink065.csv | regimelocal_shrink065 | regime_local | 0.004500 | -0.000900 | 0.000225 | 0.000893 | 0.001567 | 0.001657 |
| 5 | metricbias29_rank5_regimelocalshrink035.csv | regimelocal_shrink035 | regime_local | 0.002000 | -0.000300 | 0.000200 | 0.000481 | 0.000844 | 0.000892 |
| 6 | metricbias29_rank6_regimelocalshrink050.csv | regimelocal_shrink050 | regime_local | 0.002500 | -0.000500 | 0.000125 | 0.000687 | 0.001205 | 0.001275 |

## Recommended Public Order

1. `metricbias29_rank1_localfull.csv`
2. `metricbias29_rank2_localshrink075.csv`
3. `metricbias29_rank3_localshrink050.csv`
4. `metricbias29_rank4_regimelocalshrink065.csv`

## Decision Rule

- If rank1 improves or ties, continue with low-variance metric-center calibration.
- If rank1 drops, avoid constant-bias tuning and move to a genuinely new target formulation.
