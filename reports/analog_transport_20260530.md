# analog_transport_20260530

- created_at: `2026-05-30T16:46:49`
- public_anchor: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv = 0.69200`
- axis: orientation-equivariant analog transport

## Idea

- Convert each trajectory to a final-velocity local frame.
- Retrieve similar train trajectory shapes after translation/orientation/scale normalization.
- Reuse neighbors' +80ms local displacement distribution and transport it to each test frame.
- Also create winner-anchored selective moves as low-risk probes.

## Physics Proxy Baselines

- CV hit: `0.578800`
- CA hit: `0.513100`

## Recommended Order

| recommend_rank | candidate | kind | oof_hit | oof_mean_dist | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selected_frac | recommend_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | analog_transport_20260530_winner_move_trimmean_k128_p15_top100_a25.csv | winner_selective_move | 0.578800 | 0.012972 | 0.000185 | 0.001523 | 0.007551 | 0.100000 | 0.580755 |
| 2 | analog_transport_20260530_winner_move_mean_k64_p2_top050_a25.csv | winner_selective_move | 0.578700 | 0.012954 | 0.000076 | 0.000003 | 0.006022 | 0.050000 | 0.580694 |
| 3 | analog_transport_20260530_winner_move_median_k96_p2_top070_a30.csv | winner_selective_move | 0.578400 | 0.012966 | 0.000134 | 0.000838 | 0.009342 | 0.070000 | 0.580373 |
| 4 | analog_transport_20260530_winner_move_mean_k64_p2_top090_a35.csv | winner_selective_move | 0.577100 | 0.013000 | 0.000230 | 0.001799 | 0.010576 | 0.090000 | 0.579046 |
| 5 | analog_transport_20260530_mean_k64_p2_cvblend_a35.csv | physics_blend | 0.395200 | 0.015487 | 0.008457 | 0.017957 | 0.081083 | 1.000000 | 0.394964 |
| 6 | analog_transport_20260530_trimmean_k128_p15_cablend_a30.csv | physics_blend | 0.384200 | 0.016492 | 0.010292 | 0.022433 | 0.117333 | 1.000000 | 0.383728 |
| 7 | analog_transport_20260530_median_k96_p2_cvblend_a45.csv | physics_blend | 0.375900 | 0.016402 | 0.009565 | 0.022642 | 0.081492 | 1.000000 | 0.375482 |
| 8 | analog_transport_20260530_median_k96_p2.csv | analog_only | 0.217800 | 0.023922 | 0.018152 | 0.047050 | 0.091545 | 1.000000 | 0.215407 |

## All Candidates

| recommend_rank | candidate | kind | oof_hit | oof_mean_dist | test_vs_winner_mean_delta | test_vs_winner_p95_delta | test_vs_winner_max_delta | selected_frac | recommend_score | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | analog_transport_20260530_winner_move_trimmean_k128_p15_top100_a25.csv | winner_selective_move | 0.578800 | 0.012972 | 0.000185 | 0.001523 | 0.007551 | 0.100000 | 0.580755 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_winner_move_trimmean_k128_p15_top100_a25.csv |
| 2 | analog_transport_20260530_winner_move_mean_k64_p2_top050_a25.csv | winner_selective_move | 0.578700 | 0.012954 | 0.000076 | 0.000003 | 0.006022 | 0.050000 | 0.580694 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_winner_move_mean_k64_p2_top050_a25.csv |
| 3 | analog_transport_20260530_winner_move_median_k96_p2_top070_a30.csv | winner_selective_move | 0.578400 | 0.012966 | 0.000134 | 0.000838 | 0.009342 | 0.070000 | 0.580373 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_winner_move_median_k96_p2_top070_a30.csv |
| 4 | analog_transport_20260530_winner_move_mean_k64_p2_top090_a35.csv | winner_selective_move | 0.577100 | 0.013000 | 0.000230 | 0.001799 | 0.010576 | 0.090000 | 0.579046 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_winner_move_mean_k64_p2_top090_a35.csv |
| 5 | analog_transport_20260530_mean_k64_p2_cvblend_a35.csv | physics_blend | 0.395200 | 0.015487 | 0.008457 | 0.017957 | 0.081083 | 1.000000 | 0.394964 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_mean_k64_p2_cvblend_a35.csv |
| 6 | analog_transport_20260530_trimmean_k128_p15_cablend_a30.csv | physics_blend | 0.384200 | 0.016492 | 0.010292 | 0.022433 | 0.117333 | 1.000000 | 0.383728 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_trimmean_k128_p15_cablend_a30.csv |
| 7 | analog_transport_20260530_median_k96_p2_cvblend_a45.csv | physics_blend | 0.375900 | 0.016402 | 0.009565 | 0.022642 | 0.081492 | 1.000000 | 0.375482 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_median_k96_p2_cvblend_a45.csv |
| 8 | analog_transport_20260530_median_k96_p2.csv | analog_only | 0.217800 | 0.023922 | 0.018152 | 0.047050 | 0.091545 | 1.000000 | 0.215407 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_median_k96_p2.csv |
| 9 | analog_transport_20260530_trimmean_k128_p15.csv | analog_only | 0.180900 | 0.024760 | 0.019185 | 0.044792 | 0.075250 | 1.000000 | 0.178469 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_trimmean_k128_p15.csv |
| 10 | analog_transport_20260530_mean_k64_p2.csv | analog_only | 0.180600 | 0.024713 | 0.019146 | 0.044633 | 0.075163 | 1.000000 | 0.178176 | C:\open\dacon-mosquito-trajectory-prediction\submissions\analog_transport_20260530_mean_k64_p2.csv |

## Decision Rule

- Pure analog candidates are risky if their OOF hit is below the current physics baseline.
- If winner-selective probes drop, analog transport does not transfer to public and should be abandoned.
