# 2026-05-20 MLP Sequence Pseudo Blend

- created_at: `2026-05-20T22:30:03`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- current_best: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- public_feedback: `mlpseq_rank2_blend08base.csv = 0.69020`
- idea: leave tree/physics correction family and train a small neural MLP on normalized sequence features with temporal pseudo-supervision.
- pseudo_cutoffs: `(6, 7, 8)`
- pseudo_weight: `0.16`
- seeds: `[43, 777, 2026]`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mlpseq_rank1_blend12base.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mlpseq_rank2_blend08base.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mlpseq_rank3_blend16base.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mlpseq_rank4_blend20cap0012base.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mlpseq_rank5_blend12tilt104096.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mlpseq_rank6_blend20cap0012tilt104096.csv']`

## Seed Diagnostics

| seed | n_iter | best_validation_score | loss |
| --- | --- | --- | --- |
| 43.000000 | 31.000000 | 0.170258 | 0.090035 |
| 777.000000 | 32.000000 | 0.308557 | 0.101893 |
| 2026.000000 | 25.000000 | 0.250100 | 0.114872 |

## Outputs

| rank | submission | spec | mult | blend | cap | mlp_vs_current_best_mean_delta | mlp_vs_current_best_median_delta | mlp_vs_current_best_p95_delta | mlp_vs_current_best_max_delta | submission_vs_current_best_mean_delta | submission_vs_current_best_median_delta | submission_vs_current_best_p95_delta | submission_vs_current_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mlpseq_rank1_blend12base.csv | blend12_base | f102_s100_u100 | 0.120000 | none | 0.004937 | 0.003484 | 0.013889 | 0.093252 | 0.000592 | 0.000418 | 0.001667 | 0.011190 |
| 2 | mlpseq_rank2_blend08base.csv | blend08_base | f102_s100_u100 | 0.080000 | none | 0.004937 | 0.003484 | 0.013889 | 0.093252 | 0.000395 | 0.000279 | 0.001111 | 0.007460 |
| 3 | mlpseq_rank3_blend16base.csv | blend16_base | f102_s100_u100 | 0.160000 | none | 0.004937 | 0.003484 | 0.013889 | 0.093252 | 0.000790 | 0.000557 | 0.002222 | 0.014920 |
| 4 | mlpseq_rank4_blend20cap0012base.csv | blend20cap0012_base | f102_s100_u100 | 0.200000 | 0.001200 | 0.004937 | 0.003484 | 0.013889 | 0.093252 | 0.000729 | 0.000697 | 0.001200 | 0.001200 |
| 5 | mlpseq_rank5_blend12tilt104096.csv | blend12_tilt104096 | f102_s104_u096 | 0.120000 | none | 0.004951 | 0.003493 | 0.013967 | 0.093298 | 0.000594 | 0.000419 | 0.001676 | 0.011196 |
| 6 | mlpseq_rank6_blend20cap0012tilt104096.csv | blend20cap0012_tilt104096 | f102_s104_u096 | 0.200000 | 0.001200 | 0.004951 | 0.003493 | 0.013967 | 0.093298 | 0.000731 | 0.000699 | 0.001200 | 0.001200 |

## Notes

- This is intentionally a new model-family axis after mirror and multi-curvature failed on public.
- The outputs are small blends or capped moves from the current best to control public risk.
- If even the 8-12% blends drop, this neural bias is not complementary enough and should be deprioritized.
