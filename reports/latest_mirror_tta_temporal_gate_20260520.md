# 2026-05-20 Mirror-Symmetry Temporal TTA Gate

- created_at: `2026-05-20T22:02:01`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- current_best: `curvgate_refine_rank2_gatet52a105.csv = 0.69120`
- public_feedback: `mirror_tta_rank1_mirrortta_temporal_w55_gate_t52_a105_bestblend35.csv = 0.69020`
- idea: exploit left-right symmetry by augmenting train with `y -> -y` and averaging original/mirrored test predictions after unflipping.
- temporal_spec: `tb_c678_w020`
- temporal_mult: `(1.02, 1.0, 1.0)`
- gate: `threshold=0.52`, `alpha=0.105`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mirror_tta_rank1_mirrortta_temporal_w55_gate_t52_a105_bestblend35.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mirror_tta_rank2_mirrortta_temporal_w55_gate_t52_a105_bestblend50.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mirror_tta_rank3_mirrortta_temporal_w50_gate_t52_a105_bestblend35.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mirror_tta_rank4_mirrortta_temporal_w60_gate_t52_a105_bestblend35.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mirror_tta_rank5_mirrortta_temporal_w55_gate_t52_a105.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\mirror_tta_rank6_mirrorraw_temporal_w55_gate_t52_a105_bestblend35.csv']`

## Seed Diagnostics

| seed | raw_vs_tta_mean_delta | raw_vs_tta_p95_delta |
| --- | --- | --- |
| 42.000000 | 0.000219 | 0.000698 |
| 777.000000 | 0.000211 | 0.000685 |
| 2026.000000 | 0.000217 | 0.000685 |

## Outputs

| rank | submission | name | kind | route_fraction | vs_current_best_mean_delta | vs_current_best_median_delta | vs_current_best_p95_delta | vs_current_best_max_delta | vs_mirror_tta_temporal_mean_delta | vs_mirror_tta_temporal_median_delta | vs_mirror_tta_temporal_p95_delta | vs_mirror_tta_temporal_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mirror_tta_rank1_mirrortta_temporal_w55_gate_t52_a105_bestblend35.csv | mirrortta_temporal_w55_gate_t52_a105_bestblend35 | blend_with_current_best | 0.569200 | 0.000221 | 0.000138 | 0.000664 | 0.010969 | 0.001250 | 0.000769 | 0.003961 | 0.036395 |
| 2 | mirror_tta_rank2_mirrortta_temporal_w55_gate_t52_a105_bestblend50.csv | mirrortta_temporal_w55_gate_t52_a105_bestblend50 | blend_with_current_best | 0.569200 | 0.000316 | 0.000197 | 0.000948 | 0.015670 | 0.001187 | 0.000727 | 0.003764 | 0.033039 |
| 3 | mirror_tta_rank3_mirrortta_temporal_w50_gate_t52_a105_bestblend35.csv | mirrortta_temporal_w50_gate_t52_a105_bestblend35 | blend_with_current_best | 0.569100 | 0.000207 | 0.000129 | 0.000622 | 0.010958 | 0.001282 | 0.000787 | 0.004038 | 0.037202 |
| 4 | mirror_tta_rank4_mirrortta_temporal_w60_gate_t52_a105_bestblend35.csv | mirrortta_temporal_w60_gate_t52_a105_bestblend35 | blend_with_current_best | 0.569100 | 0.000240 | 0.000150 | 0.000723 | 0.010982 | 0.001218 | 0.000748 | 0.003869 | 0.035587 |
| 5 | mirror_tta_rank5_mirrortta_temporal_w55_gate_t52_a105.csv | mirrortta_temporal_w55_gate_t52_a105 | pure_mirror_axis | 0.569200 | 0.000632 | 0.000394 | 0.001897 | 0.031341 | 0.001015 | 0.000622 | 0.003306 | 0.021857 |
| 6 | mirror_tta_rank6_mirrorraw_temporal_w55_gate_t52_a105_bestblend35.csv | mirrorraw_temporal_w55_gate_t52_a105_bestblend35 | blend_with_current_best | 0.569100 | 0.000223 | 0.000139 | 0.000665 | 0.010803 | 0.001251 | 0.000770 | 0.003972 | 0.035671 |

## Notes

- This is intentionally a new axis, not a threshold micro-tune.
- If a blended candidate improves public, mirror-symmetry augmentation is useful but should be regularized.
- If only the pure candidate improves, the temporal model itself benefits strongly from symmetry TTA.
- If all candidates drop, the sensor-local left/right axis is not symmetric enough for this dataset.
