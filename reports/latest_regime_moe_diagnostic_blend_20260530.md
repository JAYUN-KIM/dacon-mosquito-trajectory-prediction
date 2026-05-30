# 2026-05-30 Regime MoE Diagnostic Blend

- winner: `recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv`
- moe: `regime_moe_k5_a05_oof0.5544.csv`
- pure_moe_oof_hit: `0.5544`
- note: pure MoE is much weaker than winner, so only tiny diagnostic blends are recommended.

## Outputs

| submission | blend | mean_move_m | median_move_m | p95_move_m | max_move_m |
|---|---:|---:|---:|---:|---:|
| `regimemoe30_diag_blend03_k5a05.csv` | 0.03 | 0.000180 | 0.000137 | 0.000472 | 0.001681 |
| `regimemoe30_diag_blend05_k5a05.csv` | 0.05 | 0.000301 | 0.000228 | 0.000786 | 0.002801 |
| `regimemoe30_diag_blend08_k5a05.csv` | 0.08 | 0.000481 | 0.000364 | 0.001258 | 0.004482 |
| `regimemoe30_diag_blend12_k5a05.csv` | 0.12 | 0.000721 | 0.000546 | 0.001887 | 0.006723 |

## Recommended Use

- Do not submit pure `regime_moe_*` files.
- If testing this external-AI axis, submit only `regimemoe30_diag_blend03_k5a05.csv` or `blend05` as a low-risk probe.
- If the probe drops below 0.692, abandon Regime MoE/Ridge clustering and move on.
