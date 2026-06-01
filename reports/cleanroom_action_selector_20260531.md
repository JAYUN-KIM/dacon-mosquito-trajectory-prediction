# cleanroom_action_selector_20260531

- created_at: 2026-05-31 16:08:24
- train_shape: `(10000, 11, 3)`
- test_shape: `(10000, 11, 3)`
- raw_action_count: `103`
- rule: old submissions/champion/cache files were not read

## Core Diagnosis

The clean-room raw physics pool has a much higher oracle hit rate than any single action. This points to action selection as the current bottleneck, especially around acceleration/turning regimes.

- best single raw action: `cv_acc0.5` / OOF hit `0.599300`
- raw action oracle hit: `0.776700`
- oracle gap over best single: `0.177400`

## OOF Submission Candidates

| rank | file | candidate_key | oof_hit | oof_mean_dist |
| --- | --- | --- | --- | --- |
| 1 | cleanroom31_rank1_soft_p3_blend90.csv | soft_p3_blend90 | 0.624900 | 0.012357 |
| 2 | cleanroom31_rank2_soft_p3_blend75.csv | soft_p3_blend75 | 0.624800 | 0.012308 |
| 3 | cleanroom31_rank3_soft_p3.csv | soft_p3 | 0.624400 | 0.012418 |
| 4 | cleanroom31_rank4_soft_p2_blend90.csv | soft_p2_blend90 | 0.622700 | 0.012252 |
| 5 | cleanroom31_rank5_soft_p2.csv | soft_p2 | 0.622700 | 0.012266 |
| 6 | cleanroom31_rank6_soft_p3_blend55.csv | soft_p3_blend55 | 0.622000 | 0.012322 |
| 7 | cleanroom31_rank7_soft_p2_blend75.csv | soft_p2_blend75 | 0.621900 | 0.012262 |
| 8 | cleanroom31_rank8_soft_p2_blend55.csv | soft_p2_blend55 | 0.619100 | 0.012334 |
| 9 | cleanroom31_rank9_soft_p1.csv | soft_p1 | 0.615700 | 0.012327 |
| 10 | cleanroom31_rank10_soft_p2_blend35.csv | soft_p2_blend35 | 0.614200 | 0.012474 |
| 11 | cleanroom31_rank11_single_best_raw.csv | single_best_raw | 0.599300 | 0.012895 |
| 12 | cleanroom31_rank12_hard_argmax.csv | hard_argmax | 0.594800 | 0.013984 |

## Fold Diagnostics

| fold | argmin_action_accuracy |
| --- | --- |
| 1.000000 | 0.179000 |
| 2.000000 | 0.185500 |
| 3.000000 | 0.184500 |
| 4.000000 | 0.186500 |
| 5.000000 | 0.189000 |

## Strongest Single Raw Actions

| action | hit | chosen_oof_count |
| --- | --- | --- |
| cv_acc0.5 | 0.599300 | 495 |
| cv_acc1.0 | 0.582600 | 243 |
| vf2.05 | 0.581700 | 72 |
| aniso_f2.02s0.00u0.50 | 0.581000 | 0 |
| aniso_f2.02s0.00u0.00 | 0.581000 | 44 |
| aniso_f2.02s1.00u1.00 | 0.581000 | 0 |
| aniso_f2.02s1.00u1.25 | 0.581000 | 0 |
| aniso_f2.02s1.00u0.00 | 0.581000 | 0 |
| aniso_f2.02s1.00u0.50 | 0.581000 | 0 |
| aniso_f2.02s0.35u1.00 | 0.581000 | 0 |
| aniso_f2.02s0.35u1.25 | 0.581000 | 0 |
| aniso_f2.02s0.70u0.00 | 0.581000 | 0 |
| aniso_f2.02s0.70u0.50 | 0.581000 | 0 |
| aniso_f2.02s0.70u1.00 | 0.581000 | 0 |
| aniso_f2.02s0.70u1.25 | 0.581000 | 0 |

## Interpretation

If the top soft-action files improve public LB, the next step is to expand the action pool and improve probability calibration. If they drop, the failure mode is likely OOF-public selector transfer rather than absence of useful raw physics actions.

## Public Result

- `cleanroom31_rank1_soft_p3_blend90.csv`: `0.65140`
- Final decision: abandon this clean-room raw action selector as a submission family.

The OOF signal did not transfer to the public split. The result also shows that the previous `0.69200` champion family was not just a trivial raw physics extrapolation; the accumulated temporal/gate calibration was materially important.
