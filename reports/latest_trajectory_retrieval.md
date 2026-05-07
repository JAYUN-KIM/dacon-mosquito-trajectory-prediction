# Trajectory Retrieval

- Created at: `2026-05-07T09:49:01`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Feature variants: `['local_shape', 'local_motion', 'global_motion']`
- Target modes: `['local_residual', 'local_offset', 'global_residual']`
- K values: `[1, 3, 5, 8, 12, 20, 32, 50]`
- Weight modes: `['uniform', 'inverse', 'softmax0.75', 'softmax1.25', 'rank0.85']`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retrieval_rank1_localmotion_localresidual_k50_inverse.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retrieval_rank2_localmotion_localresidual_k50_softmax075.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retrieval_rank3_localmotion_localresidual_k50_softmax125.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retrieval_rank4_localmotion_localresidual_k50_uniform.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\retrieval_rank5_localmotion_localresidual_k32_softmax075.csv']`

## Top 30 Retrieval Configs

| feature_variant | target_mode | k | weight_mode | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_motion | local_residual | 50 | inverse | 0.601900 | 0.019272 | 0.572500 | 0.012717 | 0.008096 | 0.597082 |
| local_motion | local_residual | 50 | softmax0.75 | 0.601800 | 0.018319 | 0.574000 | 0.012712 | 0.008083 | 0.597220 |
| local_motion | local_residual | 50 | softmax1.25 | 0.601500 | 0.019818 | 0.571000 | 0.012726 | 0.008099 | 0.596546 |
| local_motion | local_residual | 50 | uniform | 0.601400 | 0.019967 | 0.571500 | 0.012734 | 0.008103 | 0.596408 |
| local_motion | local_residual | 32 | softmax0.75 | 0.599300 | 0.018734 | 0.572500 | 0.012763 | 0.008192 | 0.594617 |
| local_motion | local_residual | 32 | softmax1.25 | 0.599100 | 0.019705 | 0.570500 | 0.012775 | 0.008213 | 0.594174 |
| local_motion | local_residual | 32 | inverse | 0.599000 | 0.018931 | 0.571000 | 0.012766 | 0.008206 | 0.594267 |
| local_motion | local_residual | 32 | uniform | 0.598600 | 0.019272 | 0.571000 | 0.012783 | 0.008217 | 0.593782 |
| local_motion | global_residual | 50 | softmax0.75 | 0.594500 | 0.018970 | 0.568000 | 0.013210 | 0.008227 | 0.589757 |
| local_motion | global_residual | 50 | inverse | 0.594400 | 0.019116 | 0.568500 | 0.013211 | 0.008222 | 0.589621 |
| local_motion | global_residual | 50 | softmax1.25 | 0.594100 | 0.019696 | 0.567000 | 0.013216 | 0.008211 | 0.589176 |
| local_motion | global_residual | 50 | uniform | 0.593800 | 0.020960 | 0.564500 | 0.013220 | 0.008219 | 0.588560 |
| local_motion | global_residual | 32 | softmax0.75 | 0.591100 | 0.016536 | 0.571000 | 0.013293 | 0.008284 | 0.586966 |
| local_motion | local_residual | 20 | softmax0.75 | 0.591000 | 0.019551 | 0.566500 | 0.012904 | 0.008258 | 0.586112 |
| local_motion | global_residual | 32 | inverse | 0.590400 | 0.016802 | 0.569000 | 0.013293 | 0.008278 | 0.586200 |
| local_motion | global_residual | 32 | uniform | 0.590200 | 0.017605 | 0.568500 | 0.013300 | 0.008283 | 0.585799 |
| local_motion | global_residual | 32 | softmax1.25 | 0.590100 | 0.016850 | 0.569500 | 0.013297 | 0.008277 | 0.585887 |
| local_motion | local_residual | 20 | inverse | 0.590000 | 0.019532 | 0.566000 | 0.012908 | 0.008257 | 0.585117 |
| local_motion | local_residual | 20 | softmax1.25 | 0.589000 | 0.019478 | 0.564000 | 0.012916 | 0.008264 | 0.584131 |
| local_motion | local_residual | 20 | uniform | 0.588900 | 0.019165 | 0.564000 | 0.012925 | 0.008288 | 0.584109 |
| local_motion | local_residual | 50 | rank0.85 | 0.588700 | 0.014580 | 0.567500 | 0.013039 | 0.008324 | 0.585055 |
| local_shape | global_residual | 50 | inverse | 0.588600 | 0.017210 | 0.570500 | 0.013376 | 0.008386 | 0.584298 |
| local_shape | global_residual | 50 | softmax1.25 | 0.588300 | 0.017524 | 0.570500 | 0.013379 | 0.008378 | 0.583919 |
| local_motion | local_residual | 32 | rank0.85 | 0.588200 | 0.014481 | 0.567500 | 0.013045 | 0.008337 | 0.584580 |
| local_shape | global_residual | 50 | uniform | 0.587800 | 0.017838 | 0.569500 | 0.013381 | 0.008388 | 0.583340 |
| global_motion | local_residual | 50 | softmax1.25 | 0.587600 | 0.015702 | 0.572500 | 0.012897 | 0.008383 | 0.583675 |
| global_motion | local_residual | 50 | uniform | 0.587300 | 0.016769 | 0.571500 | 0.012905 | 0.008390 | 0.583108 |
| local_shape | global_residual | 50 | softmax0.75 | 0.587200 | 0.017046 | 0.570500 | 0.013377 | 0.008390 | 0.582938 |
| global_motion | local_residual | 50 | inverse | 0.587000 | 0.015957 | 0.571500 | 0.012888 | 0.008367 | 0.583011 |
| local_motion | local_residual | 20 | rank0.85 | 0.586400 | 0.014800 | 0.564500 | 0.013086 | 0.008348 | 0.582700 |

## Readout

- This is a new nonparametric direction: retrieve similar past trajectories and aggregate their future residual/offset.
- It is intentionally different from the LightGBM local-frame residual family, so it can be useful even if standalone public score is lower.
- If standalone score is close to the current best, the next step is blend or sample-wise routing between retrieval and local-frame residual.
