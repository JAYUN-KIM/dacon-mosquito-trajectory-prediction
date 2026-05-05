# Candidate Selector

- Created at: `2026-05-05T23:54:11`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Candidates: `['v100_a000', 'v100_a100', 'v100_a150', 'v100_a200', 'v100_a250', 'v100_a275', 'v100_a300', 'v098_a175', 'v098_a200', 'v098_a225', 'v102_a200', 'v102_a250']`
- CV seeds: `[42, 777, 2026, 3407, 10007]`
- Submission: `C:\open\dacon-mosquito-trajectory-prediction\submissions\candidate_binary_hit_selector.csv`

## Single Candidates On Full Train

| candidate | velocity_scale | accel_scale | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v100_a300 | 1.000000 | 0.300000 | 0.012977 | 0.008158 | 0.027343 | 0.044839 | 0.600800 |
| v100_a275 | 1.000000 | 0.275000 | 0.012933 | 0.008148 | 0.027124 | 0.044708 | 0.600300 |
| v100_a250 | 1.000000 | 0.250000 | 0.012895 | 0.008124 | 0.026858 | 0.044660 | 0.599300 |
| v100_a200 | 1.000000 | 0.200000 | 0.012842 | 0.008143 | 0.026787 | 0.044402 | 0.597600 |
| v098_a225 | 0.980000 | 0.225000 | 0.012799 | 0.008175 | 0.026363 | 0.043937 | 0.597000 |
| v098_a200 | 0.980000 | 0.200000 | 0.012781 | 0.008188 | 0.026280 | 0.043938 | 0.596600 |
| v100_a150 | 1.000000 | 0.150000 | 0.012818 | 0.008158 | 0.026684 | 0.044505 | 0.596200 |
| v098_a175 | 0.980000 | 0.175000 | 0.012770 | 0.008219 | 0.026288 | 0.043860 | 0.594500 |
| v102_a200 | 1.020000 | 0.200000 | 0.013040 | 0.008230 | 0.027397 | 0.044997 | 0.592800 |
| v102_a250 | 1.020000 | 0.250000 | 0.013102 | 0.008248 | 0.027434 | 0.045232 | 0.592400 |
| v100_a100 | 1.000000 | 0.100000 | 0.012827 | 0.008214 | 0.026655 | 0.044375 | 0.592000 |
| v100_a000 | 1.000000 | 0.000000 | 0.012941 | 0.008325 | 0.026508 | 0.044800 | 0.578800 |

## Selector CV

| selector | detail | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_distance | upper_bound | 0.642200 | 0.021247 | 0.618500 | 0.012027 | 0.007332 |
| oracle_hit | upper_bound | 0.642200 | 0.021247 | 0.618500 | 0.012449 | 0.008312 |
| binary_hit_probability | max_predicted_hit_probability | 0.604700 | 0.020978 | 0.579500 | 0.012973 | 0.008074 |
| multiclass_best_distance | argmin_distance_label | 0.604700 | 0.020972 | 0.578500 | 0.012735 | 0.008023 |

## Readout

- `oracle_hit` is the upper bound if we could always choose a hitting candidate from the set.
- If `binary_hit_probability` beats the best single candidate in CV, this is a good aggressive public submission candidate.
- If selector CV underperforms, the candidate set has signal but the selector features are not predictive enough yet.
