# Physics Baseline Run

- Created at: `2026-05-05T23:36:35`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Validation size: `2000`
- Best method: `velocity_last`
- Best R-Hit@1cm: `0.608500`
- Best mean distance: `0.012726` m
- Submission: `C:\open\dacon-mosquito-trajectory-prediction\submissions\physics_baseline_velocity_last.csv`

## Leaderboard

| method | mean_distance | median_distance | p90_distance | p95_distance | r_hit_1cm |
| --- | --- | --- | --- | --- | --- |
| velocity_last | 0.012726 | 0.007929 | 0.028403 | 0.048451 | 0.608500 |
| acceleration_last | 0.016237 | 0.009806 | 0.037306 | 0.056637 | 0.511500 |
| poly2_recent5 | 0.019611 | 0.011433 | 0.049564 | 0.072147 | 0.450000 |
| velocity_fit_recent5 | 0.021777 | 0.013362 | 0.053006 | 0.075716 | 0.381000 |
| poly2_weighted | 0.027528 | 0.017307 | 0.063489 | 0.088849 | 0.271500 |
| poly2_all | 0.032216 | 0.021096 | 0.074917 | 0.102293 | 0.219000 |
| velocity_fit_weighted | 0.034658 | 0.026081 | 0.075948 | 0.096008 | 0.172500 |
| velocity_fit_all | 0.041984 | 0.031646 | 0.091014 | 0.116207 | 0.132500 |
| position | 0.050991 | 0.046097 | 0.104273 | 0.107457 | 0.004000 |

## Data Summary

### Train

```json
{
  "sample_count": 10000,
  "sequence_length_min": 11,
  "sequence_length_max": 11,
  "first_timestep_ms_values": [
    -400.0
  ],
  "last_timestep_ms_values": [
    0.0
  ],
  "coord_min": {
    "x": 0.495896,
    "y": -2.588907,
    "z": -1.679616
  },
  "coord_max": {
    "x": 6.87556,
    "y": 2.587411,
    "z": 2.607437
  },
  "step_distance_mean": 0.02466580716580039,
  "step_distance_p50": 0.022903374852957535,
  "step_distance_p90": 0.046402362311741746,
  "step_distance_p99": 0.05392645295696722
}
```

### Test

```json
{
  "sample_count": 10000,
  "sequence_length_min": 11,
  "sequence_length_max": 11,
  "first_timestep_ms_values": [
    -400.0
  ],
  "last_timestep_ms_values": [
    0.0
  ],
  "coord_min": {
    "x": 0.463376,
    "y": -2.707029,
    "z": -1.650929
  },
  "coord_max": {
    "x": 6.394771,
    "y": 2.647056,
    "z": 2.545505
  },
  "step_distance_mean": 0.023435558239741517,
  "step_distance_p50": 0.021704528811761225,
  "step_distance_p90": 0.04222338830935591,
  "step_distance_p99": 0.05391853849419506
}
```

## Readout

- The holdout favors linear short-horizon motion. Next experiments should focus on robust velocity estimates and recent-window blending.
- Because the official metric is thresholded at 1cm, compare methods by `r_hit_1cm` first and use mean distance only as a tie-breaker.
