# DACON Mosquito Trajectory Prediction

LiDAR sensor-local 3D coordinates from the past 400ms are used to predict a mosquito's position 80ms after the last observation.

## Competition

- Name: 모기 비행 궤적 예측 AI 경진대회
- Platform: DACON
- URL: https://dacon.io/competitions/official/236716/overview/description
- Task: 3D future coordinate regression
- Metric: R-Hit@1cm
- Daily submissions: 5
- Period: 2026-05-04 to 2026-06-01

## Problem Summary

Each sample contains 11 observations at 40ms intervals:

```text
Input  : -400ms, -360ms, ..., -40ms, 0ms
Target : +80ms position
```

Coordinates are sensor-local 3D positions:

- `x`: forward
- `y`: left
- `z`: up
- Unit: meter

The target is not just low average distance error. The final score is the hit rate: a prediction is counted as correct if the 3D Euclidean distance from the true future point is at most `0.01m`.

## Data Structure

Expected raw data layout after downloading `open.zip`:

```text
data/raw/
├── train/
│   ├── TRAIN_00001.csv
│   └── ...
├── test/
│   ├── TEST_00001.csv
│   └── ...
├── train_labels.csv
└── sample_submission.csv
```

Each train/test CSV has:

```text
timestep_ms, x, y, z
```

`train_labels.csv` and `sample_submission.csv` use:

```text
id, x, y, z
```

## Initial Modeling Direction

This is a short-horizon physics/trajectory prediction problem. The first experiments should prioritize strong deterministic baselines before heavy neural models.

Recommended first axis:

1. Constant velocity extrapolation
2. Constant acceleration extrapolation
3. Polynomial / Savitzky-Golay style local smoothing
4. Robust outlier handling for noisy LiDAR observations
5. Model residuals over physics baselines using LightGBM/CatBoost
6. Sequence model only after physics baselines are understood

## Project Structure

```text
dacon-mosquito-trajectory-prediction/
├── data/                  # Raw/processed data, ignored by git
├── docs/                  # Handoff notes and research summaries
├── experiments/           # Experiment logs
├── notebooks/             # EDA notebooks
├── scripts/               # Training, validation, submission scripts
├── src/                   # Reusable modules
├── submissions/           # Submission files, ignored by git
└── README.md
```

## Validation

```bash
python scripts/validate_submission.py submissions/example.csv
```

## First Baselines

After placing the official competition data under `data/raw`, run:

```bash
python scripts/run_physics_baselines.py
```

This evaluates constant position, constant velocity, constant acceleration, and polynomial extrapolation baselines on a train holdout, then writes the best validation method as a submission file under `submissions/`.

To run the end-to-end local loop, including baseline search, threshold-aware parameter search, submission validation, and Markdown result reports:

```bash
python scripts/auto_pipeline.py
```

For a smoke test on a small slice:

```bash
python scripts/run_physics_baselines.py --limit-train 100 --limit-test 100 --skip-submission
```

To commit and push code/reports after a successful run:

```bash
python scripts/auto_pipeline.py --publish-github
```

To search threshold-aware physics parameters after the first baseline:

```bash
python scripts/search_physics_params.py
```

To run the more aggressive multi-seed physics CV and residual LightGBM experiments:

```bash
python scripts/run_aggressive_experiments.py
```

To train a sample-wise selector that chooses among physics candidates:

```bash
python scripts/run_candidate_selector.py
```

## Notes

The raw data and submission files are excluded from GitHub. The repository is intended to track reproducible code, experiment logs, and competition strategy.
