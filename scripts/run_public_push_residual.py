from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in [SRC, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mosquito_trajectory.data import (
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, make_features, physics_prediction, split_mask, stack_samples


warnings.filterwarnings("ignore", message="X does not have valid feature names")


CV_SEEDS = [42, 777, 2026, 3407, 10007]
FULL_SEEDS = [42, 777, 2026, 3407, 10007]
SHRINKS = [0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45]


def make_lgbm_wide(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=560,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=18,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.92,
        reg_alpha=0.02,
        reg_lambda=0.20,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_axes(train_x: np.ndarray, train_y: np.ndarray, pred_x: np.ndarray, seed: int) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = make_lgbm_wide(seed + axis)
        model.fit(train_x, train_y[:, axis])
        preds.append(model.predict(pred_x))
    return np.vstack(preds).T


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    base = physics_prediction(coords, velocity_scale=1.0, accel_scale=0.275)
    residual = y - base
    rows = []
    for seed in CV_SEEDS:
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        pred_residual = fit_predict_axes(features[train_mask], residual[train_mask], features[val_mask], seed)
        y_val = y[val_mask]
        base_val = base[val_mask]
        for shrink in SHRINKS:
            pred = base_val + shrink * pred_residual
            rows.append({"seed": seed, "shrink": shrink, **distance_summary(pred, y_val)})
    df = pd.DataFrame(rows)
    return (
        df.groupby("shrink", as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )


def full_residual_prediction(features: np.ndarray, residual: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    return np.mean([fit_predict_axes(features, residual, test_features, seed) for seed in FULL_SEEDS], axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create public-push residual variants around the best public LGBM family.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_public_push_residual.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}")

    train_samples = read_trajectory_folder(data_dir / "train")
    test_samples = read_trajectory_folder(data_dir / "test")
    targets = read_targets(data_dir / "train_labels.csv")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")

    ids, missing = aligned_ids(train_samples, targets[ID_COLUMN].tolist())
    if missing:
        raise ValueError(f"{len(missing)} train ids are missing trajectory files, examples: {missing[:5]}")

    train_coords = stack_samples(train_samples, ids)
    test_coords = stack_samples(test_samples, sample_submission[ID_COLUMN].tolist())
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)

    print("Building features")
    train_features = make_features(train_coords)
    test_features = make_features(test_coords)

    print("Evaluating 5-seed shrink CV")
    leaderboard = evaluate_cv(train_coords, y, train_features, args.val_frac)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    base_train = physics_prediction(train_coords, velocity_scale=1.0, accel_scale=0.275)
    base_test = physics_prediction(test_coords, velocity_scale=1.0, accel_scale=0.275)
    residual = y - base_train
    print("Training 5-seed full ensemble")
    residual_test = full_residual_prediction(train_features, residual, test_features)

    written = []
    for shrink in [0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45]:
        pred = base_test + shrink * residual_test
        path = args.output_dir / f"public_push_lgbm_wide_a0275_s{shrink:.2f}_5seed.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Public Push Residual",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full ensemble seeds: `{FULL_SEEDS}`",
        "- Model: `lgbm_wide`, base `velocity_scale=1.0`, `accel_scale=0.275`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Shrink CV",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Readout",
        "",
        "- This is a public-oriented refinement around the family that improved the leaderboard to 0.6348.",
        "- The 5-seed full ensemble should reduce model variance compared with the previous 3-seed submissions.",
        "- Try nearby shrink variants one at a time; they differ by sub-millimeter to millimeter shifts but that matters for R-Hit@1cm.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}")
    for path in written:
        print(f"Wrote submission: {path}")


if __name__ == "__main__":
    main()
