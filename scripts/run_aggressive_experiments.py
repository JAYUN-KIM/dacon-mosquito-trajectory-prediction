from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mosquito_trajectory.data import (
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)


SEEDS = [42, 777, 2026, 3407, 10007]
SHRINKS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
PHYSICS_ACCEL_SCALES = np.round(np.arange(-0.05, 0.351, 0.025), 3)
PHYSICS_VELOCITY_SCALES = np.round(np.arange(0.94, 1.061, 0.02), 3)


def r_hit(pred: np.ndarray, true: np.ndarray, radius: float = 0.01) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=1) <= radius))


def distance_summary(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(pred - true, axis=1)
    return {
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "p90_distance": float(np.quantile(distances, 0.90)),
        "p95_distance": float(np.quantile(distances, 0.95)),
        "r_hit_1cm": float(np.mean(distances <= 0.01)),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def split_mask(n_rows: int, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    val_size = max(1, int(round(n_rows * val_frac)))
    mask = np.zeros(n_rows, dtype=bool)
    mask[indices[:val_size]] = True
    return mask


def stack_samples(samples: dict, ids: list[str]) -> np.ndarray:
    return np.stack([samples[sample_id].coords for sample_id in ids]).astype(np.float64)


def physics_prediction(coords: np.ndarray, velocity_scale: float = 1.0, accel_scale: float = 0.15) -> np.ndarray:
    last = coords[:, -1, :]
    d_last = coords[:, -1, :] - coords[:, -2, :]
    d_prev = coords[:, -2, :] - coords[:, -3, :]
    return last + 2.0 * d_last * velocity_scale + 2.0 * accel_scale * (d_last - d_prev)


def make_features(coords: np.ndarray) -> np.ndarray:
    last = coords[:, -1, :]
    rel_coords = coords - last[:, None, :]
    d = np.diff(coords, axis=1)
    dd = np.diff(d, axis=1)
    ddd = np.diff(dd, axis=1)

    speed = np.linalg.norm(d, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    jerk = np.linalg.norm(ddd, axis=2)

    d_last = d[:, -1, :]
    d_prev = d[:, -2, :]
    dd_last = dd[:, -1, :]
    dd_prev = dd[:, -2, :]

    dot = np.sum(d_last * d_prev, axis=1, keepdims=True)
    denom = (np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True)) + 1e-12
    cos_turn = dot / denom

    physics_000 = physics_prediction(coords, accel_scale=0.0)
    physics_015 = physics_prediction(coords, accel_scale=0.15)
    physics_020 = physics_prediction(coords, accel_scale=0.20)

    feature_blocks = [
        last,
        rel_coords.reshape(coords.shape[0], -1),
        d.reshape(coords.shape[0], -1),
        dd.reshape(coords.shape[0], -1),
        ddd.reshape(coords.shape[0], -1),
        speed,
        accel,
        jerk,
        d_last,
        d_prev,
        dd_last,
        dd_prev,
        cos_turn,
        physics_000,
        physics_015,
        physics_020,
        physics_015 - last,
        physics_015 - physics_000,
    ]
    return np.hstack(feature_blocks).astype(np.float32)


def evaluate_physics_cv(coords: np.ndarray, target: np.ndarray, val_frac: float) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        val_mask = split_mask(len(coords), val_frac, seed)
        y_val = target[val_mask]
        x_val = coords[val_mask]
        for velocity_scale in PHYSICS_VELOCITY_SCALES:
            for accel_scale in PHYSICS_ACCEL_SCALES:
                pred = physics_prediction(x_val, float(velocity_scale), float(accel_scale))
                rows.append(
                    {
                        "family": "physics_grid",
                        "seed": seed,
                        "velocity_scale": float(velocity_scale),
                        "accel_scale": float(accel_scale),
                        **distance_summary(pred, y_val),
                    }
                )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["family", "velocity_scale", "accel_scale"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    return grouped


def lgbm_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.03,
        reg_lambda=0.30,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_residual(train_x: np.ndarray, train_residual: np.ndarray, val_x: np.ndarray, seed: int) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = lgbm_model(seed + axis)
        model.fit(train_x, train_residual[:, axis])
        preds.append(model.predict(val_x))
    return np.vstack(preds).T


def evaluate_lgbm_cv(coords: np.ndarray, target: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    base = physics_prediction(coords, accel_scale=0.15)
    residual = target - base
    rows = []

    for seed in SEEDS:
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        residual_pred = fit_predict_residual(
            features[train_mask],
            residual[train_mask],
            features[val_mask],
            seed,
        )
        y_val = target[val_mask]
        base_val = base[val_mask]
        for shrink in SHRINKS:
            pred = base_val + float(shrink) * residual_pred
            rows.append(
                {
                    "family": "lgbm_residual",
                    "seed": seed,
                    "shrink": float(shrink),
                    **distance_summary(pred, y_val),
                }
            )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["family", "shrink"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    return grouped


def train_full_lgbm_submission(
    train_features: np.ndarray,
    train_residual: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    axis_preds = []
    for axis in range(3):
        model = lgbm_model(seed + axis)
        model.fit(train_features, train_residual[:, axis])
        axis_preds.append(model.predict(test_features))
    return np.vstack(axis_preds).T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggressive physics and residual-ML experiments.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_aggressive_experiments.md")
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
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)
    test_ids = sample_submission[ID_COLUMN].tolist()
    test_coords = stack_samples(test_samples, test_ids)

    print("Evaluating multi-seed physics grid")
    physics_cv = evaluate_physics_cv(train_coords, y, args.val_frac)
    print(physics_cv.head(15).to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\nBuilding residual features")
    train_features = make_features(train_coords)
    test_features = make_features(test_coords)

    print("Evaluating LightGBM residual shrink CV")
    lgbm_cv = evaluate_lgbm_cv(train_coords, y, train_features, args.val_frac)
    print(lgbm_cv.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    best_physics = physics_cv.iloc[0]
    best_lgbm = lgbm_cv.iloc[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    physics_pred = physics_prediction(
        test_coords,
        velocity_scale=float(best_physics["velocity_scale"]),
        accel_scale=float(best_physics["accel_scale"]),
    )
    physics_path = args.output_dir / "aggressive_physics_cv_best.csv"
    physics_submission = sample_submission[[ID_COLUMN]].copy()
    physics_submission[COORD_COLUMNS] = physics_pred
    physics_submission.to_csv(physics_path, index=False)

    base_train = physics_prediction(train_coords, accel_scale=0.15)
    base_test = physics_prediction(test_coords, accel_scale=0.15)
    residual = y - base_train
    residual_test_pred = np.mean(
        [train_full_lgbm_submission(train_features, residual, test_features, seed) for seed in SEEDS[:3]],
        axis=0,
    )
    lgbm_pred = base_test + float(best_lgbm["shrink"]) * residual_test_pred
    lgbm_path = args.output_dir / "aggressive_lgbm_residual.csv"
    lgbm_submission = sample_submission[[ID_COLUMN]].copy()
    lgbm_submission[COORD_COLUMNS] = lgbm_pred
    lgbm_submission.to_csv(lgbm_path, index=False)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Aggressive Experiments",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- CV seeds: `{SEEDS}`",
        f"- Validation fraction per seed: `{args.val_frac}`",
        f"- Best physics mean R-Hit@1cm: `{best_physics['mean_r_hit']:.6f}`",
        f"- Best physics config: velocity_scale `{best_physics['velocity_scale']:.6f}`, accel_scale `{best_physics['accel_scale']:.6f}`",
        f"- Physics submission: `{physics_path}`",
        f"- Best LGBM residual mean R-Hit@1cm: `{best_lgbm['mean_r_hit']:.6f}`",
        f"- Best LGBM shrink: `{best_lgbm['shrink']:.6f}`",
        f"- LGBM submission: `{lgbm_path}`",
        "",
        "## Physics CV Top 20",
        "",
        dataframe_to_markdown(physics_cv.head(20)),
        "",
        "## LightGBM Residual CV",
        "",
        dataframe_to_markdown(lgbm_cv),
        "",
        "## Readout",
        "",
        "- Use the physics CV candidate as the safer next submission because it is deterministic and stable across seeds.",
        "- Use the LGBM residual candidate only if CV improves both hit rate and mean distance; residual models can easily optimize distance while losing the 1cm threshold.",
        "- If physics still dominates, the next aggressive path is sample-wise method selection using trajectory noise/curvature buckets.",
    ]
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote physics submission: {physics_path}")
    print(f"Wrote LGBM submission: {lgbm_path}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()

