from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
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


CV_SEEDS = [42, 777, 2026]
FULL_SEEDS = [42, 777, 2026]
SHRINKS = [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]
BASE_CONFIGS = [
    ("a015", 1.00, 0.15),
    ("a0275", 1.00, 0.275),
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    make_model: Callable[[int], object]


def make_lgbm_base(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=520,
        learning_rate=0.022,
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


def make_lgbm_wide(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=520,
        learning_rate=0.022,
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


def make_lgbm_smooth(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=760,
        learning_rate=0.014,
        num_leaves=24,
        min_child_samples=36,
        subsample=0.84,
        subsample_freq=1,
        colsample_bytree=0.84,
        reg_alpha=0.08,
        reg_lambda=0.65,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_catboost(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=420,
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


MODEL_SPECS = [
    ModelSpec("lgbm_base", make_lgbm_base),
    ModelSpec("lgbm_wide", make_lgbm_wide),
    ModelSpec("lgbm_smooth", make_lgbm_smooth),
    ModelSpec("catboost_d5", make_catboost),
]


def fit_predict_axes(spec: ModelSpec, train_x: np.ndarray, train_y: np.ndarray, pred_x: np.ndarray, seed: int) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = spec.make_model(seed + axis)
        model.fit(train_x, train_y[:, axis])
        preds.append(model.predict(pred_x))
    return np.vstack(preds).T


def evaluate_model_zoo(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    rows = []
    for base_name, velocity_scale, accel_scale in BASE_CONFIGS:
        base = physics_prediction(coords, velocity_scale=velocity_scale, accel_scale=accel_scale)
        residual = y - base
        for spec in MODEL_SPECS:
            print(f"CV {spec.name} on {base_name}")
            for seed in CV_SEEDS:
                val_mask = split_mask(len(coords), val_frac, seed)
                train_mask = ~val_mask
                residual_pred = fit_predict_axes(
                    spec,
                    features[train_mask],
                    residual[train_mask],
                    features[val_mask],
                    seed,
                )
                base_val = base[val_mask]
                y_val = y[val_mask]
                for shrink in SHRINKS:
                    pred = base_val + shrink * residual_pred
                    rows.append(
                        {
                            "model": spec.name,
                            "base": base_name,
                            "velocity_scale": velocity_scale,
                            "accel_scale": accel_scale,
                            "seed": seed,
                            "shrink": shrink,
                            **distance_summary(pred, y_val),
                        }
                    )
    df = pd.DataFrame(rows)
    return (
        df.groupby(["model", "base", "velocity_scale", "accel_scale", "shrink"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )


def full_residual_prediction(spec: ModelSpec, train_x: np.ndarray, residual: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    return np.mean([fit_predict_axes(spec, train_x, residual, test_x, seed) for seed in FULL_SEEDS], axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = sample_submission[[ID_COLUMN]].copy()
    sub[COORD_COLUMNS] = pred
    sub.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare residual model zoo candidates and write top submissions.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_residual_model_zoo.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
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
    test_ids = sample_submission[ID_COLUMN].tolist()
    test_coords = stack_samples(test_samples, test_ids)
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)

    print("Building features")
    train_features = make_features(train_coords)
    test_features = make_features(test_coords)

    leaderboard = evaluate_model_zoo(train_coords, y, train_features, args.val_frac)
    top = leaderboard.head(20).copy()
    print("\nTop residual zoo candidates:")
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    specs_by_name = {spec.name: spec for spec in MODEL_SPECS}
    base_by_name = {name: (velocity_scale, accel_scale) for name, velocity_scale, accel_scale in BASE_CONFIGS}
    prediction_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    written_paths = []

    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        key = (str(row["model"]), str(row["base"]))
        if key not in prediction_cache:
            velocity_scale, accel_scale = base_by_name[key[1]]
            base_train = physics_prediction(train_coords, velocity_scale=velocity_scale, accel_scale=accel_scale)
            base_test = physics_prediction(test_coords, velocity_scale=velocity_scale, accel_scale=accel_scale)
            residual = y - base_train
            residual_test = full_residual_prediction(specs_by_name[key[0]], train_features, residual, test_features)
            prediction_cache[key] = (base_test, residual_test)

        base_test, residual_test = prediction_cache[key]
        pred = base_test + float(row["shrink"]) * residual_test
        out_path = args.output_dir / f"residual_zoo_rank{rank}_{row['model']}_{row['base']}_s{float(row['shrink']):.2f}.csv"
        write_submission(sample_submission, pred, out_path)
        written_paths.append(out_path)

    blend_rows = leaderboard.head(2)
    blend_preds = []
    for _, row in blend_rows.iterrows():
        key = (str(row["model"]), str(row["base"]))
        if key not in prediction_cache:
            velocity_scale, accel_scale = base_by_name[key[1]]
            base_train = physics_prediction(train_coords, velocity_scale=velocity_scale, accel_scale=accel_scale)
            base_test = physics_prediction(test_coords, velocity_scale=velocity_scale, accel_scale=accel_scale)
            residual = y - base_train
            residual_test = full_residual_prediction(specs_by_name[key[0]], train_features, residual, test_features)
            prediction_cache[key] = (base_test, residual_test)
        base_test, residual_test = prediction_cache[key]
        blend_preds.append(base_test + float(row["shrink"]) * residual_test)
    blend_path = args.output_dir / "residual_zoo_top2_blend.csv"
    write_submission(sample_submission, np.mean(blend_preds, axis=0), blend_path)
    written_paths.append(blend_path)

    report = [
        "# Residual Model Zoo",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full-train seeds: `{FULL_SEEDS}`",
        f"- Shrinks: `{SHRINKS}`",
        f"- Written submissions: `{[str(path) for path in written_paths]}`",
        "",
        "## Top 20",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- Compare these against `aggressive_lgbm_residual.csv`; public already liked residual modeling, so nearby model/shrink variants are worth trying early.",
        "- Prefer candidates that improve mean hit while keeping `min_r_hit` competitive across CV seeds.",
        "- If CatBoost appears near the top, the top-2 blend is especially interesting because model errors should be less correlated.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote report: {args.report_path}")
    for path in written_paths:
        print(f"Wrote submission: {path}")


if __name__ == "__main__":
    main()

