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
from run_aggressive_experiments import (
    dataframe_to_markdown,
    distance_summary,
    make_features,
    physics_prediction,
    split_mask,
    stack_samples,
)


warnings.filterwarnings("ignore", message="X does not have valid feature names")


CV_SEEDS = [42, 777, 2026, 3407, 10007]
FULL_SEEDS = [42, 777, 2026, 3407, 10007]
SHRINKS = [0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.55]
POLY_CONFIGS = [(3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (7, 1), (7, 2), (11, 1), (11, 2), (11, 3)]


def make_lgbm(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=620,
        learning_rate=0.018,
        num_leaves=63,
        min_child_samples=16,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.92,
        reg_alpha=0.02,
        reg_lambda=0.22,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def poly_weights(window: int, degree: int, pred_step: float = 2.0) -> np.ndarray:
    t = np.arange(-(window - 1), 1, dtype=float)
    powers = np.arange(degree + 1)
    design = t[:, None] ** powers[None, :]
    pred_row = pred_step ** powers
    return pred_row @ np.linalg.pinv(design)


def poly_prediction(coords: np.ndarray, window: int, degree: int) -> np.ndarray:
    weights = poly_weights(window, degree)
    return np.einsum("w,nwc->nc", weights, coords[:, -window:, :])


def weighted_diff_prediction(coords: np.ndarray, window: int, decay: float) -> np.ndarray:
    diffs = np.diff(coords[:, -window:, :], axis=1)
    weights = decay ** np.arange(diffs.shape[1] - 1, -1, -1, dtype=float)
    weights = weights / weights.sum()
    step = np.einsum("w,nwc->nc", weights, diffs)
    return coords[:, -1, :] + 2.0 * step


def candidate_block(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    last = coords[:, -1, :]
    base_candidates = [
        ("phys_a000", physics_prediction(coords, 1.0, 0.0)),
        ("phys_a150", physics_prediction(coords, 1.0, 0.15)),
        ("phys_a275", physics_prediction(coords, 1.0, 0.275)),
        ("phys_a400", physics_prediction(coords, 1.0, 0.40)),
        ("phys_v098_a275", physics_prediction(coords, 0.98, 0.275)),
        ("phys_v102_a275", physics_prediction(coords, 1.02, 0.275)),
    ]
    poly_candidates = [(f"poly_w{window}_d{degree}", poly_prediction(coords, window, degree)) for window, degree in POLY_CONFIGS]
    smooth_candidates = [
        ("wdiff_w5_d050", weighted_diff_prediction(coords, 5, 0.50)),
        ("wdiff_w5_d075", weighted_diff_prediction(coords, 5, 0.75)),
        ("wdiff_w7_d060", weighted_diff_prediction(coords, 7, 0.60)),
        ("wdiff_w11_d070", weighted_diff_prediction(coords, 11, 0.70)),
    ]
    candidates = base_candidates + poly_candidates + smooth_candidates

    blocks = []
    names: list[str] = []
    ref = physics_prediction(coords, 1.0, 0.275)
    for name, pred in candidates:
        delta = pred - last
        diff_ref = pred - ref
        blocks.extend([pred, delta, diff_ref, np.linalg.norm(delta, axis=1, keepdims=True), np.linalg.norm(diff_ref, axis=1, keepdims=True)])
        names.append(name)

    stacked = np.stack([pred for _, pred in candidates], axis=1)
    spread = np.std(stacked, axis=1)
    spread_norm = np.linalg.norm(spread, axis=1, keepdims=True)
    blocks.extend([spread, spread_norm])
    return np.hstack(blocks).astype(np.float32), names


def motion_summary_features(coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    last_speed = speed[:, -1:]
    mean_speed_3 = speed[:, -3:].mean(axis=1, keepdims=True)
    mean_speed_all = speed.mean(axis=1, keepdims=True)
    speed_ratio = last_speed / (mean_speed_all + 1e-8)
    accel_last = accel[:, -1:]
    accel_mean = accel.mean(axis=1, keepdims=True)
    accel_ratio = accel_last / (accel_mean + 1e-8)
    path_len = speed.sum(axis=1, keepdims=True)
    displacement = np.linalg.norm(coords[:, -1, :] - coords[:, 0, :], axis=1, keepdims=True)
    tortuosity = path_len / (displacement + 1e-8)
    return np.hstack([last_speed, mean_speed_3, mean_speed_all, speed_ratio, accel_last, accel_mean, accel_ratio, path_len, displacement, tortuosity]).astype(np.float32)


def make_rich_features(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    base = make_features(coords)
    candidates, candidate_names = candidate_block(coords)
    summary = motion_summary_features(coords)
    return np.hstack([base, candidates, summary]).astype(np.float32), candidate_names


def fit_predict_axes(train_x: np.ndarray, train_y: np.ndarray, pred_x: np.ndarray, seed: int) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = make_lgbm(seed + axis)
        model.fit(train_x, train_y[:, axis])
        preds.append(model.predict(pred_x))
    return np.vstack(preds).T


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    base = physics_prediction(coords, 1.0, 0.275)
    residual = y - base
    rows = []
    for seed in CV_SEEDS:
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        residual_pred = fit_predict_axes(features[train_mask], residual[train_mask], features[val_mask], seed)
        y_val = y[val_mask]
        base_val = base[val_mask]
        for shrink in SHRINKS:
            pred = base_val + shrink * residual_pred
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
    parser = argparse.ArgumentParser(description="Run residual LGBM with feature-rich physics/poly candidate features.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_feature_rich_residual.md")
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

    print("Building feature-rich design matrix")
    train_features, candidate_names = make_rich_features(train_coords)
    test_features, _ = make_rich_features(test_coords)
    print(f"feature_count={train_features.shape[1]}")

    print("Evaluating CV")
    leaderboard = evaluate_cv(train_coords, y, train_features, args.val_frac)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    base_train = physics_prediction(train_coords, 1.0, 0.275)
    base_test = physics_prediction(test_coords, 1.0, 0.275)
    residual = y - base_train
    print("Training full 5-seed feature-rich ensemble")
    residual_test = full_residual_prediction(train_features, residual, test_features)

    written = []
    for shrink in [0.32, 0.36, 0.40, 0.44, 0.48]:
        pred = base_test + shrink * residual_test
        path = args.output_dir / f"feature_rich_lgbm_a0275_s{shrink:.2f}_5seed.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Feature-Rich Residual",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- Feature count: `{train_features.shape[1]}`",
        f"- Candidate feature families: `{candidate_names}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full ensemble seeds: `{FULL_SEEDS}`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Shrink CV",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Readout",
        "",
        "- This tests whether richer physics/poly candidate features improve residual modeling beyond the current 0.6412 public family.",
        "- If CV rises or stays flat while public improves, keep expanding candidate-derived features rather than switching to deep models.",
        "- If CV drops, the current base feature set is already near the sweet spot and the next move should be better validation/bucketing.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}")
    for path in written:
        print(f"Wrote submission: {path}")


if __name__ == "__main__":
    main()
