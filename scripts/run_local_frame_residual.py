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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, physics_prediction, split_mask, stack_samples
from run_feature_rich_residual import make_rich_features


warnings.filterwarnings("ignore", message="X does not have valid feature names")


CV_SEEDS = [42, 777, 2026, 3407, 10007]
FULL_SEEDS = [42, 777, 2026, 3407, 10007]
SHRINKS = [0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.55]


def make_lgbm(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=640,
        learning_rate=0.017,
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


def normalize(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = norm[:, 0] > 1e-10
    out = np.empty_like(vectors, dtype=np.float64)
    out[safe] = vectors[safe] / norm[safe]
    out[~safe] = fallback
    return out


def local_basis(coords: np.ndarray) -> np.ndarray:
    forward = normalize(coords[:, -1, :] - coords[:, -2, :], np.array([1.0, 0.0, 0.0]))
    z_ref = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(coords), axis=0)
    y_ref = np.repeat(np.array([[0.0, 1.0, 0.0]]), len(coords), axis=0)

    side_raw = np.cross(z_ref, forward)
    near_vertical = np.linalg.norm(side_raw, axis=1) < 1e-8
    side_raw[near_vertical] = np.cross(y_ref[near_vertical], forward[near_vertical])
    side = normalize(side_raw, np.array([0.0, 1.0, 0.0]))
    up = normalize(np.cross(forward, side), np.array([0.0, 0.0, 1.0]))
    return np.stack([forward, side, up], axis=1)


def to_local(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", basis, vectors)


def to_global(local_vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nij->nj", local_vectors, basis)


def local_motion_features(coords: np.ndarray, basis: np.ndarray) -> np.ndarray:
    last = coords[:, -1, :]
    rel = coords - last[:, None, :]
    rel_local = np.einsum("nwc,nkc->nwk", rel, basis)
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    diffs_local = np.einsum("nwc,nkc->nwk", diffs, basis)
    dd_local = np.einsum("nwc,nkc->nwk", dd, basis)

    forward_step = diffs_local[:, -1, 0:1]
    lateral_step = np.linalg.norm(diffs_local[:, -1, 1:], axis=1, keepdims=True)
    accel_forward = dd_local[:, -1, 0:1]
    accel_lateral = np.linalg.norm(dd_local[:, -1, 1:], axis=1, keepdims=True)
    return np.hstack(
        [
            rel_local.reshape(len(coords), -1),
            diffs_local.reshape(len(coords), -1),
            dd_local.reshape(len(coords), -1),
            forward_step,
            lateral_step,
            accel_forward,
            accel_lateral,
        ]
    ).astype(np.float32)


def make_local_frame_features(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rich, candidate_names = make_rich_features(coords)
    basis = local_basis(coords)
    local = local_motion_features(coords, basis)
    return np.hstack([rich, local]).astype(np.float32), candidate_names


def fit_predict_axes(train_x: np.ndarray, train_y: np.ndarray, pred_x: np.ndarray, seed: int) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = make_lgbm(seed + axis)
        model.fit(train_x, train_y[:, axis])
        preds.append(model.predict(pred_x))
    return np.vstack(preds).T


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    base = physics_prediction(coords, 1.0, 0.275)
    basis = local_basis(coords)
    residual_local = to_local(y - base, basis)

    rows = []
    for seed in CV_SEEDS:
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        pred_local = fit_predict_axes(features[train_mask], residual_local[train_mask], features[val_mask], seed)
        pred_global_residual = to_global(pred_local, basis[val_mask])
        y_val = y[val_mask]
        base_val = base[val_mask]
        for shrink in SHRINKS:
            pred = base_val + shrink * pred_global_residual
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


def full_local_residual_prediction(features: np.ndarray, residual_local: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    return np.mean([fit_predict_axes(features, residual_local, test_features, seed) for seed in FULL_SEEDS], axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict residuals in the local last-velocity coordinate frame.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_local_frame_residual.md")
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

    print("Building local-frame feature matrix")
    train_features, candidate_names = make_local_frame_features(train_coords)
    test_features, _ = make_local_frame_features(test_coords)
    print(f"feature_count={train_features.shape[1]}")

    print("Evaluating CV")
    leaderboard = evaluate_cv(train_coords, y, train_features, args.val_frac)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    base_train = physics_prediction(train_coords, 1.0, 0.275)
    base_test = physics_prediction(test_coords, 1.0, 0.275)
    train_basis = local_basis(train_coords)
    test_basis = local_basis(test_coords)
    residual_local = to_local(y - base_train, train_basis)

    print("Training full 5-seed local-frame ensemble")
    test_residual_local = full_local_residual_prediction(train_features, residual_local, test_features)
    test_residual_global = to_global(test_residual_local, test_basis)

    written = []
    for shrink in [0.32, 0.36, 0.40, 0.44, 0.48, 0.55]:
        pred = base_test + shrink * test_residual_global
        path = args.output_dir / f"local_frame_lgbm_a0275_s{shrink:.2f}_5seed.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Local-Frame Residual",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- Feature count: `{train_features.shape[1]}`",
        f"- Candidate feature families: `{candidate_names}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full ensemble seeds: `{FULL_SEEDS}`",
        "- Target: residual projected into the final-velocity local frame",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Shrink CV",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Readout",
        "",
        "- This is the first geometry-change experiment after the 0.6412/0.6434 residual family.",
        "- If it improves, continue with local-frame features and possibly separate along-track/cross-track model capacity.",
        "- If it underperforms, keep global residual targets and focus on candidate-derived features/bucketing.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}")
    for path in written:
        print(f"Wrote submission: {path}")


if __name__ == "__main__":
    main()
