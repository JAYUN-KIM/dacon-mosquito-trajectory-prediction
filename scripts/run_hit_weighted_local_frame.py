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
from run_feature_rich_residual import POLY_CONFIGS, poly_prediction, weighted_diff_prediction
from run_local_frame_residual import local_basis, make_local_frame_features, to_global, to_local


warnings.filterwarnings("ignore", message="X does not have valid feature names")


CV_SEEDS = [42]
FULL_SEEDS = [42, 777, 2026]
OBJECTIVES = ["l2"]
WEIGHT_MODES = ["uniform", "base_boundary", "candidate_boundary", "near_hit_band", "hard_miss_band"]
FORWARD_SHRINKS = [0.46, 0.48, 0.52]
SIDE_SHRINKS = [0.52, 0.55, 0.58]
UP_SHRINKS = [0.62, 0.70, 0.78]


def make_lgbm(seed: int, objective_name: str) -> LGBMRegressor:
    objective = "regression" if objective_name == "l2" else objective_name
    params = {
        "objective": objective,
        "n_estimators": 360,
        "learning_rate": 0.026,
        "num_leaves": 63,
        "min_child_samples": 14,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.92,
        "reg_alpha": 0.02,
        "reg_lambda": 0.24,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if objective_name == "huber":
        params["alpha"] = 0.90
    return LGBMRegressor(**params)


def safe_scale(coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    scale = np.median(speed[:, -5:], axis=1, keepdims=True)
    return np.maximum(scale, 1e-4)


def normalized_geometry_features(coords: np.ndarray) -> np.ndarray:
    last = coords[:, -1, :]
    scale = safe_scale(coords)
    basis = local_basis(coords)

    rel = (coords - last[:, None, :]) / scale[:, None, :]
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    ddd = np.diff(dd, axis=1)
    diffs_norm = diffs / scale[:, None, :]
    dd_norm = dd / scale[:, None, :]
    ddd_norm = ddd / scale[:, None, :]

    rel_local = np.einsum("nwc,nkc->nwk", coords - last[:, None, :], basis) / scale[:, None, :]
    diffs_local = np.einsum("nwc,nkc->nwk", diffs, basis) / scale[:, None, :]
    dd_local = np.einsum("nwc,nkc->nwk", dd, basis) / scale[:, None, :]

    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    jerk = np.linalg.norm(ddd, axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    dot = np.sum(d_last * d_prev, axis=1, keepdims=True)
    denom = np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-12
    turn_cos = dot / denom

    vertical_step_ratio = np.abs(diffs_local[:, -1, 2:3]) / (np.linalg.norm(diffs_local[:, -1, :], axis=1, keepdims=True) + 1e-8)
    side_step_ratio = np.abs(diffs_local[:, -1, 1:2]) / (np.linalg.norm(diffs_local[:, -1, :], axis=1, keepdims=True) + 1e-8)
    accel_ratio = accel[:, -1:] / (accel.mean(axis=1, keepdims=True) + 1e-8)
    speed_ratio = speed[:, -1:] / (speed.mean(axis=1, keepdims=True) + 1e-8)

    blocks = [
        rel.reshape(len(coords), -1),
        diffs_norm.reshape(len(coords), -1),
        dd_norm.reshape(len(coords), -1),
        ddd_norm.reshape(len(coords), -1),
        rel_local.reshape(len(coords), -1),
        diffs_local.reshape(len(coords), -1),
        dd_local.reshape(len(coords), -1),
        speed,
        accel,
        jerk,
        speed[:, -1:],
        speed[:, -3:].mean(axis=1, keepdims=True),
        speed[:, -5:].std(axis=1, keepdims=True),
        accel[:, -1:],
        accel[:, -3:].mean(axis=1, keepdims=True),
        turn_cos,
        vertical_step_ratio,
        side_step_ratio,
        accel_ratio,
        speed_ratio,
    ]
    features = np.hstack(blocks).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    return features


def make_features(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    local_features, candidate_names = make_local_frame_features(coords)
    normalized = normalized_geometry_features(coords)
    return np.hstack([local_features, normalized]).astype(np.float32), candidate_names


def physics_poly_candidates(coords: np.ndarray) -> np.ndarray:
    candidates = [
        physics_prediction(coords, 1.00, 0.00),
        physics_prediction(coords, 1.00, 0.15),
        physics_prediction(coords, 1.00, 0.275),
        physics_prediction(coords, 1.00, 0.40),
        physics_prediction(coords, 0.98, 0.275),
        physics_prediction(coords, 1.02, 0.275),
    ]
    for window, degree in POLY_CONFIGS:
        candidates.append(poly_prediction(coords, window, degree))
    candidates.extend(
        [
            weighted_diff_prediction(coords, 5, 0.50),
            weighted_diff_prediction(coords, 5, 0.75),
            weighted_diff_prediction(coords, 7, 0.60),
            weighted_diff_prediction(coords, 11, 0.70),
        ]
    )
    return np.stack(candidates, axis=1)


def sample_weights(coords: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(coords), dtype=np.float64)

    base_dist = np.linalg.norm(physics_prediction(coords, 1.0, 0.275) - y, axis=1)
    candidate_min_dist = np.linalg.norm(physics_poly_candidates(coords) - y[:, None, :], axis=2).min(axis=1)

    if mode == "base_boundary":
        weights = 1.0 + 4.0 * np.exp(-0.5 * ((base_dist - 0.010) / 0.004) ** 2)
    elif mode == "candidate_boundary":
        weights = 1.0 + 5.0 * np.exp(-0.5 * ((candidate_min_dist - 0.010) / 0.003) ** 2)
    elif mode == "near_hit_band":
        weights = 1.0 + 2.0 * (base_dist <= 0.018) + 2.0 * (candidate_min_dist <= 0.012)
    elif mode == "hard_miss_band":
        weights = 1.0 + 4.0 * ((base_dist > 0.010) & (base_dist <= 0.025))
    else:
        raise ValueError(f"unknown weight mode: {mode}")

    weights = np.clip(weights, 0.5, 8.0)
    return weights / np.mean(weights)


def shrink_grid() -> list[tuple[float, float, float]]:
    return [(f, s, u) for f in FORWARD_SHRINKS for s in SIDE_SHRINKS for u in UP_SHRINKS]


def fit_predict_axes_weighted(
    train_x: np.ndarray,
    train_y: np.ndarray,
    pred_x: np.ndarray,
    seed: int,
    objective_name: str,
    weights: np.ndarray,
) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = make_lgbm(seed + axis, objective_name)
        model.fit(train_x, train_y[:, axis], sample_weight=weights)
        preds.append(model.predict(pred_x))
    return np.vstack(preds).T


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    base = physics_prediction(coords, 1.0, 0.275)
    basis = local_basis(coords)
    residual_local = to_local(y - base, basis)
    grid = shrink_grid()
    rows = []

    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        y_val = y[val_mask]
        base_val = base[val_mask]
        basis_val = basis[val_mask]

        for objective_name in OBJECTIVES:
            for weight_mode in WEIGHT_MODES:
                weights = sample_weights(coords[train_mask], y[train_mask], weight_mode)
                pred_local = fit_predict_axes_weighted(
                    features[train_mask],
                    residual_local[train_mask],
                    features[val_mask],
                    seed,
                    objective_name,
                    weights,
                )
                for forward, side, up in grid:
                    shrink = np.array([forward, side, up], dtype=np.float64)
                    pred = base_val + to_global(pred_local * shrink[None, :], basis_val)
                    rows.append(
                        {
                            "objective": objective_name,
                            "weight_mode": weight_mode,
                            "seed": seed,
                            "forward_shrink": forward,
                            "side_shrink": side,
                            "up_shrink": up,
                            **distance_summary(pred, y_val),
                        }
                    )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(["objective", "weight_mode", "forward_shrink", "side_shrink", "up_shrink"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    leaderboard["risk_adjusted_hit"] = leaderboard["mean_r_hit"] - 0.25 * leaderboard["std_r_hit"].fillna(0.0)
    return leaderboard


def full_prediction(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    test_coords: np.ndarray,
    test_features: np.ndarray,
    objective_name: str,
    weight_mode: str,
) -> np.ndarray:
    base_train = physics_prediction(coords, 1.0, 0.275)
    residual_local = to_local(y - base_train, local_basis(coords))
    weights = sample_weights(coords, y, weight_mode)
    seed_preds = []
    for seed in FULL_SEEDS:
        seed_preds.append(fit_predict_axes_weighted(features, residual_local, test_features, seed, objective_name, weights))
    return np.mean(seed_preds, axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metric-aware weighted local-frame residual experiments.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_hit_weighted_local_frame.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}", flush=True)

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

    print("Building hit-weighted feature matrix", flush=True)
    train_features, candidate_names = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    print(f"feature_count={train_features.shape[1]}", flush=True)

    print("Evaluating weighted local-frame CV", flush=True)
    leaderboard = evaluate_cv(train_coords, y, train_features, args.val_frac)
    top = leaderboard.head(40)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    base_test = physics_prediction(test_coords, 1.0, 0.275)
    test_basis = local_basis(test_coords)
    cache: dict[tuple[str, str], np.ndarray] = {}
    written = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        objective_name = str(row["objective"])
        weight_mode = str(row["weight_mode"])
        key = (objective_name, weight_mode)
        if key not in cache:
            print(f"Training full model: objective={objective_name}, weight_mode={weight_mode}", flush=True)
            cache[key] = full_prediction(train_coords, y, train_features, test_coords, test_features, objective_name, weight_mode)
        shrink = np.array([row["forward_shrink"], row["side_shrink"], row["up_shrink"]], dtype=np.float64)
        pred = base_test + to_global(cache[key] * shrink[None, :], test_basis)
        path = (
            args.output_dir
            / (
                f"hit_weighted_rank{rank}_{objective_name}_{weight_mode}_"
                f"f{shrink[0]:.2f}_s{shrink[1]:.2f}_u{shrink[2]:.2f}.csv"
            )
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Hit-Weighted Local Frame",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- Feature count: `{train_features.shape[1]}`",
        f"- Objectives: `{OBJECTIVES}`",
        f"- Weight modes: `{WEIGHT_MODES}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full ensemble seeds: `{FULL_SEEDS}`",
        f"- Candidate feature families: `{candidate_names}`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Top 40 Weighted Local-Frame Configs",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- This is a broader restart experiment, not a route-blend micro-tune.",
        "- It adds normalized trajectory geometry features and trains with hit-boundary-aware sample weights.",
        "- If this beats the 0.6604 anchor, continue with metric-aware objectives; otherwise move to regime/cluster routing or sequence neural models.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
