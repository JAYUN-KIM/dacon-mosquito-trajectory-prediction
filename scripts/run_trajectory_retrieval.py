from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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
from run_local_frame_residual import local_basis, to_global, to_local


CV_SEEDS = [42, 777, 2026, 3407, 10007]
FEATURE_VARIANTS = ["local_shape", "local_motion", "global_motion"]
TARGET_MODES = ["local_residual", "local_offset", "global_residual"]
K_VALUES = [1, 3, 5, 8, 12, 20, 32, 50]
WEIGHT_MODES = ["uniform", "inverse", "softmax0.75", "softmax1.25", "rank0.85"]
MAX_NEIGHBORS = max(K_VALUES)


def safe_scale(coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    step_norm = np.linalg.norm(diffs, axis=2)
    scale = np.median(step_norm[:, -5:], axis=1, keepdims=True)
    return np.maximum(scale, 1e-4)


def local_sequence_blocks(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = local_basis(coords)
    last = coords[:, -1, :]
    rel = coords - last[:, None, :]
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    rel_local = np.einsum("nwc,nkc->nwk", rel, basis)
    diffs_local = np.einsum("nwc,nkc->nwk", diffs, basis)
    dd_local = np.einsum("nwc,nkc->nwk", dd, basis)
    return rel_local, diffs_local, dd_local


def motion_summaries(coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)

    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    dot = np.sum(d_last * d_prev, axis=1, keepdims=True)
    denom = np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-12
    turn_cos = dot / denom

    path_len = speed.sum(axis=1, keepdims=True)
    displacement = np.linalg.norm(coords[:, -1, :] - coords[:, 0, :], axis=1, keepdims=True)
    tortuosity = path_len / (displacement + 1e-8)
    return np.hstack(
        [
            speed,
            accel,
            speed[:, -1:],
            speed[:, -3:].mean(axis=1, keepdims=True),
            speed.mean(axis=1, keepdims=True),
            accel[:, -1:],
            accel.mean(axis=1, keepdims=True),
            turn_cos,
            path_len,
            displacement,
            tortuosity,
        ]
    )


def make_retrieval_features(coords: np.ndarray, variant: str) -> np.ndarray:
    scale = safe_scale(coords)
    last = coords[:, -1, :]
    rel = (coords - last[:, None, :]) / scale[:, None, :]
    diffs = np.diff(coords, axis=1) / scale[:, None, :]
    dd = np.diff(np.diff(coords, axis=1), axis=1) / scale[:, None, :]
    summaries = motion_summaries(coords)

    if variant == "global_motion":
        blocks = [rel.reshape(len(coords), -1), diffs.reshape(len(coords), -1), dd.reshape(len(coords), -1), summaries]
    elif variant == "local_shape":
        rel_local, diffs_local, dd_local = local_sequence_blocks(coords)
        blocks = [
            (rel_local / scale[:, None, :]).reshape(len(coords), -1),
            (diffs_local / scale[:, None, :]).reshape(len(coords), -1),
            (dd_local / scale[:, None, :]).reshape(len(coords), -1),
        ]
    elif variant == "local_motion":
        rel_local, diffs_local, dd_local = local_sequence_blocks(coords)
        blocks = [
            (rel_local / scale[:, None, :]).reshape(len(coords), -1),
            (diffs_local / scale[:, None, :]).reshape(len(coords), -1),
            (dd_local / scale[:, None, :]).reshape(len(coords), -1),
            summaries,
        ]
    else:
        raise ValueError(f"unknown feature variant: {variant}")

    features = np.hstack(blocks).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    return features


def target_values(coords: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "global_residual":
        return y - physics_prediction(coords, 1.0, 0.275)
    if mode == "local_residual":
        return to_local(y - physics_prediction(coords, 1.0, 0.275), local_basis(coords))
    if mode == "local_offset":
        return to_local(y - coords[:, -1, :], local_basis(coords))
    raise ValueError(f"unknown target mode: {mode}")


def materialize_prediction(coords: np.ndarray, aggregated: np.ndarray, mode: str) -> np.ndarray:
    if mode == "global_residual":
        return physics_prediction(coords, 1.0, 0.275) + aggregated
    if mode == "local_residual":
        return physics_prediction(coords, 1.0, 0.275) + to_global(aggregated, local_basis(coords))
    if mode == "local_offset":
        return coords[:, -1, :] + to_global(aggregated, local_basis(coords))
    raise ValueError(f"unknown target mode: {mode}")


def neighbor_weights(distances: np.ndarray, mode: str, k: int) -> np.ndarray:
    d = distances[:, :k]
    if mode == "uniform":
        weights = np.ones_like(d)
    elif mode == "inverse":
        weights = 1.0 / np.maximum(d, 1e-6)
    elif mode.startswith("softmax"):
        tau = float(mode.replace("softmax", ""))
        local_scale = np.maximum(np.median(d, axis=1, keepdims=True), 1e-6)
        weights = np.exp(-0.5 * (d / (tau * local_scale)) ** 2)
    elif mode.startswith("rank"):
        decay = float(mode.replace("rank", ""))
        weights = decay ** np.arange(k, dtype=np.float64)[None, :]
    else:
        raise ValueError(f"unknown weight mode: {mode}")

    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def aggregate_neighbors(values: np.ndarray, indices: np.ndarray, distances: np.ndarray, k: int, weight_mode: str) -> np.ndarray:
    weights = neighbor_weights(distances, weight_mode, k)
    gathered = values[indices[:, :k]]
    return np.einsum("nk,nkc->nc", weights, gathered)


def evaluate_feature_variant(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    variant: str,
    val_frac: float,
) -> pd.DataFrame:
    rows = []
    for seed in CV_SEEDS:
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask

        scaler = StandardScaler()
        train_x = scaler.fit_transform(features[train_mask])
        val_x = scaler.transform(features[val_mask])
        nn = NearestNeighbors(n_neighbors=MAX_NEIGHBORS, metric="euclidean", n_jobs=-1)
        nn.fit(train_x)
        distances, indices = nn.kneighbors(val_x, return_distance=True)
        y_val = y[val_mask]
        coords_val = coords[val_mask]

        for target_mode in TARGET_MODES:
            train_values = target_values(coords[train_mask], y[train_mask], target_mode)
            for k in K_VALUES:
                for weight_mode in WEIGHT_MODES:
                    aggregated = aggregate_neighbors(train_values, indices, distances, k, weight_mode)
                    pred = materialize_prediction(coords_val, aggregated, target_mode)
                    rows.append(
                        {
                            "feature_variant": variant,
                            "target_mode": target_mode,
                            "seed": seed,
                            "k": k,
                            "weight_mode": weight_mode,
                            **distance_summary(pred, y_val),
                        }
                    )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(["feature_variant", "target_mode", "k", "weight_mode"], as_index=False)
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


def predict_test_config(
    train_coords: np.ndarray,
    y: np.ndarray,
    test_coords: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
    target_mode: str,
    k: int,
    weight_mode: str,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features)
    test_x = scaler.transform(test_features)
    nn = NearestNeighbors(n_neighbors=max(k, 1), metric="euclidean", n_jobs=-1)
    nn.fit(train_x)
    distances, indices = nn.kneighbors(test_x, return_distance=True)
    train_values = target_values(train_coords, y, target_mode)
    aggregated = aggregate_neighbors(train_values, indices, distances, k, weight_mode)
    return materialize_prediction(test_coords, aggregated, target_mode)


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nonparametric trajectory retrieval / kNN future aggregation.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_trajectory_retrieval.md")
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

    feature_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    leaderboards = []
    for variant in FEATURE_VARIANTS:
        print(f"Building retrieval features: {variant}", flush=True)
        train_features = make_retrieval_features(train_coords, variant)
        test_features = make_retrieval_features(test_coords, variant)
        feature_map[variant] = (train_features, test_features)
        print(f"feature_count[{variant}]={train_features.shape[1]}", flush=True)

        print(f"Evaluating retrieval CV: {variant}", flush=True)
        leaderboard = evaluate_feature_variant(train_coords, y, train_features, variant, args.val_frac)
        leaderboards.append(leaderboard)
        print(leaderboard.head(10).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    all_leaderboard = pd.concat(leaderboards, ignore_index=True).sort_values(
        ["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True]
    )
    top = all_leaderboard.head(30)
    print("\nTop retrieval configs", flush=True)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    for rank, (_, row) in enumerate(all_leaderboard.head(args.top_k).iterrows(), start=1):
        variant = str(row["feature_variant"])
        target_mode = str(row["target_mode"])
        k = int(row["k"])
        weight_mode = str(row["weight_mode"])
        train_features, test_features = feature_map[variant]
        pred = predict_test_config(train_coords, y, test_coords, train_features, test_features, target_mode, k, weight_mode)
        path = (
            args.output_dir
            / f"retrieval_rank{rank}_{slug(variant)}_{slug(target_mode)}_k{k}_{slug(weight_mode)}.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Trajectory Retrieval",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Feature variants: `{FEATURE_VARIANTS}`",
        f"- Target modes: `{TARGET_MODES}`",
        f"- K values: `{K_VALUES}`",
        f"- Weight modes: `{WEIGHT_MODES}`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Top 30 Retrieval Configs",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- This is a new nonparametric direction: retrieve similar past trajectories and aggregate their future residual/offset.",
        "- It is intentionally different from the LightGBM local-frame residual family, so it can be useful even if standalone public score is lower.",
        "- If standalone score is close to the current best, the next step is blend or sample-wise routing between retrieval and local-frame residual.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()

