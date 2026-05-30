from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in [SRC, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mosquito_trajectory.data import (  # noqa: E402
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import delta_summary, read_submission_coords, write_submission  # noqa: E402


EXP_TAG = "analog_transport_20260530"
PUBLIC_WINNER = "recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv"
PUBLIC_WINNER_SCORE = 0.69200
R_HIT = 0.01
N_SPLITS = 5
SEED = 42
EPS = 1e-9


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def norm_columns(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True)


def hit_rate(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=1) <= R_HIT))


def constant_velocity_pred(coords: np.ndarray) -> np.ndarray:
    step = coords[:, -1, :] - coords[:, -2, :]
    return coords[:, -1, :] + 2.0 * step


def constant_acceleration_pred(coords: np.ndarray) -> np.ndarray:
    step = coords[:, -1, :] - coords[:, -2, :]
    accel = coords[:, -1, :] - 2.0 * coords[:, -2, :] + coords[:, -3, :]
    return coords[:, -1, :] + 2.0 * step + 2.0 * accel


def make_local_frame(coords: np.ndarray) -> np.ndarray:
    velocity = coords[:, -1, :] - coords[:, -2, :]
    accel = coords[:, -1, :] - 2.0 * coords[:, -2, :] + coords[:, -3, :]

    e1 = velocity / np.maximum(norm_columns(velocity), EPS)
    near_stationary = np.linalg.norm(velocity, axis=1) < 1e-8
    if np.any(near_stationary):
        fallback_velocity = coords[near_stationary, -1, :] - coords[near_stationary, -4, :]
        e1[near_stationary] = fallback_velocity / np.maximum(norm_columns(fallback_velocity), EPS)

    still_bad = np.linalg.norm(e1, axis=1) < 1e-8
    if np.any(still_bad):
        e1[still_bad] = np.array([1.0, 0.0, 0.0])

    accel_orth = accel - np.sum(accel * e1, axis=1, keepdims=True) * e1
    e2 = accel_orth / np.maximum(norm_columns(accel_orth), EPS)
    bad_accel = np.linalg.norm(accel_orth, axis=1) < 1e-8
    if np.any(bad_accel):
        axes = np.eye(3)
        fallback = np.zeros((int(bad_accel.sum()), 3), dtype=np.float64)
        for idx, direction in enumerate(e1[bad_accel]):
            axis = axes[np.argmin(np.abs(axes @ direction))]
            orth = axis - np.sum(axis * direction) * direction
            fallback[idx] = orth / max(float(np.linalg.norm(orth)), EPS)
        e2[bad_accel] = fallback

    e3 = np.cross(e1, e2)
    e3 = e3 / np.maximum(norm_columns(e3), EPS)
    e2 = np.cross(e3, e1)
    e2 = e2 / np.maximum(norm_columns(e2), EPS)
    return np.stack([e1, e2, e3], axis=2)


def global_to_local(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("nd,ndk->nk", vectors, basis)


def local_to_global(local_vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("nk,ndk->nd", local_vectors, basis)


def project_sequence_to_local(seq: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ntd,ndk->ntk", seq, basis)


def build_analog_features(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(coords)
    basis = make_local_frame(coords)
    rel = coords - coords[:, -1:, :]
    d1 = np.diff(coords, axis=1)
    d2 = np.diff(d1, axis=1)
    d3 = np.diff(d2, axis=1)

    step_norm = np.sqrt(np.mean(np.sum(d1**2, axis=2), axis=1, keepdims=True))
    step_norm = np.maximum(step_norm, 1e-7)

    rel_local = project_sequence_to_local(rel, basis) / step_norm[:, None, :]
    d1_local = project_sequence_to_local(d1, basis) / step_norm[:, None, :]
    d2_local = project_sequence_to_local(d2, basis) / step_norm[:, None, :]
    d3_local = project_sequence_to_local(d3, basis) / step_norm[:, None, :]

    speed = np.linalg.norm(d1, axis=2)
    speed_scaled = speed / step_norm
    accel_scaled = np.linalg.norm(d2, axis=2) / step_norm
    jerk_scaled = np.linalg.norm(d3, axis=2) / step_norm

    v0 = d1[:, :-1, :]
    v1 = d1[:, 1:, :]
    turn_cos = np.sum(v0 * v1, axis=2) / np.maximum(
        np.linalg.norm(v0, axis=2) * np.linalg.norm(v1, axis=2),
        EPS,
    )

    path_len = np.sum(speed, axis=1)
    chord = np.linalg.norm(coords[:, -1, :] - coords[:, 0, :], axis=1)
    straightness = chord / np.maximum(path_len, EPS)
    bbox = (np.max(coords, axis=1) - np.min(coords, axis=1)) / step_norm

    scalar = np.column_stack(
        [
            speed_scaled,
            accel_scaled,
            jerk_scaled,
            turn_cos,
            straightness,
            bbox,
            np.mean(speed_scaled, axis=1),
            np.std(speed_scaled, axis=1),
            np.max(speed_scaled, axis=1),
            np.min(speed_scaled, axis=1),
            speed_scaled[:, -1],
            speed_scaled[:, -1] / np.maximum(np.mean(speed_scaled, axis=1), EPS),
            accel_scaled[:, -1],
            np.mean(turn_cos, axis=1),
            np.std(turn_cos, axis=1),
        ]
    )

    features = np.hstack(
        [
            rel_local[:, -6:, :].reshape(n_rows, -1),
            d1_local[:, -6:, :].reshape(n_rows, -1),
            d2_local[:, -5:, :].reshape(n_rows, -1),
            d3_local[:, -4:, :].reshape(n_rows, -1),
            d1_local.reshape(n_rows, -1),
            d2_local.reshape(n_rows, -1),
            scalar,
        ]
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float64), basis


def local_targets(coords: np.ndarray, target: np.ndarray) -> np.ndarray:
    basis = make_local_frame(coords)
    return global_to_local(target - coords[:, -1, :], basis)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[..., None], axis=1)


def weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumsum = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumsum, quantile * cumsum[-1], side="left"))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def weighted_median_3d(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros((values.shape[0], 3), dtype=np.float64)
    for row_idx in range(values.shape[0]):
        for axis in range(3):
            out[row_idx, axis] = weighted_quantile_1d(values[row_idx, :, axis], weights[row_idx], 0.5)
    return out


def knn_predict_local(
    ref_features: np.ndarray,
    ref_local_target: np.ndarray,
    query_features: np.ndarray,
    k: int,
    mode: str,
    kernel_power: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    k_eff = min(k, len(ref_features))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(ref_features)
    distances, indices = nn.kneighbors(query_features, return_distance=True)
    neighbor_targets = ref_local_target[indices]

    sigma = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-6)
    weights = np.exp(-kernel_power * (distances / sigma) ** 2)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)

    if mode == "mean":
        pred_local = weighted_mean(neighbor_targets, weights)
    elif mode == "median":
        pred_local = weighted_median_3d(neighbor_targets, weights)
    elif mode == "trimmed_mean":
        keep = max(8, k_eff // 2)
        weights_keep = weights[:, :keep]
        weights_keep = weights_keep / np.maximum(weights_keep.sum(axis=1, keepdims=True), EPS)
        pred_local = weighted_mean(neighbor_targets[:, :keep, :], weights_keep)
    else:
        raise ValueError(f"unknown analog mode: {mode}")

    residual = neighbor_targets - pred_local[:, None, :]
    local_radius = np.sqrt(np.sum(weights * np.sum(residual**2, axis=2), axis=1))
    info = {
        "local_radius": local_radius,
        "mean_feature_dist": np.sum(weights * distances, axis=1),
        "kth_feature_dist": distances[:, -1],
    }
    return pred_local, info


def oof_analog_predictions(
    coords: np.ndarray,
    target: np.ndarray,
    k: int,
    mode: str,
    kernel_power: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features, basis = build_analog_features(coords)
    target_local = local_targets(coords, target)
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    pred = np.zeros_like(target, dtype=np.float64)
    info_all = {
        "local_radius": np.zeros(len(coords), dtype=np.float64),
        "mean_feature_dist": np.zeros(len(coords), dtype=np.float64),
        "kth_feature_dist": np.zeros(len(coords), dtype=np.float64),
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(features), start=1):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_idx])
        val_features = scaler.transform(features[val_idx])
        pred_local, info = knn_predict_local(
            train_features,
            target_local[train_idx],
            val_features,
            k,
            mode,
            kernel_power,
        )
        pred[val_idx] = coords[val_idx, -1, :] + local_to_global(pred_local, basis[val_idx])
        for key, values in info.items():
            info_all[key][val_idx] = values
        print(
            f"  fold {fold_idx}/{N_SPLITS}: "
            f"hit={hit_rate(pred[val_idx], target[val_idx]):.6f} "
            f"mean_dist={distance_summary(pred[val_idx], target[val_idx])['mean_distance']:.6f}",
            flush=True,
        )
    return pred, info_all


def fit_predict_test_analog(
    train_coords: np.ndarray,
    target: np.ndarray,
    test_coords: np.ndarray,
    k: int,
    mode: str,
    kernel_power: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_features, _ = build_analog_features(train_coords)
    test_features, test_basis = build_analog_features(test_coords)
    target_local = local_targets(train_coords, target)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    pred_local, info = knn_predict_local(train_features, target_local, test_features, k, mode, kernel_power)
    return test_coords[:, -1, :] + local_to_global(pred_local, test_basis), info


def percentile_rank_low(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def confidence_score(analog_pred: np.ndarray, anchor_pred: np.ndarray, info: dict[str, np.ndarray]) -> np.ndarray:
    radius_rank = percentile_rank_low(info["local_radius"])
    feature_rank = percentile_rank_low(info["mean_feature_dist"])
    move_rank = percentile_rank_low(np.linalg.norm(analog_pred - anchor_pred, axis=1))
    return -(0.45 * radius_rank + 0.35 * feature_rank + 0.20 * move_rank)


def selective_move(
    anchor: np.ndarray,
    analog: np.ndarray,
    score: np.ndarray,
    top_frac: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(anchor)
    n_select = max(1, int(round(n_rows * top_frac)))
    order = np.argsort(-score)
    mask = np.zeros(n_rows, dtype=bool)
    mask[order[:n_select]] = True
    pred = anchor.copy()
    pred[mask] = anchor[mask] + alpha * (analog[mask] - anchor[mask])
    return pred, mask


def main() -> None:
    data_dir = resolve_raw_data_dir(ROOT / "data" / "raw")
    submission_dir = ROOT / "submissions"
    report_dir = ROOT / "reports"
    submission_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}", flush=True)
    train_samples = read_trajectory_folder(data_dir / "train")
    test_samples = read_trajectory_folder(data_dir / "test")
    targets = read_targets(data_dir / "train_labels.csv")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    ids, missing = aligned_ids(train_samples, targets[ID_COLUMN].tolist())
    if missing:
        raise ValueError(f"{len(missing)} train ids are missing trajectory files")

    train_coords = stack_samples(train_samples, ids)
    test_ids = sample_submission[ID_COLUMN].tolist()
    test_coords = stack_samples(test_samples, test_ids)
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)

    winner_path = submission_dir / PUBLIC_WINNER
    winner_test = read_submission_coords(winner_path)
    cv_oof = constant_velocity_pred(train_coords)
    ca_oof = constant_acceleration_pred(train_coords)
    cv_test = constant_velocity_pred(test_coords)
    ca_test = constant_acceleration_pred(test_coords)

    print("Physics proxy baselines", flush=True)
    print(f"  CV hit={hit_rate(cv_oof, y):.6f}")
    print(f"  CA hit={hit_rate(ca_oof, y):.6f}")

    configs = [
        {"key": "mean_k64_p2", "k": 64, "mode": "mean", "kernel_power": 2.0},
        {"key": "median_k96_p2", "k": 96, "mode": "median", "kernel_power": 2.0},
        {"key": "trimmean_k128_p15", "k": 128, "mode": "trimmed_mean", "kernel_power": 1.5},
    ]
    rows = []
    analog_cache: dict[str, dict[str, object]] = {}

    for config in configs:
        print(f"Analog config={config['key']}", flush=True)
        oof_pred, oof_info = oof_analog_predictions(
            train_coords,
            y,
            k=config["k"],
            mode=config["mode"],
            kernel_power=config["kernel_power"],
        )
        test_pred, test_info = fit_predict_test_analog(
            train_coords,
            y,
            test_coords,
            k=config["k"],
            mode=config["mode"],
            kernel_power=config["kernel_power"],
        )
        analog_cache[config["key"]] = {
            "oof_pred": oof_pred,
            "oof_info": oof_info,
            "test_pred": test_pred,
            "test_info": test_info,
        }
        path = submission_dir / f"{EXP_TAG}_{config['key']}.csv"
        write_submission(sample_submission, test_pred, path)
        metrics = distance_summary(oof_pred, y)
        row = {
            "candidate": path.name,
            "kind": "analog_only",
            "oof_hit": metrics["r_hit_1cm"],
            "oof_mean_dist": metrics["mean_distance"],
            **delta_summary(test_pred, winner_test, "test_vs_winner"),
            "selected_frac": 1.0,
            "path": str(path),
        }
        rows.append(row)
        print(f"  saved={path.name} oof_hit={metrics['r_hit_1cm']:.6f}", flush=True)

    blend_specs = [
        ("mean_k64_p2", "cv", cv_oof, cv_test, 0.35),
        ("median_k96_p2", "cv", cv_oof, cv_test, 0.45),
        ("trimmean_k128_p15", "ca", ca_oof, ca_test, 0.30),
    ]
    for key, anchor_name, anchor_oof, anchor_test, alpha in blend_specs:
        cache = analog_cache[key]
        pred_oof = anchor_oof + alpha * (cache["oof_pred"] - anchor_oof)
        pred_test = anchor_test + alpha * (cache["test_pred"] - anchor_test)
        path = submission_dir / f"{EXP_TAG}_{key}_{anchor_name}blend_a{int(alpha * 100):02d}.csv"
        write_submission(sample_submission, pred_test, path)
        metrics = distance_summary(pred_oof, y)
        row = {
            "candidate": path.name,
            "kind": "physics_blend",
            "oof_hit": metrics["r_hit_1cm"],
            "oof_mean_dist": metrics["mean_distance"],
            **delta_summary(pred_test, winner_test, "test_vs_winner"),
            "selected_frac": 1.0,
            "path": str(path),
        }
        rows.append(row)
        print(f"  saved={path.name} oof_hit={metrics['r_hit_1cm']:.6f}", flush=True)

    selective_specs = [
        ("mean_k64_p2", 0.05, 0.25),
        ("mean_k64_p2", 0.09, 0.35),
        ("median_k96_p2", 0.07, 0.30),
        ("trimmean_k128_p15", 0.10, 0.25),
    ]
    for key, top_frac, alpha in selective_specs:
        cache = analog_cache[key]
        score_oof = confidence_score(cache["oof_pred"], cv_oof, cache["oof_info"])
        score_test = confidence_score(cache["test_pred"], winner_test, cache["test_info"])
        pred_oof, mask_oof = selective_move(cv_oof, cache["oof_pred"], score_oof, top_frac, alpha)
        pred_test, mask_test = selective_move(winner_test, cache["test_pred"], score_test, top_frac, alpha)
        path = submission_dir / (
            f"{EXP_TAG}_winner_move_{key}_top{int(top_frac * 1000):03d}_a{int(alpha * 100):02d}.csv"
        )
        write_submission(sample_submission, pred_test, path)
        metrics = distance_summary(pred_oof, y)
        row = {
            "candidate": path.name,
            "kind": "winner_selective_move",
            "oof_hit": metrics["r_hit_1cm"],
            "oof_mean_dist": metrics["mean_distance"],
            **delta_summary(pred_test, winner_test, "test_vs_winner"),
            "selected_frac": float(mask_test.mean()),
            "path": str(path),
        }
        rows.append(row)
        print(
            f"  saved={path.name} proxy_oof_hit={metrics['r_hit_1cm']:.6f} "
            f"selected={mask_test.mean():.3f}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    kind_bonus = result["kind"].map(
        {
            "winner_selective_move": 0.0020,
            "physics_blend": 0.0008,
            "analog_only": 0.0000,
        }
    ).fillna(0.0)
    result["recommend_score"] = (
        result["oof_hit"]
        + kind_bonus
        - 0.08 * result["test_vs_winner_mean_delta"]
        - 0.02 * result["test_vs_winner_p95_delta"]
    )
    result = result.sort_values(["recommend_score", "oof_hit"], ascending=[False, False]).reset_index(drop=True)
    result.insert(0, "recommend_rank", np.arange(1, len(result) + 1))

    display_cols = [
        "recommend_rank",
        "candidate",
        "kind",
        "oof_hit",
        "oof_mean_dist",
        "test_vs_winner_mean_delta",
        "test_vs_winner_p95_delta",
        "test_vs_winner_max_delta",
        "selected_frac",
        "recommend_score",
    ]
    print(result[display_cols].to_string(index=False), flush=True)

    report_path = report_dir / f"{EXP_TAG}.md"
    metrics_path = report_dir / f"{EXP_TAG}_metrics.json"
    report = [
        f"# {EXP_TAG}",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- public_anchor: `{PUBLIC_WINNER} = {PUBLIC_WINNER_SCORE:.5f}`",
        "- axis: orientation-equivariant analog transport",
        "",
        "## Idea",
        "",
        "- Convert each trajectory to a final-velocity local frame.",
        "- Retrieve similar train trajectory shapes after translation/orientation/scale normalization.",
        "- Reuse neighbors' +80ms local displacement distribution and transport it to each test frame.",
        "- Also create winner-anchored selective moves as low-risk probes.",
        "",
        "## Physics Proxy Baselines",
        "",
        f"- CV hit: `{hit_rate(cv_oof, y):.6f}`",
        f"- CA hit: `{hit_rate(ca_oof, y):.6f}`",
        "",
        "## Recommended Order",
        "",
        dataframe_to_markdown(result[display_cols].head(8)),
        "",
        "## All Candidates",
        "",
        dataframe_to_markdown(result[display_cols + ["path"]]),
        "",
        "## Decision Rule",
        "",
        "- Pure analog candidates are risky if their OOF hit is below the current physics baseline.",
        "- If winner-selective probes drop, analog transport does not transfer to public and should be abandoned.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    result.to_json(metrics_path, orient="records", force_ascii=False, indent=2)
    print(f"Wrote report: {report_path}", flush=True)
    print(f"Wrote metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
