from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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

from mosquito_trajectory.data import (  # noqa: E402
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, split_mask, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import curvature_correction, read_submission_coords, write_submission  # noqa: E402
from run_hit_weighted_local_frame import make_features  # noqa: E402
from run_local_target_manifold_projection_20260521 import build_champion_oof  # noqa: E402
from run_regime_miss_policy_20260524 import CHAMPION, CURRENT_SCORE, R_HIT, regime_features  # noqa: E402


CV_SEEDS = [42, 777, 2026]


@dataclass(frozen=True)
class AnalogConfig:
    k: int
    power: float
    shrink: float
    cap: float
    route_fraction: float


CONFIGS = [
    AnalogConfig(12, 1.0, 0.10, 0.0015, 0.08),
    AnalogConfig(16, 1.0, 0.12, 0.0020, 0.10),
    AnalogConfig(24, 1.0, 0.14, 0.0025, 0.12),
    AnalogConfig(32, 1.5, 0.16, 0.0030, 0.16),
    AnalogConfig(48, 1.5, 0.18, 0.0035, 0.20),
    AnalogConfig(64, 2.0, 0.20, 0.0040, 0.24),
    AnalogConfig(32, 1.0, 0.10, 0.0020, 1.00),
    AnalogConfig(64, 1.0, 0.12, 0.0025, 1.00),
    AnalogConfig(96, 1.0, 0.14, 0.0030, 1.00),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def analog_features(coords: np.ndarray, base_features: np.ndarray) -> np.ndarray:
    reg = regime_features(coords)
    diffs = np.diff(coords, axis=1)
    last = coords[:, -1, :]
    rel = coords[:, -6:, :] - last[:, None, :]
    recent_diffs = diffs[:, -6:, :]
    compact = np.hstack(
        [
            reg,
            rel.reshape(len(coords), -1),
            recent_diffs.reshape(len(coords), -1),
            base_features,
        ]
    )
    return compact.astype(np.float32)


def cap_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / (norm + 1e-12))
    return vectors * scale


def weighted_neighbor_residual(
    distances: np.ndarray,
    indices: np.ndarray,
    train_residuals: np.ndarray,
    config: AnalogConfig,
) -> tuple[np.ndarray, np.ndarray]:
    d = distances[:, : config.k]
    idx = indices[:, : config.k]
    weights = 1.0 / np.maximum(d, 1e-6) ** config.power
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    neigh_resid = train_residuals[idx]
    residual = np.sum(neigh_resid * weights[:, :, None], axis=1)
    spread = np.sum(np.linalg.norm(neigh_resid - residual[:, None, :], axis=2) * weights, axis=1)
    mean_dist = np.sum(d * weights, axis=1)
    uncertainty = mean_dist * (1.0 + 25.0 * spread)
    return residual, uncertainty


def route_mask_from_uncertainty(uncertainty: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 0.999:
        return np.ones(len(uncertainty), dtype=bool)
    n_route = max(1, int(round(len(uncertainty) * fraction)))
    order = np.argsort(uncertainty)
    mask = np.zeros(len(uncertainty), dtype=bool)
    mask[order[:n_route]] = True
    return mask


def apply_residual(
    champion: np.ndarray,
    residual: np.ndarray,
    uncertainty: np.ndarray,
    config: AnalogConfig,
) -> tuple[np.ndarray, np.ndarray]:
    pred = champion.copy()
    mask = route_mask_from_uncertainty(uncertainty, config.route_fraction)
    update = cap_vectors(config.shrink * residual, config.cap)
    pred[mask] = champion[mask] + update[mask]
    return pred, mask


def load_or_build_champion_oof(
    cache_path: Path,
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    anchor_oof: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path)
        if len(data["champion_oof"]) == len(coords):
            return data["champion_oof"], data["gate_proba"]
    champion_oof, gate_proba = build_champion_oof(coords, y, features, anchor_oof, correction)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, champion_oof=champion_oof, gate_proba=gate_proba)
    return champion_oof, gate_proba


def evaluate_cv(
    features: np.ndarray,
    champion_oof: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
) -> pd.DataFrame:
    max_k = max(config.k for config in CONFIGS)
    rows = []
    for seed in CV_SEEDS:
        val_mask = split_mask(len(y), 0.2, seed)
        train_mask = ~val_mask
        scaler = StandardScaler()
        train_x = scaler.fit_transform(features[train_mask])
        val_x = scaler.transform(features[val_mask])
        model = NearestNeighbors(n_neighbors=max_k, metric="euclidean", algorithm="auto")
        model.fit(train_x)
        distances, local_indices = model.kneighbors(val_x, return_distance=True)
        train_indices = np.where(train_mask)[0]
        neighbor_indices = train_indices[local_indices]
        baseline = distance_summary(champion_oof[val_mask], y[val_mask])["r_hit_1cm"]

        for config_id, config in enumerate(CONFIGS, start=1):
            residual, uncertainty = weighted_neighbor_residual(distances, neighbor_indices, residuals, config)
            pred, mask = apply_residual(champion_oof[val_mask], residual, uncertainty, config)
            row = {
                "config_id": config_id,
                "seed": seed,
                "k": config.k,
                "power": config.power,
                "shrink": config.shrink,
                "cap": config.cap,
                "route_fraction": config.route_fraction,
                "actual_route_fraction": float(np.mean(mask)),
                **distance_summary(pred, y[val_mask]),
                **delta_summary(pred, champion_oof[val_mask], "vs_champion"),
            }
            row["baseline_r_hit"] = baseline
            row["delta_hit_vs_baseline"] = row["r_hit_1cm"] - baseline
            rows.append(row)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["config_id", "k", "power", "shrink", "cap", "route_fraction"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_baseline_r_hit=("baseline_r_hit", "mean"),
            mean_delta_hit=("delta_hit_vs_baseline", "mean"),
            min_delta_hit=("delta_hit_vs_baseline", "min"),
            mean_distance=("mean_distance", "mean"),
            actual_route_fraction=("actual_route_fraction", "mean"),
            vs_champion_mean_delta=("vs_champion_mean_delta", "mean"),
            vs_champion_p95_delta=("vs_champion_p95_delta", "mean"),
            vs_champion_max_delta=("vs_champion_max_delta", "mean"),
        )
        .sort_values(["mean_delta_hit", "min_delta_hit", "mean_r_hit"], ascending=[False, False, False])
    )
    return summary


def fit_full_predict(
    train_features: np.ndarray,
    test_features: np.ndarray,
    residuals: np.ndarray,
    champion_test: np.ndarray,
    config: AnalogConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features)
    test_x = scaler.transform(test_features)
    model = NearestNeighbors(n_neighbors=max(config.k, 2), metric="euclidean", algorithm="auto")
    model.fit(train_x)
    distances, indices = model.kneighbors(test_x, return_distance=True)
    residual, uncertainty = weighted_neighbor_residual(distances, indices, residuals, config)
    return (*apply_residual(champion_test, residual, uncertainty, config), uncertainty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analog KNN residual transfer around the champion prediction.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--champion-cache", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_analog_knn_residual_20260524.md")
    parser.add_argument("--top-k", type=int, default=3)
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

    print("Building analog features and champion proxy", flush=True)
    base_features, _ = make_features(train_coords)
    test_base_features, _ = make_features(test_coords)
    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    _, _, train_correction = curvature_correction(train_coords)
    champion_oof, _ = load_or_build_champion_oof(args.champion_cache, train_coords, y, base_features, anchor_oof, train_correction)
    champion_test = read_submission_coords(args.submission_dir / CHAMPION)

    features = analog_features(train_coords, base_features)
    test_features = analog_features(test_coords, test_base_features)
    residuals = y - champion_oof
    champion_diag = pd.DataFrame([{"name": "champion_oof_proxy", **distance_summary(champion_oof, y)}])

    print("Evaluating analog residual CV", flush=True)
    leaderboard = evaluate_cv(features, champion_oof, y, residuals)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    for _, row in leaderboard.iterrows():
        config = CONFIGS[int(row["config_id"]) - 1]
        pred, mask, uncertainty = fit_full_predict(features, test_features, residuals, champion_test, config)
        rank = len(written) + 1
        path = args.submission_dir / (
            f"analogknn_rank{rank}_k{config.k}_p{slug(config.power)}"
            f"_s{int(round(config.shrink * 100)):03d}_cap{slug(config.cap)}_r{int(round(config.route_fraction * 100)):03d}.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "config_id": int(row["config_id"]),
                "cv_mean_delta_hit": float(row["mean_delta_hit"]),
                "cv_min_delta_hit": float(row["min_delta_hit"]),
                "cv_mean_route_fraction": float(row["actual_route_fraction"]),
                "test_route_fraction": float(np.mean(mask)),
                "test_uncertainty_p50": float(np.quantile(uncertainty, 0.50)),
                "test_uncertainty_p95": float(np.quantile(uncertainty, 0.95)),
                **delta_summary(pred, champion_test, "vs_champion"),
            }
        )
        if len(written) >= args.top_k:
            break

    output_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-24 Analog KNN Residual",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- champion: `{CHAMPION} = {CURRENT_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Start over with analog forecasting: find train trajectories with similar recent motion and transfer their champion residuals.",
        "- The update is heavily shrunk and capped, because the target is 1cm hit-rate and wrong residual transfer can easily destroy good champion hits.",
        "- Route fractions test two modes: only high-confidence analog neighborhoods versus applying a tiny residual everywhere.",
        "",
        "## Champion OOF Proxy",
        "",
        dataframe_to_markdown(champion_diag),
        "",
        "## CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(output_df),
        "",
        "## Notes",
        "",
        "- This is a genuinely new axis versus previous curvature/post-process probes.",
        "- If public score drops, the miss residual is not locally transferable by recent trajectory shape; the useful next step would be regime-specific pseudo-labeling rather than KNN residual.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
