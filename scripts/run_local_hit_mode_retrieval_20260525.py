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
from run_local_frame_residual import local_basis, to_global, to_local  # noqa: E402
from run_local_target_manifold_projection_20260521 import build_champion_oof  # noqa: E402
from run_trajectory_retrieval import make_retrieval_features  # noqa: E402


CURRENT_CHAMPION = "champmicro_rank3_gatet520a1025.csv"
CURRENT_CHAMPION_SCORE = 0.69140
FALLBACK_CHAMPION = "champalpha_rank1_t52a1015.csv"
R_HIT = 0.01
CV_SEEDS = [42, 777]


@dataclass(frozen=True)
class ModeConfig:
    feature_variant: str
    k: int
    target_sigma: float
    mode_kind: str


MODE_CONFIGS = [
    ModeConfig(feature_variant, k, sigma, mode_kind)
    for feature_variant in ["local_shape", "local_motion"]
    for k in [32, 64, 96]
    for sigma in [0.006, 0.008, 0.010]
    for mode_kind in ["medoid", "cluster_mean"]
]
ROUTE_FRACTIONS = [0.05, 0.10, 0.18, 0.30]
BLENDS = [0.08, 0.12, 0.18]
CAPS = [0.0008, 0.0012, 0.0018]


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


def clip_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / (norm + 1e-12))
    return vectors * scale


def local_offset_targets(coords: np.ndarray, y: np.ndarray) -> np.ndarray:
    return to_local(y - coords[:, -1, :], local_basis(coords))


def materialize_local_offsets(coords: np.ndarray, local_offsets: np.ndarray) -> np.ndarray:
    return coords[:, -1, :] + to_global(local_offsets, local_basis(coords))


def hit_mode_from_neighbors(
    train_values: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    config: ModeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    idx = neighbor_indices[:, : config.k]
    feature_d = neighbor_distances[:, : config.k]
    values = train_values[idx]

    local_scale = np.maximum(np.median(feature_d, axis=1, keepdims=True), 1e-6)
    feature_w = np.exp(-0.5 * (feature_d / local_scale) ** 2)
    feature_w = feature_w / np.maximum(feature_w.sum(axis=1, keepdims=True), 1e-12)

    target_diff = values[:, :, None, :] - values[:, None, :, :]
    target_d = np.linalg.norm(target_diff, axis=3)
    target_kernel = np.exp(-0.5 * (target_d / config.target_sigma) ** 2)
    density = target_kernel @ feature_w[:, :, None]
    density = density[:, :, 0]
    best = np.argmax(density, axis=1)

    row_idx = np.arange(len(idx))
    medoid = values[row_idx, best]
    best_density = density[row_idx, best]
    sorted_density = np.sort(density, axis=1)
    margin = best_density - sorted_density[:, -2]
    near_count = np.sum(target_d[row_idx, best, :] <= R_HIT, axis=1)
    confidence = best_density + 0.15 * near_count + 2.0 * margin

    if config.mode_kind == "medoid":
        mode = medoid
    elif config.mode_kind == "cluster_mean":
        cluster_w = target_kernel[row_idx, best, :] * feature_w
        cluster_w = cluster_w / np.maximum(cluster_w.sum(axis=1, keepdims=True), 1e-12)
        mode = np.einsum("nk,nkc->nc", cluster_w, values)
    else:
        raise ValueError(f"unknown mode_kind: {config.mode_kind}")

    return mode.astype(np.float64), confidence.astype(np.float64)


def route_mask(confidence: np.ndarray, fraction: float) -> np.ndarray:
    n_route = max(1, int(round(len(confidence) * fraction)))
    order = np.argsort(confidence)[::-1]
    mask = np.zeros(len(confidence), dtype=bool)
    mask[order[:n_route]] = True
    return mask


def apply_mode_pull(
    champion: np.ndarray,
    mode_pred: np.ndarray,
    confidence: np.ndarray,
    route_fraction: float,
    blend: float,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred = champion.copy()
    mask = route_mask(confidence, route_fraction)
    move = clip_vectors(blend * (mode_pred - champion), cap)
    pred[mask] = champion[mask] + move[mask]
    return pred, mask


def load_or_build_gate_proba(
    cache_path: Path,
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    anchor_oof: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    if cache_path.exists():
        data = np.load(cache_path)
        if "gate_proba" in data and len(data["gate_proba"]) == len(coords):
            return data["gate_proba"]
    _, gate_proba = build_champion_oof(coords, y, features, anchor_oof, correction)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, gate_proba=gate_proba)
    return gate_proba


def champion_oof_alpha_down(anchor_oof: np.ndarray, correction: np.ndarray, gate_proba: np.ndarray, alpha: float = 0.1025) -> np.ndarray:
    pred = anchor_oof.copy()
    mask = gate_proba >= 0.52
    pred[mask] = anchor_oof[mask] + alpha * correction[mask]
    return pred


def evaluate_configs(
    coords: np.ndarray,
    y: np.ndarray,
    features_by_variant: dict[str, np.ndarray],
    champion_oof: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_values_all = local_offset_targets(coords, y)
    rows = []
    base_rows = []
    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(coords), 0.2, seed)
        train_mask = ~val_mask
        baseline_hit = distance_summary(champion_oof[val_mask], y[val_mask])["r_hit_1cm"]

        for feature_variant, features in features_by_variant.items():
            scaler = StandardScaler()
            train_x = scaler.fit_transform(features[train_mask])
            val_x = scaler.transform(features[val_mask])
            max_k = max(config.k for config in MODE_CONFIGS if config.feature_variant == feature_variant)
            nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean", n_jobs=-1)
            nn.fit(train_x)
            distances, local_indices = nn.kneighbors(val_x, return_distance=True)
            train_indices = np.where(train_mask)[0]
            neighbor_indices = train_indices[local_indices]

            for config in [cfg for cfg in MODE_CONFIGS if cfg.feature_variant == feature_variant]:
                mode_local, confidence = hit_mode_from_neighbors(train_values_all, neighbor_indices, distances, config)
                mode_pred = materialize_local_offsets(coords[val_mask], mode_local)
                mode_diag = distance_summary(mode_pred, y[val_mask])
                base_rows.append(
                    {
                        "seed": seed,
                        "feature_variant": config.feature_variant,
                        "k": config.k,
                        "target_sigma": config.target_sigma,
                        "mode_kind": config.mode_kind,
                        "mode_r_hit": mode_diag["r_hit_1cm"],
                        "mode_mean_distance": mode_diag["mean_distance"],
                        "confidence_p50": float(np.quantile(confidence, 0.50)),
                        "confidence_p95": float(np.quantile(confidence, 0.95)),
                    }
                )

                for route_fraction in ROUTE_FRACTIONS:
                    for blend in BLENDS:
                        for cap in CAPS:
                            pred, mask = apply_mode_pull(
                                champion_oof[val_mask],
                                mode_pred,
                                confidence,
                                route_fraction,
                                blend,
                                cap,
                            )
                            row = {
                                "seed": seed,
                                "feature_variant": config.feature_variant,
                                "k": config.k,
                                "target_sigma": config.target_sigma,
                                "mode_kind": config.mode_kind,
                                "route_fraction": route_fraction,
                                "blend": blend,
                                "cap": cap,
                                "actual_route_fraction": float(np.mean(mask)),
                                **distance_summary(pred, y[val_mask]),
                                **delta_summary(pred, champion_oof[val_mask], "vs_champion"),
                            }
                            row["baseline_r_hit"] = baseline_hit
                            row["delta_hit_vs_baseline"] = row["r_hit_1cm"] - baseline_hit
                            rows.append(row)

    cv = pd.DataFrame(rows)
    summary = (
        cv.groupby(
            ["feature_variant", "k", "target_sigma", "mode_kind", "route_fraction", "blend", "cap"],
            as_index=False,
        )
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_baseline_r_hit=("baseline_r_hit", "mean"),
            mean_delta_hit=("delta_hit_vs_baseline", "mean"),
            min_delta_hit=("delta_hit_vs_baseline", "min"),
            mean_distance=("mean_distance", "mean"),
            vs_champion_mean_delta=("vs_champion_mean_delta", "mean"),
            vs_champion_p95_delta=("vs_champion_p95_delta", "mean"),
        )
        .sort_values(["mean_delta_hit", "min_delta_hit", "mean_r_hit"], ascending=[False, False, False])
    )
    mode_summary = (
        pd.DataFrame(base_rows)
        .groupby(["feature_variant", "k", "target_sigma", "mode_kind"], as_index=False)
        .agg(
            mode_mean_r_hit=("mode_r_hit", "mean"),
            mode_min_r_hit=("mode_r_hit", "min"),
            mode_mean_distance=("mode_mean_distance", "mean"),
            confidence_p50=("confidence_p50", "mean"),
            confidence_p95=("confidence_p95", "mean"),
        )
        .sort_values(["mode_mean_r_hit", "mode_min_r_hit"], ascending=[False, False])
    )
    return summary, mode_summary


def predict_mode_test(
    train_coords: np.ndarray,
    y: np.ndarray,
    test_coords: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
    config: ModeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = local_offset_targets(train_coords, y)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features)
    test_x = scaler.transform(test_features)
    nn = NearestNeighbors(n_neighbors=config.k, metric="euclidean", n_jobs=-1)
    nn.fit(train_x)
    distances, indices = nn.kneighbors(test_x, return_distance=True)
    mode_local, confidence = hit_mode_from_neighbors(train_values, indices, distances, config)
    return materialize_local_offsets(test_coords, mode_local), confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hit-mode retrieval: choose local target modes instead of KNN means.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--champion-cache", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260525.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_local_hit_mode_retrieval_20260525.md")
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

    print("Building champion OOF and retrieval features", flush=True)
    base_features, _ = make_features(train_coords)
    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    _, _, train_correction = curvature_correction(train_coords)
    gate_proba = load_or_build_gate_proba(args.champion_cache, train_coords, y, base_features, anchor_oof, train_correction)
    champion_oof = champion_oof_alpha_down(anchor_oof, train_correction, gate_proba)
    champion_diag = pd.DataFrame([{"name": "champion_oof_alpha1025", **distance_summary(champion_oof, y)}])

    features_by_variant = {
        "local_shape": make_retrieval_features(train_coords, "local_shape"),
        "local_motion": make_retrieval_features(train_coords, "local_motion"),
    }
    print("Evaluating local hit-mode retrieval", flush=True)
    leaderboard, mode_leaderboard = evaluate_configs(train_coords, y, features_by_variant, champion_oof)
    print(leaderboard.head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    champion_path = args.submission_dir / CURRENT_CHAMPION
    if not champion_path.exists():
        champion_path = args.submission_dir / FALLBACK_CHAMPION
    champion_test = read_submission_coords(champion_path)
    test_features_by_variant = {
        "local_shape": make_retrieval_features(test_coords, "local_shape"),
        "local_motion": make_retrieval_features(test_coords, "local_motion"),
    }

    written = []
    output_rows = []
    used: set[tuple] = set()
    for _, row in leaderboard.iterrows():
        key = (
            str(row["feature_variant"]),
            int(row["k"]),
            float(row["target_sigma"]),
            str(row["mode_kind"]),
            float(row["route_fraction"]),
            float(row["blend"]),
            float(row["cap"]),
        )
        if key in used:
            continue
        used.add(key)
        config = ModeConfig(key[0], key[1], key[2], key[3])
        mode_pred, confidence = predict_mode_test(
            train_coords,
            y,
            test_coords,
            features_by_variant[config.feature_variant],
            test_features_by_variant[config.feature_variant],
            config,
        )
        pred, mask = apply_mode_pull(champion_test, mode_pred, confidence, key[4], key[5], key[6])
        rank = len(written) + 1
        path = args.submission_dir / (
            f"hitmode_rank{rank}_{slug(config.feature_variant)}"
            f"_k{config.k}_s{slug(config.target_sigma)}_{slug(config.mode_kind)}"
            f"_r{int(round(key[4] * 100)):02d}_b{int(round(key[5] * 100)):02d}_c{slug(key[6])}.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "cv_mean_delta_hit": float(row["mean_delta_hit"]),
                "cv_min_delta_hit": float(row["min_delta_hit"]),
                "test_route_fraction": float(np.mean(mask)),
                "confidence_p50": float(np.quantile(confidence, 0.50)),
                "confidence_p95": float(np.quantile(confidence, 0.95)),
                **delta_summary(pred, champion_test, "vs_champion"),
            }
        )
        if len(written) >= args.top_k:
            break

    report = [
        "# 2026-05-25 Local Hit-Mode Retrieval",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{champion_path.name} = {CURRENT_CHAMPION_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Restart from the competition definition: the metric is not mean distance, but whether the prediction falls inside a 1cm sphere.",
        "- Existing retrieval averaged neighbor futures, which estimates a conditional mean.",
        "- This experiment chooses a local target medoid or dense cluster mean among neighbors, approximating a conditional mode for R-Hit.",
        "- The mode prediction does not replace the champion globally; it only pulls high-confidence samples by a small capped vector.",
        "",
        "## Champion OOF Proxy",
        "",
        dataframe_to_markdown(champion_diag),
        "",
        "## Mode-Only Diagnostics",
        "",
        dataframe_to_markdown(mode_leaderboard.head(40)),
        "",
        "## Pull CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard.head(80)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- If this public score drops, local future modes are not separable enough from the observed 400ms shape.",
        "- If it ties or improves, expand by using richer hit-density features rather than larger coordinate moves.",
        "- This uses only train labels and raw test trajectories for inference; no test labels or external server models are used.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
