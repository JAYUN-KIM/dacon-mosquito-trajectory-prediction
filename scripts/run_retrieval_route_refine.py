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
from run_local_frame_residual import fit_predict_axes, local_basis, make_local_frame_features, to_global, to_local
from run_trajectory_retrieval import aggregate_neighbors, make_retrieval_features, target_values


CV_SEEDS = [42, 777, 2026, 3407, 10007]
FULL_SEEDS = [42, 777, 2026, 3407, 10007]
LOCAL_CONFIGS = [
    ("axis_fine_048_052_070", (0.48, 0.52, 0.70)),
    ("axis_best_048_055_062", (0.48, 0.55, 0.62)),
]
RETRIEVAL_CONFIGS = [
    ("retr_k50_softmax075", "local_motion", "local_residual", 50, "softmax0.75"),
    ("retr_k50_inverse", "local_motion", "local_residual", 50, "inverse"),
    ("retr_k32_softmax075", "local_motion", "local_residual", 32, "softmax0.75"),
]
BLEND_WEIGHTS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.36]
ROUTE_FRACTIONS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32]
CONFIDENCE_MODES = ["neighbor_mean", "neighbor_min", "neighbor_k", "agreement", "combo_mean_agreement"]


def shrink_local_prediction(coords: np.ndarray, pred_local: np.ndarray, shrink: tuple[float, float, float]) -> np.ndarray:
    shrink_array = np.array(shrink, dtype=np.float64)
    return physics_prediction(coords, 1.0, 0.275) + to_global(pred_local * shrink_array[None, :], local_basis(coords))


def predict_retrieval_with_scores(
    train_coords: np.ndarray,
    train_y: np.ndarray,
    pred_coords: np.ndarray,
    train_features: np.ndarray,
    pred_features: np.ndarray,
    target_mode: str,
    k: int,
    weight_mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features)
    pred_x = scaler.transform(pred_features)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(train_x)
    distances, indices = nn.kneighbors(pred_x, return_distance=True)
    train_values = target_values(train_coords, train_y, target_mode)
    aggregated = aggregate_neighbors(train_values, indices, distances, k, weight_mode)

    if target_mode == "global_residual":
        pred = physics_prediction(pred_coords, 1.0, 0.275) + aggregated
    elif target_mode == "local_residual":
        pred = physics_prediction(pred_coords, 1.0, 0.275) + to_global(aggregated, local_basis(pred_coords))
    elif target_mode == "local_offset":
        pred = pred_coords[:, -1, :] + to_global(aggregated, local_basis(pred_coords))
    else:
        raise ValueError(f"unknown target mode: {target_mode}")

    scores = {
        "neighbor_mean": distances.mean(axis=1),
        "neighbor_min": distances[:, 0],
        "neighbor_k": distances[:, -1],
    }
    return pred, scores


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks


def confidence_score(
    mode: str,
    neighbor_scores: dict[str, np.ndarray],
    local_pred: np.ndarray,
    retrieval_pred: np.ndarray,
) -> np.ndarray:
    agreement = np.linalg.norm(local_pred - retrieval_pred, axis=1)
    if mode == "agreement":
        return agreement
    if mode == "combo_mean_agreement":
        return 0.65 * rank01(neighbor_scores["neighbor_mean"]) + 0.35 * rank01(agreement)
    if mode in neighbor_scores:
        return neighbor_scores[mode]
    raise ValueError(f"unknown confidence mode: {mode}")


def route_blend(
    local_pred: np.ndarray,
    retrieval_pred: np.ndarray,
    score: np.ndarray,
    fraction: float,
    blend_weight: float,
) -> np.ndarray:
    cutoff = np.quantile(score, fraction)
    selected = score <= cutoff
    pred = local_pred.copy()
    pred[selected] = (1.0 - blend_weight) * local_pred[selected] + blend_weight * retrieval_pred[selected]
    return pred


def evaluate_cv(
    train_coords: np.ndarray,
    y: np.ndarray,
    local_features: np.ndarray,
    retrieval_features: np.ndarray,
    val_frac: float,
) -> pd.DataFrame:
    base = physics_prediction(train_coords, 1.0, 0.275)
    residual_local = to_local(y - base, local_basis(train_coords))
    rows = []

    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(train_coords), val_frac, seed)
        fit_mask = ~val_mask
        coords_val = train_coords[val_mask]
        y_val = y[val_mask]

        pred_local_residual = fit_predict_axes(
            local_features[fit_mask],
            residual_local[fit_mask],
            local_features[val_mask],
            seed,
        )
        local_preds = {
            local_name: shrink_local_prediction(coords_val, pred_local_residual, shrink)
            for local_name, shrink in LOCAL_CONFIGS
        }

        retrieval_cache = {}
        for retr_name, _, target_mode, k, weight_mode in RETRIEVAL_CONFIGS:
            retrieval_cache[retr_name] = predict_retrieval_with_scores(
                train_coords[fit_mask],
                y[fit_mask],
                coords_val,
                retrieval_features[fit_mask],
                retrieval_features[val_mask],
                target_mode,
                k,
                weight_mode,
            )

        for local_name, local_pred in local_preds.items():
            rows.append(
                {
                    "strategy": "local_only",
                    "local_config": local_name,
                    "retrieval_config": "none",
                    "confidence_mode": "none",
                    "weight": 0.0,
                    "route_fraction": 0.0,
                    "seed": seed,
                    **distance_summary(local_pred, y_val),
                }
            )

            for retr_name, (retrieval_pred, neighbor_scores) in retrieval_cache.items():
                for confidence_mode in CONFIDENCE_MODES:
                    score = confidence_score(confidence_mode, neighbor_scores, local_pred, retrieval_pred)
                    for route_fraction in ROUTE_FRACTIONS:
                        for blend_weight in BLEND_WEIGHTS:
                            pred = route_blend(local_pred, retrieval_pred, score, route_fraction, blend_weight)
                            rows.append(
                                {
                                    "strategy": "route_blend",
                                    "local_config": local_name,
                                    "retrieval_config": retr_name,
                                    "confidence_mode": confidence_mode,
                                    "weight": blend_weight,
                                    "route_fraction": route_fraction,
                                    "seed": seed,
                                    **distance_summary(pred, y_val),
                                }
                            )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(
            ["strategy", "local_config", "retrieval_config", "confidence_mode", "weight", "route_fraction"],
            as_index=False,
        )
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


def full_local_predictions(
    train_coords: np.ndarray,
    y: np.ndarray,
    test_coords: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> dict[str, np.ndarray]:
    base_train = physics_prediction(train_coords, 1.0, 0.275)
    residual_local = to_local(y - base_train, local_basis(train_coords))
    test_residual_local = np.mean(
        [fit_predict_axes(train_features, residual_local, test_features, seed) for seed in FULL_SEEDS],
        axis=0,
    )
    return {name: shrink_local_prediction(test_coords, test_residual_local, shrink) for name, shrink in LOCAL_CONFIGS}


def build_test_retrievals(
    train_coords: np.ndarray,
    y: np.ndarray,
    test_coords: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    out = {}
    for retr_name, _, target_mode, k, weight_mode in RETRIEVAL_CONFIGS:
        print(f"Predicting full retrieval: {retr_name}", flush=True)
        out[retr_name] = predict_retrieval_with_scores(
            train_coords,
            y,
            test_coords,
            train_features,
            test_features,
            target_mode,
            k,
            weight_mode,
        )
    return out


def build_full_prediction(
    row: pd.Series,
    local_test: dict[str, np.ndarray],
    retrieval_test: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
) -> np.ndarray:
    if str(row["strategy"]) == "local_only":
        return local_test[str(row["local_config"])]
    local_pred = local_test[str(row["local_config"])]
    retrieval_pred, neighbor_scores = retrieval_test[str(row["retrieval_config"])]
    score = confidence_score(str(row["confidence_mode"]), neighbor_scores, local_pred, retrieval_pred)
    return route_blend(local_pred, retrieval_pred, score, float(row["route_fraction"]), float(row["weight"]))


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine route-blend search around the 0.6604 retrieval blend direction.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_retrieval_route_refine.md")
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

    print("Building local-frame residual features", flush=True)
    local_train_features, candidate_names = make_local_frame_features(train_coords)
    local_test_features, _ = make_local_frame_features(test_coords)

    print("Building retrieval features: local_motion", flush=True)
    retrieval_train_features = make_retrieval_features(train_coords, "local_motion")
    retrieval_test_features = make_retrieval_features(test_coords, "local_motion")

    print("Evaluating route refine CV", flush=True)
    leaderboard = evaluate_cv(train_coords, y, local_train_features, retrieval_train_features, args.val_frac)
    top = leaderboard.head(50)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Training full local-frame ensemble", flush=True)
    local_test = full_local_predictions(train_coords, y, test_coords, local_train_features, local_test_features)
    retrieval_test = build_test_retrievals(train_coords, y, test_coords, retrieval_train_features, retrieval_test_features)

    written = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        pred = build_full_prediction(row, local_test, retrieval_test)
        path = (
            args.output_dir
            / (
                f"retr_route_refine_rank{rank}_{slug(row['local_config'])}_{slug(row['retrieval_config'])}_"
                f"{slug(row['confidence_mode'])}_w{float(row['weight']):.2f}_r{float(row['route_fraction']):.2f}.csv"
            )
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Retrieval Route Refine",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        "- Public anchor: `retr_blend_rank1...w0.15_r0.20.csv = 0.6604`",
        f"- Local configs: `{LOCAL_CONFIGS}`",
        f"- Retrieval configs: `{RETRIEVAL_CONFIGS}`",
        f"- Confidence modes: `{CONFIDENCE_MODES}`",
        f"- Blend weights: `{BLEND_WEIGHTS}`",
        f"- Route fractions: `{ROUTE_FRACTIONS}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full local ensemble seeds: `{FULL_SEEDS}`",
        f"- Local feature families: `{candidate_names}`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Top 50 Route Refine Configs",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- This refines the public-improving selective retrieval route instead of trying retrieval as a standalone model.",
        "- The safest follow-up is a neighbor-distance route near the 0.15 weight / 0.20 fraction anchor.",
        "- Agreement-based confidence is included as an exploratory hedge; use it only if it beats the anchor in CV.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()

