from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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
from run_hit_weighted_local_frame import (
    make_features,
    make_lgbm,
    physics_poly_candidates,
    fit_predict_axes_weighted,
)
from run_local_frame_residual import local_basis, to_global, to_local


CV_SEEDS = [42, 777, 2026]
FULL_SEEDS = [42, 777, 2026, 3407, 10007]
WEIGHT_CONFIGS = [
    ("base_a3_s0035", "base", 3.0, 0.0035, 0.0100),
    ("base_a4_s0040", "base", 4.0, 0.0040, 0.0100),
    ("base_a5_s0045", "base", 5.0, 0.0045, 0.0100),
    ("base_a4_s0035_c0095", "base", 4.0, 0.0035, 0.0095),
    ("base_a4_s0045_c0105", "base", 4.0, 0.0045, 0.0105),
    ("near_hit_band", "band", 2.0, 0.0, 0.0),
]
SHRINKS = [
    (0.44, 0.58, 0.70),
    (0.46, 0.58, 0.70),
    (0.48, 0.58, 0.70),
    (0.46, 0.55, 0.62),
    (0.46, 0.58, 0.62),
    (0.46, 0.58, 0.78),
    (0.52, 0.58, 0.70),
]


def sample_weights_param(coords: np.ndarray, y: np.ndarray, config: tuple[str, str, float, float, float]) -> np.ndarray:
    _, mode, amplitude, sigma, center = config
    base_dist = np.linalg.norm(physics_prediction(coords, 1.0, 0.275) - y, axis=1)
    candidate_min_dist = np.linalg.norm(physics_poly_candidates(coords) - y[:, None, :], axis=2).min(axis=1)

    if mode == "base":
        weights = 1.0 + amplitude * np.exp(-0.5 * ((base_dist - center) / sigma) ** 2)
    elif mode == "band":
        weights = 1.0 + 2.0 * (base_dist <= 0.018) + 2.0 * (candidate_min_dist <= 0.012)
    else:
        raise ValueError(f"unknown weight config mode: {mode}")

    weights = np.clip(weights, 0.5, 8.0)
    return weights / np.mean(weights)


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    base = physics_prediction(coords, 1.0, 0.275)
    basis = local_basis(coords)
    residual_local = to_local(y - base, basis)
    rows = []

    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        y_val = y[val_mask]
        base_val = base[val_mask]
        basis_val = basis[val_mask]

        for config in WEIGHT_CONFIGS:
            config_name = config[0]
            weights = sample_weights_param(coords[train_mask], y[train_mask], config)
            pred_local = fit_predict_axes_weighted(
                features[train_mask],
                residual_local[train_mask],
                features[val_mask],
                seed,
                "l2",
                weights,
            )
            for forward, side, up in SHRINKS:
                shrink = np.array([forward, side, up], dtype=np.float64)
                pred = base_val + to_global(pred_local * shrink[None, :], basis_val)
                rows.append(
                    {
                        "weight_config": config_name,
                        "seed": seed,
                        "forward_shrink": forward,
                        "side_shrink": side,
                        "up_shrink": up,
                        **distance_summary(pred, y_val),
                    }
                )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(["weight_config", "forward_shrink", "side_shrink", "up_shrink"], as_index=False)
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


def full_local_prediction(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    test_features: np.ndarray,
    config: tuple[str, str, float, float, float],
) -> np.ndarray:
    base = physics_prediction(coords, 1.0, 0.275)
    residual_local = to_local(y - base, local_basis(coords))
    weights = sample_weights_param(coords, y, config)
    seed_preds = []
    for seed in FULL_SEEDS:
        seed_preds.append(fit_predict_axes_weighted(features, residual_local, test_features, seed, "l2", weights))
    return np.mean(seed_preds, axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine the 0.671 hit-weighted local-frame breakthrough.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_hit_weighted_breakthrough_refine.md")
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

    print("Building hit-weighted breakthrough features", flush=True)
    train_features, candidate_names = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    print(f"feature_count={train_features.shape[1]}", flush=True)

    print("Evaluating breakthrough refine CV", flush=True)
    leaderboard = evaluate_cv(train_coords, y, train_features, args.val_frac)
    top = leaderboard.head(30)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    config_by_name = {config[0]: config for config in WEIGHT_CONFIGS}
    base_test = physics_prediction(test_coords, 1.0, 0.275)
    test_basis = local_basis(test_coords)
    cache: dict[str, np.ndarray] = {}
    written = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        config_name = str(row["weight_config"])
        if config_name not in cache:
            print(f"Training full 5-seed model: {config_name}", flush=True)
            cache[config_name] = full_local_prediction(train_coords, y, train_features, test_features, config_by_name[config_name])
        shrink = np.array([row["forward_shrink"], row["side_shrink"], row["up_shrink"]], dtype=np.float64)
        pred = base_test + to_global(cache[config_name] * shrink[None, :], test_basis)
        path = (
            args.output_dir
            / f"hit_breakthrough_rank{rank}_{slug(config_name)}_f{shrink[0]:.2f}_s{shrink[1]:.2f}_u{shrink[2]:.2f}_5seed.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Hit-Weighted Breakthrough Refine",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        "- Public anchor: `hit_weighted_rank1_l2_base_boundary_f0.46_s0.58_u0.70.csv = 0.671`",
        f"- Feature count: `{train_features.shape[1]}`",
        f"- Weight configs: `{WEIGHT_CONFIGS}`",
        f"- Shrinks: `{SHRINKS}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full ensemble seeds: `{FULL_SEEDS}`",
        f"- Candidate feature families: `{candidate_names}`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Top 30 Breakthrough Refine Configs",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- This stabilizes the 0.671 breakthrough with 3-seed CV and 5-seed full predictions.",
        "- If the 5-seed anchor underperforms, keep the 3-seed public winner as the current anchor and explore new weight functions.",
        "- If it improves, make hit-boundary weighting the main modeling branch.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()

