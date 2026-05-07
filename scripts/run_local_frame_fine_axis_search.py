from __future__ import annotations

import argparse
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
from run_local_frame_residual import fit_predict_axes, local_basis, make_local_frame_features, to_global, to_local


CV_SEEDS = [42, 777, 2026, 3407, 10007]
FULL_SEEDS = [42, 777, 2026, 3407, 10007]
DEFAULT_FORWARD_SHRINKS = "0.42,0.44,0.46,0.48,0.50,0.52,0.55,0.58"
DEFAULT_SIDE_SHRINKS = "0.44,0.48,0.50,0.52,0.55,0.58,0.60,0.62,0.66"
DEFAULT_UP_SHRINKS = "0.58,0.62,0.66,0.70,0.74,0.78,0.82,0.88,0.95"


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("shrink list must contain at least one value")
    return values


def shrink_grid(
    forward_values: list[float], side_values: list[float], up_values: list[float]
) -> list[tuple[float, float, float]]:
    return [(forward, side, up) for forward in forward_values for side in side_values for up in up_values]


def evaluate_axis_shrink_cv(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    val_frac: float,
    grid: list[tuple[float, float, float]],
) -> pd.DataFrame:
    base = physics_prediction(coords, 1.0, 0.275)
    basis = local_basis(coords)
    residual_local = to_local(y - base, basis)
    rows = []

    for seed in CV_SEEDS:
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        pred_local = fit_predict_axes(features[train_mask], residual_local[train_mask], features[val_mask], seed)
        y_val = y[val_mask]
        base_val = base[val_mask]
        basis_val = basis[val_mask]

        for forward, side, up in grid:
            shrink = np.array([forward, side, up], dtype=np.float64)
            pred = base_val + to_global(pred_local * shrink[None, :], basis_val)
            rows.append(
                {
                    "seed": seed,
                    "forward_shrink": forward,
                    "side_shrink": side,
                    "up_shrink": up,
                    **distance_summary(pred, y_val),
                }
            )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(["forward_shrink", "side_shrink", "up_shrink"], as_index=False)
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


def full_local_prediction(features: np.ndarray, residual_local: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    return np.mean([fit_predict_axes(features, residual_local, test_features, seed) for seed in FULL_SEEDS], axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine search axis-wise shrink around the local-frame public best.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_local_frame_fine_axis_search.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--forward-shrinks", type=str, default=DEFAULT_FORWARD_SHRINKS)
    parser.add_argument("--side-shrinks", type=str, default=DEFAULT_SIDE_SHRINKS)
    parser.add_argument("--up-shrinks", type=str, default=DEFAULT_UP_SHRINKS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forward_values = parse_float_list(args.forward_shrinks)
    side_values = parse_float_list(args.side_shrinks)
    up_values = parse_float_list(args.up_shrinks)
    grid = shrink_grid(forward_values, side_values, up_values)

    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}")
    print(f"Fine shrink grid size={len(grid)}")

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

    print("Building local-frame features")
    train_features, candidate_names = make_local_frame_features(train_coords)
    test_features, _ = make_local_frame_features(test_coords)
    print(f"feature_count={train_features.shape[1]}")

    print("Evaluating fine axis-wise shrink CV")
    leaderboard = evaluate_axis_shrink_cv(train_coords, y, train_features, args.val_frac, grid)
    top = leaderboard.head(30)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    base_train = physics_prediction(train_coords, 1.0, 0.275)
    base_test = physics_prediction(test_coords, 1.0, 0.275)
    train_basis = local_basis(train_coords)
    test_basis = local_basis(test_coords)
    residual_local = to_local(y - base_train, train_basis)

    print("Training full local-frame ensemble")
    test_residual_local = full_local_prediction(train_features, residual_local, test_features)

    written = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        shrink = np.array([row["forward_shrink"], row["side_shrink"], row["up_shrink"]], dtype=np.float64)
        pred = base_test + to_global(test_residual_local * shrink[None, :], test_basis)
        path = (
            args.output_dir
            / f"local_axis_fine_rank{rank}_f{shrink[0]:.2f}_s{shrink[1]:.2f}_u{shrink[2]:.2f}.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Local-Frame Fine Axis Search",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- Feature count: `{train_features.shape[1]}`",
        f"- Grid size: `{len(grid)}`",
        f"- Forward shrinks: `{forward_values}`",
        f"- Side shrinks: `{side_values}`",
        f"- Up shrinks: `{up_values}`",
        f"- Candidate feature families: `{candidate_names}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full ensemble seeds: `{FULL_SEEDS}`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Top 30 Fine Axis Shrinks",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- This is the first 2026-05-07 exploit experiment.",
        "- The grid is centered around the 2026-05-06 public best `f0.48_s0.55_u0.62`.",
        "- Use the top one or two candidates first; reserve later submissions for basis and hit-aware experiments.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}")
    for path in written:
        print(f"Wrote submission: {path}")


if __name__ == "__main__":
    main()

