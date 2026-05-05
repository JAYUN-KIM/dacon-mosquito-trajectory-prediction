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
FORWARD_SHRINKS = [0.40, 0.48, 0.55, 0.62, 0.70, 0.80]
SIDE_SHRINKS = [0.40, 0.48, 0.55, 0.62, 0.70, 0.80, 0.95]
UP_SHRINKS = [0.40, 0.48, 0.55, 0.62, 0.70, 0.80, 0.95]


def shrink_grid() -> list[tuple[float, float, float]]:
    rows = []
    for forward in FORWARD_SHRINKS:
        for side in SIDE_SHRINKS:
            for up in UP_SHRINKS:
                rows.append((forward, side, up))
    return rows


def evaluate_axis_shrink_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    base = physics_prediction(coords, 1.0, 0.275)
    basis = local_basis(coords)
    residual_local = to_local(y - base, basis)
    grid = shrink_grid()
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
    return (
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


def full_local_prediction(features: np.ndarray, residual_local: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    return np.mean([fit_predict_axes(features, residual_local, test_features, seed) for seed in FULL_SEEDS], axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search axis-wise shrink for local-frame residual predictions.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_local_frame_axis_shrink.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
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

    print("Building local-frame features")
    train_features, candidate_names = make_local_frame_features(train_coords)
    test_features, _ = make_local_frame_features(test_coords)
    print(f"feature_count={train_features.shape[1]}")

    print("Evaluating axis-wise shrink CV")
    leaderboard = evaluate_axis_shrink_cv(train_coords, y, train_features, args.val_frac)
    top = leaderboard.head(20)
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
            / f"local_axis_rank{rank}_f{shrink[0]:.2f}_s{shrink[1]:.2f}_u{shrink[2]:.2f}.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    report = [
        "# Local-Frame Axis Shrink",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- Feature count: `{train_features.shape[1]}`",
        f"- Candidate feature families: `{candidate_names}`",
        f"- CV seeds: `{CV_SEEDS}`",
        f"- Full ensemble seeds: `{FULL_SEEDS}`",
        f"- Written submissions: `{[str(path) for path in written]}`",
        "",
        "## Top 20 Axis Shrinks",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- Public 0.659 confirmed the local-frame target is a strong direction.",
        "- This experiment calibrates forward, side, and up residual corrections separately instead of using one scalar shrink.",
        "- If one axis consistently wants a larger shrink, the next step is separate model capacity and features by local axis.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}")
    for path in written:
        print(f"Wrote submission: {path}")


if __name__ == "__main__":
    main()

