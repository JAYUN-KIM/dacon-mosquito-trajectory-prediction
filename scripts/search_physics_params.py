from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mosquito_trajectory.data import (
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)


def r_hit(pred: np.ndarray, true: np.ndarray, radius: float = 0.01) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=1) <= radius))


def distance_summary(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(pred - true, axis=1)
    return {
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "p90_distance": float(np.quantile(distances, 0.90)),
        "p95_distance": float(np.quantile(distances, 0.95)),
        "r_hit_1cm": float(np.mean(distances <= 0.01)),
    }


def split_ids(ids: list[str], val_frac: float, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(ids, dtype=object)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_frac)))
    return sorted(shuffled[:val_size].tolist())


def trajectory_arrays(samples: dict, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.stack([samples[sample_id].coords for sample_id in ids])
    times = np.stack([samples[sample_id].timesteps_ms for sample_id in ids])
    dt_last = (times[:, -1] - times[:, -2])[:, None]
    dt_prev = (times[:, -2] - times[:, -3])[:, None]
    v_last = (coords[:, -1, :] - coords[:, -2, :]) / dt_last
    v_prev = (coords[:, -2, :] - coords[:, -3, :]) / dt_prev
    accel = (v_last - v_prev) / ((dt_last + dt_prev) * 0.5)
    return coords[:, -1, :], v_last, accel


def evaluate_grid(last: np.ndarray, velocity: np.ndarray, accel: np.ndarray, true: np.ndarray) -> pd.DataFrame:
    rows = []
    delta_ms = 80.0

    for velocity_scale in np.round(np.arange(0.70, 1.301, 0.01), 2):
        pred = last + velocity * delta_ms * velocity_scale
        rows.append(
            {
                "family": "velocity_scale",
                "velocity_scale": float(velocity_scale),
                "accel_scale": 0.0,
                "blend_accel": 0.0,
                **distance_summary(pred, true),
            }
        )

    velocity_pred = last + velocity * delta_ms
    acceleration_pred = last + velocity * delta_ms + 0.5 * accel * delta_ms**2

    for blend_accel in np.round(np.arange(0.0, 1.001, 0.02), 2):
        pred = (1.0 - blend_accel) * velocity_pred + blend_accel * acceleration_pred
        rows.append(
            {
                "family": "velocity_accel_blend",
                "velocity_scale": 1.0,
                "accel_scale": 1.0,
                "blend_accel": float(blend_accel),
                **distance_summary(pred, true),
            }
        )

    for velocity_scale in np.round(np.arange(0.80, 1.201, 0.02), 2):
        for accel_scale in np.round(np.arange(-0.40, 0.601, 0.05), 2):
            pred = last + velocity * delta_ms * velocity_scale + 0.5 * accel * delta_ms**2 * accel_scale
            rows.append(
                {
                    "family": "velocity_accel_grid",
                    "velocity_scale": float(velocity_scale),
                    "accel_scale": float(accel_scale),
                    "blend_accel": 0.0,
                    **distance_summary(pred, true),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["r_hit_1cm", "mean_distance", "median_distance"],
        ascending=[False, True, True],
    )


def make_prediction(last: np.ndarray, velocity: np.ndarray, accel: np.ndarray, row: pd.Series) -> np.ndarray:
    delta_ms = 80.0
    family = str(row["family"])
    if family == "velocity_scale":
        return last + velocity * delta_ms * float(row["velocity_scale"])
    if family == "velocity_accel_blend":
        velocity_pred = last + velocity * delta_ms
        acceleration_pred = last + velocity * delta_ms + 0.5 * accel * delta_ms**2
        blend = float(row["blend_accel"])
        return (1.0 - blend) * velocity_pred + blend * acceleration_pred
    if family == "velocity_accel_grid":
        return (
            last
            + velocity * delta_ms * float(row["velocity_scale"])
            + 0.5 * accel * delta_ms**2 * float(row["accel_scale"])
        )
    raise ValueError(f"unknown family: {family}")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search physics baseline parameters against R-Hit@1cm.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_physics_param_search.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)

    train_samples = read_trajectory_folder(data_dir / "train")
    test_samples = read_trajectory_folder(data_dir / "test")
    targets = read_targets(data_dir / "train_labels.csv")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")

    ids, missing = aligned_ids(train_samples, targets[ID_COLUMN].tolist())
    if missing:
        raise ValueError(f"{len(missing)} train ids are missing trajectory files, examples: {missing[:5]}")

    val_ids = split_ids(ids, args.val_frac, args.seed)
    y_true = targets.set_index(ID_COLUMN).loc[val_ids, COORD_COLUMNS].to_numpy(dtype=float)
    last, velocity, accel = trajectory_arrays(train_samples, val_ids)

    leaderboard = evaluate_grid(last, velocity, accel, y_true)
    top = leaderboard.head(20).copy()
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    best = leaderboard.iloc[0]
    test_ids = sample_submission[ID_COLUMN].tolist()
    test_last, test_velocity, test_accel = trajectory_arrays(test_samples, test_ids)
    pred = make_prediction(test_last, test_velocity, test_accel, best)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "physics_param_search_best.csv"
    submission = sample_submission[[ID_COLUMN]].copy()
    submission[COORD_COLUMNS] = pred
    submission.to_csv(output_path, index=False)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Physics Parameter Search",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- Validation size: `{len(val_ids)}`",
        f"- Best family: `{best['family']}`",
        f"- Best velocity scale: `{best['velocity_scale']:.6f}`",
        f"- Best acceleration scale: `{best['accel_scale']:.6f}`",
        f"- Best acceleration blend: `{best['blend_accel']:.6f}`",
        f"- Best R-Hit@1cm: `{best['r_hit_1cm']:.6f}`",
        f"- Best mean distance: `{best['mean_distance']:.6f}` m",
        f"- Submission: `{output_path}`",
        "",
        "## Top 20",
        "",
        dataframe_to_markdown(top),
        "",
        "## Readout",
        "",
        "- This search optimizes the threshold metric directly, not just average distance.",
        "- If the best velocity scale is below 1.0, the validation set prefers conservative extrapolation to stay inside the 1cm ball more often.",
        "- If acceleration terms do not win, acceleration is likely too noisy at the last two intervals and should be smoothed before reuse.",
    ]
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote submission: {output_path}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()

