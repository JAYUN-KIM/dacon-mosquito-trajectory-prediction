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
from run_curvature_gate_20260519 import curvature_correction, read_submission_coords, write_submission  # noqa: E402


CHAMPION = "curvgate_refine_rank2_gatet52a105.csv"
CURRENT_SCORE = 0.69120
SMOOTH_BLEND_WEIGHTS = [0.18, 0.28, 0.40, 0.55]
SNAP_BLEND_WEIGHTS = [0.18, 0.28, 0.40]


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


def weighted_poly_weights(window: int, degree: int, decay: float, pred_step: float = 2.0) -> np.ndarray:
    t = np.arange(-(window - 1), 1, dtype=np.float64)
    powers = np.arange(degree + 1)
    design = t[:, None] ** powers[None, :]
    obs_w = decay ** np.arange(window - 1, -1, -1, dtype=np.float64)
    weighted_design = design * np.sqrt(obs_w)[:, None]
    pred_row = pred_step ** powers
    return pred_row @ np.linalg.pinv(weighted_design) @ np.diag(np.sqrt(obs_w))


def weighted_poly_prediction(coords: np.ndarray, window: int, degree: int, decay: float) -> np.ndarray:
    weights = weighted_poly_weights(window, degree, decay)
    return np.einsum("w,nwc->nc", weights, coords[:, -window:, :])


def smooth_candidates(coords: np.ndarray) -> list[tuple[str, np.ndarray]]:
    configs = [(7, 2, 0.55), (7, 2, 0.70), (9, 2, 0.55), (9, 2, 0.70), (11, 2, 0.55), (11, 3, 0.60)]
    return [
        (f"ewpoly_w{window}_d{degree}_r{int(decay * 100):02d}", weighted_poly_prediction(coords, window, degree, decay))
        for window, degree, decay in configs
    ]


def snap_prediction(coords: np.ndarray, speed_scale: float, accel_scale: float, jerk_scale: float, damp: float) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    v = diffs[:, -1, :]
    a = diffs[:, -1, :] - diffs[:, -2, :]
    prev_a = diffs[:, -2, :] - diffs[:, -3, :]
    jerk = a - prev_a
    # A deliberately aggressive discrete Taylor-style extrapolation.
    future_delta = 2.0 * speed_scale * v + 2.0 * accel_scale * a + jerk_scale * jerk
    return coords[:, -1, :] + damp * future_delta


def snap_candidates(coords: np.ndarray) -> list[tuple[str, np.ndarray]]:
    configs = [
        (0.98, 0.20, -0.35, 0.98),
        (1.00, 0.30, -0.25, 0.98),
        (1.02, 0.35, -0.20, 0.96),
        (1.00, 0.45, 0.15, 0.96),
        (0.96, 0.55, 0.25, 0.94),
        (1.04, 0.15, -0.45, 1.00),
    ]
    return [
        (f"snap_v{int(v * 100):03d}_a{int(a * 100):03d}_j{str(j).replace('-', 'm').replace('.', 'p')}_d{int(d * 100):03d}", snap_prediction(coords, v, a, j, d))
        for v, a, j, d in configs
    ]


def evaluate_blends(
    name_prefix: str,
    train_cands: list[tuple[str, np.ndarray]],
    test_cands: list[tuple[str, np.ndarray]],
    blend_weights: list[float],
    champion_oof: np.ndarray,
    champion_test: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    arrays = {}
    test_lookup = {name: pred for name, pred in test_cands}
    for name, train_pred in train_cands:
        for blend in blend_weights:
            pred_train = (1.0 - blend) * champion_oof + blend * train_pred
            pred_test = (1.0 - blend) * champion_test + blend * test_lookup[name]
            out_name = f"{name_prefix}_{name}_blend{int(round(blend * 100)):02d}"
            arrays[out_name] = pred_test
            rows.append(
                {
                    "name": out_name,
                    "candidate": name,
                    "blend": blend,
                    **distance_summary(pred_train, y),
                    **delta_summary(pred_test, champion_test, "test_vs_champion"),
                }
            )
    df = pd.DataFrame(rows).sort_values(["r_hit_1cm", "mean_distance", "test_vs_champion_mean_delta"], ascending=[False, True, True])
    return df, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast two aggressive new-axis candidates.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_fast_two_new_axes_20260523.md")
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
    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    _, _, train_correction = curvature_correction(train_coords)
    # Use fixed-a105 as a cheap champion proxy to avoid rebuilding the expensive gate model.
    champion_oof_proxy = anchor_oof + 0.105 * train_correction
    champion_test = read_submission_coords(args.submission_dir / CHAMPION)

    smooth_df, smooth_arrays = evaluate_blends(
        "smooth",
        smooth_candidates(train_coords),
        smooth_candidates(test_coords),
        SMOOTH_BLEND_WEIGHTS,
        champion_oof_proxy,
        champion_test,
        y,
    )
    snap_df, snap_arrays = evaluate_blends(
        "snap",
        snap_candidates(train_coords),
        snap_candidates(test_coords),
        SNAP_BLEND_WEIGHTS,
        champion_oof_proxy,
        champion_test,
        y,
    )

    smooth_choice = smooth_df.iloc[0]
    snap_choice = snap_df.iloc[0]
    smooth_name = str(smooth_choice["name"])
    snap_name = str(snap_choice["name"])
    smooth_path = args.submission_dir / f"fastnew_rank1_{slug(smooth_name)}.csv"
    snap_path = args.submission_dir / f"fastnew_rank2_{slug(snap_name)}.csv"
    write_submission(sample_submission, smooth_arrays[smooth_name], smooth_path)
    write_submission(sample_submission, snap_arrays[snap_name], snap_path)

    output_df = pd.DataFrame(
        [
            {"rank": 1, "submission": smooth_path.name, **smooth_choice.to_dict()},
            {"rank": 2, "submission": snap_path.name, **snap_choice.to_dict()},
        ]
    )
    report = [
        "# 2026-05-23 Fast Two New Axes",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- champion: `{CHAMPION} = {CURRENT_SCORE:.5f}`",
        f"- generated_outputs: `{[str(smooth_path), str(snap_path)]}`",
        "",
        "## Idea",
        "",
        "- The temporal curriculum probes scored 0.69060, so this is a fast aggressive pivot.",
        "- Axis 1 uses exponentially weighted polynomial smoothing as a new denoising physics bias.",
        "- Axis 2 uses jerk/snap rebound extrapolation, which is intentionally different from the existing constant-turn correction.",
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(output_df),
        "",
        "## Smooth Axis Leaderboard",
        "",
        dataframe_to_markdown(smooth_df.head(20)),
        "",
        "## Snap Axis Leaderboard",
        "",
        dataframe_to_markdown(snap_df.head(20)),
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {smooth_path}", flush=True)
    print(f"Wrote {snap_path}", flush=True)
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
