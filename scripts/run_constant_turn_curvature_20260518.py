from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mosquito_trajectory.data import (  # noqa: E402
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import (  # noqa: E402
    dataframe_to_markdown,
    distance_summary,
    physics_prediction,
    stack_samples,
)


TEMPORAL_BEST_SUBMISSION = "temporalbc_refine_r1f102s100u100_w55.csv"
TEMPORAL_BEST_PUBLIC = 0.68880
SELECTOR_SOFT_ANCHOR = "direct_selector_rank2_selectorsoft.csv"
SELECTOR_SOFT_PUBLIC = 0.68440


@dataclass(frozen=True)
class TurnConfig:
    name: str
    rot_window: int
    turn_scale: float
    speed_scale: float
    disp_scale: float


TURN_CONFIGS = [
    TurnConfig(f"w{w}_t{str(t).replace('-', 'm').replace('.', 'p')}_s{str(s).replace('.', 'p')}_d{str(d).replace('.', 'p')}", w, t, s, d)
    for w in [1, 2, 3]
    for t in [-0.25, 0.25, 0.50, 0.75, 1.00, 1.25]
    for s in [0.00, 0.25, 0.50]
    for d in [0.98, 1.00, 1.02]
]
CORRECTION_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 1.00]
ANCHOR_ALPHAS = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_submission_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def unit_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 1e-12)
    return unit, norm[:, 0]


def rotation_vectors(diffs: np.ndarray) -> np.ndarray:
    prev = diffs[:, :-1, :]
    curr = diffs[:, 1:, :]
    u_prev, _ = unit_vectors(prev.reshape(-1, 3))
    u_curr, _ = unit_vectors(curr.reshape(-1, 3))
    u_prev = u_prev.reshape(prev.shape)
    u_curr = u_curr.reshape(curr.shape)
    cross = np.cross(u_prev, u_curr)
    cross_norm = np.linalg.norm(cross, axis=2, keepdims=True)
    dot = np.sum(u_prev * u_curr, axis=2, keepdims=True)
    angle = np.arctan2(cross_norm, np.clip(dot, -1.0, 1.0))
    axis = np.divide(cross, cross_norm, out=np.zeros_like(cross), where=cross_norm > 1e-12)
    return axis * angle


def rotate(vectors: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = np.divide(rotvec, theta, out=np.zeros_like(rotvec), where=theta > 1e-12)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cross = np.cross(axis, vectors)
    dot = np.sum(axis * vectors, axis=1, keepdims=True)
    rotated = vectors * cos_t + cross * sin_t + axis * dot * (1.0 - cos_t)
    return np.where(theta > 1e-12, rotated, vectors)


def constant_turn_prediction(coords: np.ndarray, cfg: TurnConfig) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    rotvecs = rotation_vectors(diffs)
    recent = rotvecs[:, -cfg.rot_window :, :]
    weights = np.linspace(1.0, 2.0, cfg.rot_window, dtype=np.float64)
    step_rot = np.average(recent, axis=1, weights=weights) * cfg.turn_scale

    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    direction, speed_last = unit_vectors(d_last)
    speed_prev = np.linalg.norm(d_prev, axis=1)
    speed_delta = speed_last - speed_prev

    speed1 = np.maximum(speed_last + cfg.speed_scale * speed_delta, 1e-6)
    speed2 = np.maximum(speed_last + 2.0 * cfg.speed_scale * speed_delta, 1e-6)
    direction1 = rotate(direction, step_rot)
    direction2 = rotate(direction1, step_rot)
    future_delta = cfg.disp_scale * (speed1[:, None] * direction1 + speed2[:, None] * direction2)
    return coords[:, -1, :] + future_delta


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def evaluate_curvature(coords: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = physics_prediction(coords, 1.0, 0.0)
    ca = physics_prediction(coords, 1.0, 0.275)
    raw_rows = [
        {"config": "cv_base", "rot_window": 0, "turn_scale": 0.0, "speed_scale": 0.0, "disp_scale": 1.0, **distance_summary(cv, y)},
        {"config": "ca_base", "rot_window": 0, "turn_scale": 0.0, "speed_scale": 0.0, "disp_scale": 1.0, **distance_summary(ca, y)},
    ]
    correction_rows = []
    for cfg in TURN_CONFIGS:
        turn = constant_turn_prediction(coords, cfg)
        raw_rows.append(
            {
                "config": cfg.name,
                "rot_window": cfg.rot_window,
                "turn_scale": cfg.turn_scale,
                "speed_scale": cfg.speed_scale,
                "disp_scale": cfg.disp_scale,
                **distance_summary(turn, y),
            }
        )
        correction = turn - cv
        for base_name, base in [("cv", cv), ("ca", ca)]:
            for alpha in CORRECTION_ALPHAS:
                pred = base + alpha * correction
                correction_rows.append(
                    {
                        "config": cfg.name,
                        "base": base_name,
                        "alpha": alpha,
                        "rot_window": cfg.rot_window,
                        "turn_scale": cfg.turn_scale,
                        "speed_scale": cfg.speed_scale,
                        "disp_scale": cfg.disp_scale,
                        **distance_summary(pred, y),
                    }
                )
    raw = pd.DataFrame(raw_rows).sort_values(["r_hit_1cm", "mean_distance"], ascending=[False, True])
    corr = pd.DataFrame(correction_rows).sort_values(["r_hit_1cm", "mean_distance"], ascending=[False, True])
    return raw, corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constant-turn curvature correction experiment.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_constant_turn_curvature_20260518.md")
    parser.add_argument("--top-k", type=int, default=6)
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

    print("Evaluating constant-turn physics configs", flush=True)
    raw, corr = evaluate_curvature(train_coords, y)
    print("Raw curvature top:")
    print(raw.head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    print("Correction top:")
    print(corr.head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    temporal_anchor = read_submission_coords(args.submission_dir / TEMPORAL_BEST_SUBMISSION)
    selector_anchor = read_submission_coords(args.submission_dir / SELECTOR_SOFT_ANCHOR)
    test_cv = physics_prediction(test_coords, 1.0, 0.0)
    cfg_by_name = {cfg.name: cfg for cfg in TURN_CONFIGS}

    output_rows = []
    written = []
    used_pairs: set[tuple[str, float, str]] = set()
    for _, row in corr.head(24).iterrows():
        cfg_name = str(row["config"])
        cfg = cfg_by_name[cfg_name]
        turn_test = constant_turn_prediction(test_coords, cfg)
        correction = turn_test - test_cv
        for anchor_name, anchor, public_score in [
            ("temporal_best", temporal_anchor, TEMPORAL_BEST_PUBLIC),
            ("selector_soft", selector_anchor, SELECTOR_SOFT_PUBLIC),
        ]:
            for alpha in ANCHOR_ALPHAS:
                key = (cfg_name, alpha, anchor_name)
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                pred = anchor + alpha * correction
                rank = len(written) + 1
                path = args.submission_dir / f"turncurve_rank{rank}_{slug(anchor_name)}_{slug(cfg_name)}_a{int(alpha * 100):02d}.csv"
                write_submission(sample_submission, pred, path)
                written.append(path)
                output_rows.append(
                    {
                        "rank": rank,
                        "submission": path.name,
                        "anchor": anchor_name,
                        "anchor_public": public_score,
                        "config": cfg_name,
                        "anchor_alpha": alpha,
                        "cv_base": row["base"],
                        "cv_alpha": float(row["alpha"]),
                        "cv_r_hit": float(row["r_hit_1cm"]),
                        "cv_mean_distance": float(row["mean_distance"]),
                        **delta_summary(pred, anchor, "vs_anchor"),
                    }
                )
                if len(written) >= args.top_k:
                    break
            if len(written) >= args.top_k:
                break
        if len(written) >= args.top_k:
            break

    report = [
        "# 2026-05-18 Constant-Turn Curvature Correction",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- temporal_best_anchor: `{TEMPORAL_BEST_SUBMISSION} = {TEMPORAL_BEST_PUBLIC:.5f}`",
        f"- selector_soft_anchor: `{SELECTOR_SOFT_ANCHOR} = {SELECTOR_SOFT_PUBLIC:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- New axis after temporal-backcast saturation: nonlinear physics curvature.",
        "- Estimate recent 3D rotation vectors of the velocity direction and extrapolate two future 40ms steps under a constant-turn assumption.",
        "- Use only the curvature correction `(constant_turn - constant_velocity)` and add a small amount to strong public anchors.",
        "",
        "## Raw Constant-Turn CV",
        "",
        dataframe_to_markdown(raw.head(30)),
        "",
        "## Correction CV",
        "",
        dataframe_to_markdown(corr.head(40)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- This is intentionally different from temporal-backcast pseudo-supervision and candidate routing.",
        "- If public improves, expand with regime-specific curvature alphas and learn a curvature correction gate.",
        "- If public drops, the recent turn estimate is too noisy for direct correction and should be used only as a feature for a future model.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
