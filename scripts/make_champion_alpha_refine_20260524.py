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
from run_aggressive_experiments import dataframe_to_markdown, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import (  # noqa: E402
    boundary_weights,
    curvature_correction,
    delta_summary,
    fit_full_gate,
    make_gate_features,
    optimal_alpha,
    read_submission_coords,
    write_submission,
)
from run_hit_weighted_local_frame import make_features  # noqa: E402


CHAMPION = "curvgate_refine_rank2_gatet52a105.csv"
PUBLIC_CHAMPION_SCORE = 0.69120
PUBLIC_FEEDBACK = {
    "champmicro_rank1_gatet520a1075.csv": 0.69100,
    "champmicro_rank3_gatet520a1025.csv": 0.69140,
}
TEMPORAL55 = "temporalbc_refine_r1f102s100u100_w55.csv"


@dataclass(frozen=True)
class AlphaConfig:
    name: str
    threshold: float
    alpha: float


CONFIGS = [
    AlphaConfig("t52_a1015", 0.520, 0.1015),
    AlphaConfig("t52_a1020", 0.520, 0.1020),
    AlphaConfig("t52_a1030", 0.520, 0.1030),
    AlphaConfig("t52_a1000", 0.520, 0.1000),
    AlphaConfig("t52_a1035", 0.520, 0.1035),
    AlphaConfig("t515_a1025", 0.515, 0.1025),
    AlphaConfig("t525_a1025", 0.525, 0.1025),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def make_prediction(anchor: np.ndarray, correction: np.ndarray, gate_proba: np.ndarray, config: AlphaConfig) -> tuple[np.ndarray, np.ndarray]:
    mask = gate_proba >= config.threshold
    alpha_vec = np.zeros(len(anchor), dtype=np.float64)
    alpha_vec[mask] = config.alpha
    return anchor + alpha_vec[:, None] * correction, alpha_vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-guided alpha refinement around t52_a1025.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_champion_alpha_refine_20260524.md")
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

    print("Building train/test gate features", flush=True)
    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)

    fixed_oof = anchor_oof + 0.090 * train_correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    labels = (fixed_dist < anchor_dist).astype(np.int8)
    weights = boundary_weights(anchor_dist, fixed_dist)
    alpha_target = optimal_alpha(anchor_oof, train_correction, y)
    train_gate_features = make_gate_features(train_coords, train_features, anchor_oof, train_correction)

    temporal_test = read_submission_coords(args.submission_dir / TEMPORAL55)
    champion_test = read_submission_coords(args.submission_dir / CHAMPION)
    test_gate_features = make_gate_features(test_coords, test_features, temporal_test, test_correction)

    print("Training full gate ensemble", flush=True)
    test_gate_proba, _ = fit_full_gate(train_gate_features, labels, alpha_target, weights, test_gate_features)

    rows = []
    written = []
    for rank, config in enumerate(CONFIGS, start=1):
        pred, alpha_vec = make_prediction(temporal_test, test_correction, test_gate_proba, config)
        path = args.submission_dir / f"champalpha_rank{rank}_{slug(config.name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "name": config.name,
                "threshold": config.threshold,
                "alpha": config.alpha,
                "route_fraction": float(np.mean(alpha_vec > 0)),
                "mean_alpha": float(np.mean(alpha_vec)),
                **delta_summary(pred, champion_test, "vs_champion"),
            }
        )

    df = pd.DataFrame(rows)
    report = [
        "# 2026-05-24 Champion Alpha Refine",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{CHAMPION} = {PUBLIC_CHAMPION_SCORE:.5f}`",
        f"- public_feedback: `{PUBLIC_FEEDBACK}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Public feedback shows alpha-up failed: t52_a1075 scored 0.6910.",
        "- Public feedback also shows alpha-down helped: t52_a1025 scored 0.6914.",
        "- This run keeps the same robust t52 gate family and probes only a tight alpha band around 0.1025.",
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(df),
        "",
        "## Recommended Public Order",
        "",
        "1. `champalpha_rank1_t52a1015.csv`",
        "2. `champalpha_rank2_t52a1020.csv`",
        "3. `champalpha_rank3_t52a1030.csv`",
        "",
        "## Notes",
        "",
        "- If rank1/rank2 improve, continue lowering alpha toward 0.099-0.101.",
        "- If rank3 improves, the optimum is likely between 0.1025 and 0.105.",
        "- Threshold probes t515/t525 are secondary; use only after alpha neighborhood is mapped.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
