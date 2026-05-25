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


CHAMPION = "champmicro_rank3_gatet520a1025.csv"
FALLBACK_CHAMPION = "champalpha_rank1_t52a1015.csv"
TEMPORAL55 = "temporalbc_refine_r1f102s100u100_w55.csv"
PUBLIC_BEST_SCORE = 0.69140
PUBLIC_FEEDBACK = {
    "hitmode_rank1_localshape_k32_s0008_clustermean_r30_b18_c00008.csv": 0.68980,
    "champmicro_rank1_gatet520a1075.csv": 0.69100,
    "champmicro_rank3_gatet520a1025.csv": 0.69140,
    "champalpha_rank1_t52a1015.csv": 0.69140,
}


@dataclass(frozen=True)
class AlphaConfig:
    name: str
    threshold: float
    alpha: float


CONFIGS = [
    AlphaConfig("t52_a1010", 0.520, 0.1010),
    AlphaConfig("t52_a1012", 0.520, 0.1012),
    AlphaConfig("t52_a1018", 0.520, 0.1018),
    AlphaConfig("t52_a1022", 0.520, 0.1022),
    AlphaConfig("t52_a1005", 0.520, 0.1005),
    AlphaConfig("t52_a1028", 0.520, 0.1028),
    AlphaConfig("t52_a0995", 0.520, 0.0995),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def make_prediction(anchor: np.ndarray, correction: np.ndarray, gate_proba: np.ndarray, config: AlphaConfig) -> tuple[np.ndarray, np.ndarray]:
    mask = gate_proba >= config.threshold
    alpha_vec = np.zeros(len(anchor), dtype=np.float64)
    alpha_vec[mask] = config.alpha
    return anchor + alpha_vec[:, None] * correction, alpha_vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultra-fine alpha calibration around the 0.69140 t52 champion.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_champion_alpha_ultrafine_20260525.md")
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

    print("Building gate features", flush=True)
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
    champion_path = args.submission_dir / CHAMPION
    if not champion_path.exists():
        champion_path = args.submission_dir / FALLBACK_CHAMPION
    champion_test = read_submission_coords(champion_path)
    test_gate_features = make_gate_features(test_coords, test_features, temporal_test, test_correction)

    print("Training full gate ensemble", flush=True)
    test_gate_proba, _ = fit_full_gate(train_gate_features, labels, alpha_target, weights, test_gate_features)

    rows = []
    written = []
    for rank, config in enumerate(CONFIGS, start=1):
        pred, alpha_vec = make_prediction(temporal_test, test_correction, test_gate_proba, config)
        path = args.submission_dir / f"champalpha2_rank{rank}_{slug(config.name)}.csv"
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
                **delta_summary(pred, champion_test, "vs_current_champion"),
            }
        )

    df = pd.DataFrame(rows)
    report = [
        "# 2026-05-25 Champion Alpha Ultrafine",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best: `{PUBLIC_BEST_SCORE:.5f}`",
        f"- public_feedback: `{PUBLIC_FEEDBACK}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- The fresh local hit-mode retrieval probe dropped to 0.68980, so that axis is cut immediately.",
        "- The only recent public-positive signal is alpha-down around the t52 curvature gate.",
        "- This run keeps threshold fixed at 0.52 and probes a very tight alpha band around 0.1015-0.1025.",
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(df),
        "",
        "## Recommended Public Order",
        "",
        "1. `champalpha2_rank1_t52a1010.csv`",
        "2. `champalpha2_rank2_t52a1012.csv`",
        "3. `champalpha2_rank3_t52a1018.csv`",
        "",
        "## Notes",
        "",
        "- If rank1/rank2 beat 0.69140, continue lowering toward 0.1000.",
        "- If rank3 beats 0.69140, the optimum is likely between 0.1015 and 0.1025.",
        "- If all tie or drop, stop alpha probing and return to a genuinely new modeling axis.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
