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
    CURRENT_PUBLIC_BEST,
    TEMPORAL_BEST_ANCHOR,
    boundary_weights,
    curvature_correction,
    delta_summary,
    fit_full_gate,
    make_gate_features,
    optimal_alpha,
    read_submission_coords,
    slug,
    write_submission,
)
from run_hit_weighted_local_frame import make_features  # noqa: E402


REFINE_CONFIGS = [
    {"name": "gate_t48_a105", "kind": "gate", "threshold": 0.48, "high_alpha": 0.105, "low_alpha": 0.000},
    {"name": "gate_t52_a105", "kind": "gate", "threshold": 0.52, "high_alpha": 0.105, "low_alpha": 0.000},
    {"name": "gate_t56_a105", "kind": "gate", "threshold": 0.56, "high_alpha": 0.105, "low_alpha": 0.000},
    {"name": "gate_t50_a100", "kind": "gate", "threshold": 0.50, "high_alpha": 0.100, "low_alpha": 0.000},
    {"name": "gate_t50_a110", "kind": "gate", "threshold": 0.50, "high_alpha": 0.110, "low_alpha": 0.000},
    {"name": "gate_t50_a105_low025", "kind": "gate", "threshold": 0.50, "high_alpha": 0.105, "low_alpha": 0.025},
    {"name": "gate_t50_a105_low050", "kind": "gate", "threshold": 0.50, "high_alpha": 0.105, "low_alpha": 0.050},
    {"name": "gate_t52_a110", "kind": "gate", "threshold": 0.52, "high_alpha": 0.110, "low_alpha": 0.000},
]


def make_prediction(anchor: np.ndarray, correction: np.ndarray, gate_proba: np.ndarray, config: dict[str, float | str]) -> tuple[np.ndarray, float]:
    if config["kind"] != "gate":
        raise ValueError(f"unknown config kind: {config['kind']}")
    threshold = float(config["threshold"])
    high_alpha = float(config["high_alpha"])
    low_alpha = float(config["low_alpha"])
    mask = gate_proba >= threshold
    alpha = np.full(len(anchor), low_alpha, dtype=np.float64)
    alpha[mask] = high_alpha
    return anchor + alpha[:, None] * correction, float(np.mean(mask))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-guided curvature gate refine candidates.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_curvature_gate_refine_20260519.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    if not args.cache_path.exists():
        raise FileNotFoundError(f"OOF cache is missing: {args.cache_path}")

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

    print("Loading OOF cache and building features", flush=True)
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
    temporal_anchor_test = read_submission_coords(args.submission_dir / TEMPORAL_BEST_ANCHOR)
    public_best = read_submission_coords(args.submission_dir / CURRENT_PUBLIC_BEST)
    test_gate_features = make_gate_features(test_coords, test_features, temporal_anchor_test, test_correction)

    print("Training full gate ensemble", flush=True)
    test_gate_proba, _ = fit_full_gate(train_gate_features, labels, alpha_target, weights, test_gate_features)

    rows = []
    written: list[Path] = []
    for rank, config in enumerate(REFINE_CONFIGS, start=1):
        pred, route_fraction = make_prediction(temporal_anchor_test, test_correction, test_gate_proba, config)
        path = args.submission_dir / f"curvgate_refine_rank{rank}_{slug(config['name'])}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "name": config["name"],
                "threshold": config["threshold"],
                "high_alpha": config["high_alpha"],
                "low_alpha": config["low_alpha"],
                "route_fraction": route_fraction,
                **delta_summary(pred, public_best, "vs_public_best"),
                **delta_summary(pred, temporal_anchor_test, "vs_temporal_anchor"),
            }
        )

    df = pd.DataFrame(rows)
    recommended = df[df["name"].isin(["gate_t52_a105", "gate_t50_a110", "gate_t48_a105", "gate_t50_a105_low025"])]

    report = [
        "# 2026-05-19 Curvature Gate Refine",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- public feedback: `gate_t50_a105 = 0.691`, `gate_t38_a105 = 0.690`",
        "- interpretation: broad curvature application is not enough; the useful region is around threshold 0.50.",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Recommended Remaining Public Probes",
        "",
        dataframe_to_markdown(recommended),
        "",
        "## All Refine Candidates",
        "",
        dataframe_to_markdown(df),
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
