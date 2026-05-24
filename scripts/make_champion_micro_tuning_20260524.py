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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, stack_samples  # noqa: E402
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
from run_local_target_manifold_projection_20260521 import build_champion_oof  # noqa: E402


CHAMPION = "curvgate_refine_rank2_gatet52a105.csv"
COCHAMP_T54 = "curvgate_rank4_gatet54a105.csv"
TEMPORAL55 = "temporalbc_refine_r1f102s100u100_w55.csv"
CURRENT_SCORE = 0.69120
R_HIT = 0.01


@dataclass(frozen=True)
class MicroConfig:
    name: str
    kind: str
    threshold: float = 0.52
    high_alpha: float = 0.105
    ring_low: float = 0.52
    ring_high: float = 0.54
    ring_alpha: float = 0.0525
    temp: float = 0.018
    center: float = 0.53


CONFIGS = [
    # Threshold interpolation between the two public co-champions t52/t54.
    MicroConfig("gate_t525_a105", "gate", threshold=0.525, high_alpha=0.105),
    MicroConfig("gate_t530_a105", "gate", threshold=0.530, high_alpha=0.105),
    MicroConfig("gate_t535_a105", "gate", threshold=0.535, high_alpha=0.105),
    MicroConfig("gate_t545_a105", "gate", threshold=0.545, high_alpha=0.105),
    MicroConfig("gate_t550_a105", "gate", threshold=0.550, high_alpha=0.105),
    # Keep t52 route but test tiny alpha shrink/expand around the champion alpha.
    MicroConfig("gate_t520_a1025", "gate", threshold=0.520, high_alpha=0.1025),
    MicroConfig("gate_t520_a1075", "gate", threshold=0.520, high_alpha=0.1075),
    MicroConfig("gate_t540_a1025", "gate", threshold=0.540, high_alpha=0.1025),
    MicroConfig("gate_t540_a1075", "gate", threshold=0.540, high_alpha=0.1075),
    # Ring-only tuning: only samples where t52 and t54 disagree are changed.
    MicroConfig("ring52_54_a025", "ring", ring_alpha=0.0250),
    MicroConfig("ring52_54_a040", "ring", ring_alpha=0.0400),
    MicroConfig("ring52_54_a065", "ring", ring_alpha=0.0650),
    MicroConfig("ring52_54_a080", "ring", ring_alpha=0.0800),
    # Smooth confidence around the same region. Riskier, but still anchored near champion.
    MicroConfig("sigmoid_c530_t012", "sigmoid", center=0.530, temp=0.012),
    MicroConfig("sigmoid_c535_t016", "sigmoid", center=0.535, temp=0.016),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def make_prediction(anchor: np.ndarray, correction: np.ndarray, gate_proba: np.ndarray, config: MicroConfig) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.zeros(len(anchor), dtype=np.float64)
    if config.kind == "gate":
        alpha[gate_proba >= config.threshold] = config.high_alpha
    elif config.kind == "ring":
        alpha[gate_proba >= config.ring_high] = config.high_alpha
        ring = (gate_proba >= config.ring_low) & (gate_proba < config.ring_high)
        alpha[ring] = config.ring_alpha
    elif config.kind == "sigmoid":
        factor = 1.0 / (1.0 + np.exp(-(gate_proba - config.center) / config.temp))
        alpha = config.high_alpha * factor
    else:
        raise ValueError(f"unknown config kind: {config.kind}")
    return anchor + alpha[:, None] * correction, alpha


def load_or_build_train_gate(
    cache_path: Path,
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    anchor_oof: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path)
        if len(data["champion_oof"]) == len(coords):
            return data["champion_oof"], data["gate_proba"]
    champion, gate_proba = build_champion_oof(coords, y, features, anchor_oof, correction)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, champion_oof=champion, gate_proba=gate_proba)
    return champion, gate_proba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro tune around the 0.6912 curvature-gate champion.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--champion-cache", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_champion_micro_tuning_20260524.md")
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

    print("Building gate features", flush=True)
    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)

    champion_oof, train_gate_proba = load_or_build_train_gate(
        args.champion_cache,
        train_coords,
        y,
        train_features,
        anchor_oof,
        train_correction,
    )

    fixed_oof = anchor_oof + 0.090 * train_correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    labels = (fixed_dist < anchor_dist).astype(np.int8)
    weights = boundary_weights(anchor_dist, fixed_dist)
    alpha_target = optimal_alpha(anchor_oof, train_correction, y)
    train_gate_features = make_gate_features(train_coords, train_features, anchor_oof, train_correction)

    temporal_test = read_submission_coords(args.submission_dir / TEMPORAL55)
    champion_test = read_submission_coords(args.submission_dir / CHAMPION)
    cochamp_t54 = read_submission_coords(args.submission_dir / COCHAMP_T54)
    test_gate_features = make_gate_features(test_coords, test_features, temporal_test, test_correction)

    print("Training full gate for test micro candidates", flush=True)
    test_gate_proba, _ = fit_full_gate(train_gate_features, labels, alpha_target, weights, test_gate_features)

    baseline_hit = distance_summary(champion_oof, y)["r_hit_1cm"]
    rows = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray, MicroConfig]] = {}
    for config in CONFIGS:
        train_pred, train_alpha = make_prediction(anchor_oof, train_correction, train_gate_proba, config)
        test_pred, test_alpha = make_prediction(temporal_test, test_correction, test_gate_proba, config)
        row = {
            "name": config.name,
            "kind": config.kind,
            "mean_alpha_train": float(np.mean(train_alpha)),
            "route_fraction_train": float(np.mean(train_alpha > 1e-12)),
            "mean_alpha_test": float(np.mean(test_alpha)),
            "route_fraction_test": float(np.mean(test_alpha > 1e-12)),
            **distance_summary(train_pred, y),
            **delta_summary(train_pred, champion_oof, "train_vs_champion"),
            **delta_summary(test_pred, champion_test, "test_vs_champion"),
            **delta_summary(test_pred, cochamp_t54, "test_vs_t54"),
        }
        row["delta_hit_vs_champion"] = row["r_hit_1cm"] - baseline_hit
        # Public prior: exact t52/t54 plateau was best; reward small movement and punish broad sigmoid moves.
        row["selection_score"] = (
            row["delta_hit_vs_champion"]
            - 0.05 * row["test_vs_champion_mean_delta"]
            - (0.0002 if config.kind == "sigmoid" else 0.0)
        )
        rows.append(row)
        predictions[config.name] = (test_pred, test_alpha, config)

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "delta_hit_vs_champion", "test_vs_champion_mean_delta"],
        ascending=[False, False, True],
    )
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        name = str(row["name"])
        test_pred, test_alpha, _ = predictions[name]
        path = args.submission_dir / f"champmicro_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, test_pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "name": name,
                "delta_hit_vs_champion_oof": float(row["delta_hit_vs_champion"]),
                "selection_score": float(row["selection_score"]),
                "test_route_fraction": float(np.mean(test_alpha > 1e-12)),
                "test_mean_alpha": float(np.mean(test_alpha)),
                "test_changed_fraction_vs_champion": float(np.mean(np.linalg.norm(test_pred - champion_test, axis=1) > 1e-12)),
                **delta_summary(test_pred, champion_test, "vs_champion"),
            }
        )

    report = [
        "# 2026-05-24 Champion Micro Tuning",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{CHAMPION} = {CURRENT_SCORE:.5f}`",
        f"- failed_new_axes_today: `regimemiss_rank1 = 0.6906`, `analogknn_rank1 = 0.6886`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Public feedback says new miss-policy/KNN axes are worse, so we return to the robust 0.6912 curvature-gate family.",
        "- The only active search space is a very small neighborhood around the t52/t54 co-champion plateau.",
        "- Candidates either interpolate the t52/t54 disagreement ring or adjust alpha by 0.0025 around the known 0.105 correction.",
        "",
        "## OOF Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- Treat this as conservative exploitation, not a new discovery axis.",
        "- If these stay at or below 0.6912, the next meaningful work should be a better gate calibration objective, not more coordinate perturbation.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
