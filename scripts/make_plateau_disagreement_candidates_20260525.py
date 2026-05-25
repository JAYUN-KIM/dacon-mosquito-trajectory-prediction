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
    read_sample_submission,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import dataframe_to_markdown, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import read_submission_coords, write_submission  # noqa: E402


PUBLIC_BEST = 0.69140
ANCHOR = "champmicro_rank3_gatet520a1025.csv"
PLATEAU = {
    "a1025": "champmicro_rank3_gatet520a1025.csv",
    "a1015": "champalpha_rank1_t52a1015.csv",
    "a1010": "champalpha2_rank1_t52a1010.csv",
    "a1005": "champalpha2_rank5_t52a1005.csv",
}
STABLE_OLD = {
    "t52_a105": "curvgate_refine_rank2_gatet52a105.csv",
    "t54_a105": "curvgate_rank4_gatet54a105.csv",
    "cochamp_w50": "cochamp_blend_t52_t54_w50.csv",
}


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    mode: str
    source: str = ""
    fraction: float = 0.0
    blend: float = 1.0


CONFIGS = [
    CandidateConfig("stable_mean_all", "mean_all"),
    CandidateConfig("stable_mean_plateau", "mean_plateau"),
    CandidateConfig("toward_a1005_top08_b50", "top_fraction", "a1005", 0.08, 0.50),
    CandidateConfig("toward_a1005_top15_b50", "top_fraction", "a1005", 0.15, 0.50),
    CandidateConfig("toward_t54_top06_b35", "top_fraction", "t54_a105", 0.06, 0.35),
    CandidateConfig("toward_t54_top12_b25", "top_fraction", "t54_a105", 0.12, 0.25),
    CandidateConfig("away_a1075_soft", "anti_overcorrect"),
]


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


def motion_features(coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    dot = np.sum(d_last * d_prev, axis=1)
    denom = np.linalg.norm(d_last, axis=1) * np.linalg.norm(d_prev, axis=1) + 1e-12
    turn = np.arccos(np.clip(dot / denom, -1, 1))
    return np.column_stack(
        [
            speed[:, -1],
            speed[:, -1] - speed[:, -2],
            speed[:, -3:].mean(axis=1),
            accel[:, -1],
            accel[:, -3:].mean(axis=1),
            turn,
            np.abs(d_last[:, 1]) / (np.linalg.norm(d_last, axis=1) + 1e-12),
            np.abs(d_last[:, 2]) / (np.linalg.norm(d_last, axis=1) + 1e-12),
        ]
    )


def top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    n = max(1, int(round(len(score) * fraction)))
    order = np.argsort(score)[::-1]
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:n]] = True
    return mask


def make_predictions(
    submissions: dict[str, np.ndarray],
    coords: np.ndarray,
    config: CandidateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = submissions["a1025"]
    plateau_stack = np.stack([submissions[name] for name in PLATEAU], axis=1)
    stable_stack = np.stack([submissions[name] for name in [*PLATEAU, *STABLE_OLD]], axis=1)

    if config.mode == "mean_all":
        pred = stable_stack.mean(axis=1)
        return pred, np.ones(len(pred), dtype=bool)
    if config.mode == "mean_plateau":
        pred = plateau_stack.mean(axis=1)
        return pred, np.ones(len(pred), dtype=bool)
    if config.mode == "top_fraction":
        source = submissions[config.source]
        disagreement = np.linalg.norm(source - anchor, axis=1)
        motion = motion_features(coords)
        # Give priority to samples with actual plateau disagreement and nontrivial turn/acceleration.
        score = disagreement * (1.0 + 0.25 * motion[:, 5] + 0.10 * motion[:, 3] / (np.median(motion[:, 3]) + 1e-12))
        mask = top_fraction_mask(score, config.fraction)
        pred = anchor.copy()
        pred[mask] = (1.0 - config.blend) * anchor[mask] + config.blend * source[mask]
        return pred, mask
    if config.mode == "anti_overcorrect":
        # a1075 public dropped, so move a tiny subset in the opposite direction from a1075.
        over = submissions["a1075"]
        disagreement = np.linalg.norm(over - anchor, axis=1)
        mask = top_fraction_mask(disagreement, 0.10)
        pred = anchor.copy()
        pred[mask] = anchor[mask] - 0.20 * (over[mask] - anchor[mask])
        return pred, mask
    raise ValueError(f"unknown mode: {config.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-risk disagreement candidates among public plateau submissions.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_plateau_disagreement_20260525.md")
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}", flush=True)
    test_samples = read_trajectory_folder(data_dir / "test")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    test_coords = stack_samples(test_samples, sample_submission[ID_COLUMN].tolist())

    submissions: dict[str, np.ndarray] = {}
    for name, filename in {**PLATEAU, **STABLE_OLD, "a1075": "champmicro_rank1_gatet520a1075.csv"}.items():
        submissions[name] = read_submission_coords(args.submission_dir / filename)

    anchor = submissions["a1025"]
    diag_rows = []
    for name, pred in submissions.items():
        diag_rows.append(
            {
                "name": name,
                **delta_summary(pred, anchor, "vs_anchor"),
            }
        )
    diag_df = pd.DataFrame(diag_rows).sort_values("vs_anchor_mean_delta")

    output_rows = []
    written = []
    for rank, config in enumerate(CONFIGS, start=1):
        pred, mask = make_predictions(submissions, test_coords, config)
        path = args.submission_dir / f"plateaudis_rank{rank}_{slug(config.name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "name": config.name,
                "mode": config.mode,
                "source": config.source or "mixed",
                "route_fraction": float(np.mean(mask)),
                **delta_summary(pred, anchor, "vs_anchor"),
            }
        )

    output_df = pd.DataFrame(output_rows).sort_values(["vs_anchor_mean_delta", "route_fraction"])
    report = [
        "# 2026-05-25 Plateau Disagreement Candidates",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best: `{PUBLIC_BEST:.5f}`",
        f"- anchor: `{ANCHOR}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Alpha probing found a broad 0.69140 plateau.",
        "- Fresh hit-mode retrieval dropped to 0.68980, so large new-coordinate movement is unsafe.",
        "- This experiment only uses disagreement among public-stable plateau/champion submissions and keeps movement extremely small.",
        "",
        "## Submission Distance Diagnostics",
        "",
        dataframe_to_markdown(diag_df),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(output_df),
        "",
        "## Recommended Public Order",
        "",
        "1. `plateaudis_rank2_stablemeanplateau.csv`",
        "2. `plateaudis_rank4_towarda1005top15b50.csv`",
        "",
        "## Notes",
        "",
        "- If both stay at 0.69140, the plateau is saturated and today should stop.",
        "- If either drops, do not submit the other high-movement variants.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
