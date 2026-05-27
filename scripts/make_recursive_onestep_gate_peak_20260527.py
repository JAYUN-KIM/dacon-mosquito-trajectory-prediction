from __future__ import annotations

import argparse
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
from run_curvature_gate_20260519 import delta_summary, read_submission_coords, write_submission  # noqa: E402
from run_recursive_onestep_dynamics_20260526 import (  # noqa: E402
    CHAMPION_SUBMISSION,
    FALLBACK_CHAMPION,
    OOF_FOLDS,
    OOF_SEED,
    Candidate,
    apply_candidate,
    build_gain_proba_oof,
    build_oof_recursive,
    fit_gain_proba_test,
    full_recursive_predictions,
    make_folds,
    mult_slug,
)


PUBLIC_WINNER = "recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv"
PUBLIC_WINNER_SCORE = 0.69200
PUBLIC_FEEDBACK = {
    "recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv": 0.69200,
    "recstepgate27_rank2_top090_b400_f100s100u100_top090_b400.csv": 0.69160,
}
SPEC_NAME = "os_c89_b005_late"
WINNER_MULT = (1.00, 1.00, 1.00)


@dataclass(frozen=True)
class PeakConfig:
    name: str
    frac: float
    weight: float


CONFIGS = [
    # Public says top090_b450 > top090_b400, so push strength first.
    PeakConfig("top090_b475", 0.090, 0.475),
    PeakConfig("top090_b500", 0.090, 0.500),
    PeakConfig("top090_b525", 0.090, 0.525),
    # Check whether the winning selected region wants to widen a bit.
    PeakConfig("top095_b450", 0.095, 0.450),
    PeakConfig("top100_b450", 0.100, 0.450),
    PeakConfig("top095_b475", 0.095, 0.475),
    # Guardrails: slightly narrower or slightly weaker around the new winner.
    PeakConfig("top085_b450", 0.085, 0.450),
    PeakConfig("top090_b425", 0.090, 0.425),
    PeakConfig("top100_b475", 0.100, 0.475),
    PeakConfig("top085_b475", 0.085, 0.475),
]


def find_spec():
    from run_recursive_onestep_dynamics_20260526 import SPECS  # noqa: PLC0415

    return next(spec for spec in SPECS if spec.name == SPEC_NAME)


def output_name(rank: int, config: PeakConfig) -> str:
    return (
        f"recstepgate27b_rank{rank}_{config.name}_{mult_slug(WINNER_MULT)}_"
        f"top{int(round(config.frac * 1000)):03d}_b{int(round(config.weight * 1000)):03d}.csv"
    )


def public_distance_penalty(pred: np.ndarray, public_winner: np.ndarray) -> dict[str, float]:
    dist = np.linalg.norm(pred - public_winner, axis=1)
    return {
        "vs_public_winner_mean_delta": float(np.mean(dist)),
        "vs_public_winner_median_delta": float(np.median(dist)),
        "vs_public_winner_p95_delta": float(np.quantile(dist, 0.95)),
        "vs_public_winner_max_delta": float(np.max(dist)),
        "changed_vs_public_winner": int(np.sum(dist > 1e-12)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peak search around 0.6920 recursive one-step gate winner.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_recursive_onestep_gate_peak_20260527.md")
    parser.add_argument("--top-k", type=int, default=8)
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

    base_champion_path = args.submission_dir / CHAMPION_SUBMISSION
    if not base_champion_path.exists():
        base_champion_path = args.submission_dir / FALLBACK_CHAMPION
    base_champion_test = read_submission_coords(base_champion_path)
    public_winner_test = read_submission_coords(args.submission_dir / PUBLIC_WINNER)
    base_champion_oof = np.load(args.champion_oof)["champion_oof"]
    base_champion_hit = distance_summary(base_champion_oof, y)["r_hit_1cm"]

    spec = find_spec()
    folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED)
    print(f"Building OOF recursive predictions for {SPEC_NAME}", flush=True)
    rec_oof = build_oof_recursive(train_coords, spec, folds)[WINNER_MULT]
    print("Building full recursive test prediction", flush=True)
    rec_test = full_recursive_predictions(train_coords, test_coords, spec)[WINNER_MULT]

    print("Training gain selectors", flush=True)
    gain_oof = build_gain_proba_oof(train_coords, y, base_champion_oof, rec_oof, folds)
    gain_test = fit_gain_proba_test(
        train_coords,
        y,
        base_champion_oof,
        rec_oof,
        test_coords,
        base_champion_test,
        rec_test,
    )

    public_proxy_candidate = Candidate("gate", SPEC_NAME, WINNER_MULT, 0.450, frac=0.090)
    public_proxy_oof = apply_candidate(base_champion_oof, rec_oof, public_proxy_candidate, gain_oof)
    public_proxy_hit = distance_summary(public_proxy_oof, y)["r_hit_1cm"]

    rows = []
    pred_cache = {}
    for config in CONFIGS:
        candidate = Candidate("gate", SPEC_NAME, WINNER_MULT, config.weight, frac=config.frac)
        pred_oof = apply_candidate(base_champion_oof, rec_oof, candidate, gain_oof)
        pred_test = apply_candidate(base_champion_test, rec_test, candidate, gain_test)
        metrics = distance_summary(pred_oof, y)
        penalty = public_distance_penalty(pred_test, public_winner_test)
        row = {
            "config": config.name,
            "frac": config.frac,
            "weight": config.weight,
            "oof_hit": metrics["r_hit_1cm"],
            "oof_delta_vs_base_champion": metrics["r_hit_1cm"] - base_champion_hit,
            "oof_delta_vs_public_proxy": metrics["r_hit_1cm"] - public_proxy_hit,
            **delta_summary(pred_oof, public_proxy_oof, "oof_vs_public_proxy"),
            **penalty,
        }
        # Public prior: rank1 showed strength should go up from 0.40 to 0.45.
        # Still penalize candidates that move too far from the known 0.6920 winner.
        row["selection_score"] = (
            0.55 * row["oof_delta_vs_base_champion"]
            + 0.45 * row["oof_delta_vs_public_proxy"]
            - 0.07 * max(0.0, row["vs_public_winner_mean_delta"] - 0.00008)
            - 0.00007 * max(0, row["changed_vs_public_winner"] - 1000) / 100.0
        )
        rows.append(row)
        pred_cache[config.name] = pred_test

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "oof_delta_vs_base_champion", "vs_public_winner_mean_delta"],
        ascending=[False, False, True],
    )
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        config = next(config for config in CONFIGS if config.name == row["config"])
        pred = pred_cache[config.name]
        path = args.submission_dir / output_name(rank, config)
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "config": config.name,
                "frac": config.frac,
                "weight": config.weight,
                "oof_delta_vs_base_champion": float(row["oof_delta_vs_base_champion"]),
                "oof_delta_vs_public_proxy": float(row["oof_delta_vs_public_proxy"]),
                "selection_score": float(row["selection_score"]),
                "changed_vs_public_winner": int(row["changed_vs_public_winner"]),
                "vs_public_winner_mean_delta": float(row["vs_public_winner_mean_delta"]),
                "vs_public_winner_p95_delta": float(row["vs_public_winner_p95_delta"]),
            }
        )

    outputs_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-27 Recursive One-Step Gate Peak",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_winner: `{PUBLIC_WINNER} = {PUBLIC_WINNER_SCORE:.5f}`",
        f"- public_feedback: `{PUBLIC_FEEDBACK}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Public feedback says `top090_b450` improved to 0.6920, while `top090_b400` dropped to 0.6916.",
        "- Therefore the selected fraction around 9% is plausible, and strength should be >= 45%.",
        "- Search 47.5%-52.5% at top 9%, plus 9.5%-10% region checks.",
        "",
        "## Candidate Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(outputs_df),
        "",
        "## Recommended Public Order",
        "",
    ]
    for _, out_row in outputs_df.head(5).iterrows():
        report.append(f"{int(out_row['rank'])}. `{out_row['submission']}`")
    report.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- If rank1 improves, continue strength search above 0.475 at top 9%.",
            "- If rank2 or rank3 wins, the selected fraction should widen toward 9.5%-10%.",
            "- If all drop, keep `top090_b450` as champion and switch to gain selector feature engineering.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
