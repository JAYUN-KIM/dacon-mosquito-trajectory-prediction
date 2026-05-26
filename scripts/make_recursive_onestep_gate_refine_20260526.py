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
    PUBLIC_BEST_SCORE,
    Candidate,
    apply_candidate,
    build_gain_proba_oof,
    build_oof_recursive,
    fit_gain_proba_test,
    full_recursive_predictions,
    make_folds,
    mult_slug,
)


PUBLIC_FEEDBACK = {
    "recstep_rank1_global_osc89b005late_f100s104u096_w18.csv": "miss",
    "recstep_rank2_global_osc89b005late_f100s100u100_w18.csv": "miss",
    "recstep_rank3_global_osc789b006recent_f104s100u100_w04.csv": "miss",
    "recstep_rank4_gate_osc89b005late_f100s100u100_top080_b40.csv": 0.69180,
    "recstep_rank5_global_osc89b005late_f104s100u100_w04.csv": "miss",
}


@dataclass(frozen=True)
class GateRefineConfig:
    name: str
    mult: tuple[float, float, float]
    frac: float
    weight: float


SPEC_NAME = "os_c89_b005_late"
WINNER_MULT = (1.00, 1.00, 1.00)
SIDE_TILT_MULT = (1.00, 1.04, 0.96)

CONFIGS = [
    GateRefineConfig("top08_b45", WINNER_MULT, 0.080, 0.45),
    GateRefineConfig("top06_b40", WINNER_MULT, 0.060, 0.40),
    GateRefineConfig("top10_b40", WINNER_MULT, 0.100, 0.40),
    GateRefineConfig("top08_b35", WINNER_MULT, 0.080, 0.35),
    GateRefineConfig("top08_b50", WINNER_MULT, 0.080, 0.50),
    GateRefineConfig("top12_b40", WINNER_MULT, 0.120, 0.40),
    GateRefineConfig("tilt_top08_b40", SIDE_TILT_MULT, 0.080, 0.40),
    GateRefineConfig("top05_b45", WINNER_MULT, 0.050, 0.45),
]


def find_spec():
    from run_recursive_onestep_dynamics_20260526 import SPECS  # noqa: PLC0415

    return next(spec for spec in SPECS if spec.name == SPEC_NAME)


def output_name(rank: int, config: GateRefineConfig) -> str:
    return (
        f"recstepgate_refine_rank{rank}_{config.name}_"
        f"{mult_slug(config.mult)}_top{int(round(config.frac * 1000)):03d}_"
        f"b{int(round(config.weight * 100)):02d}.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-guided refine around recursive one-step gate winner.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_recursive_onestep_gate_refine_20260526.md")
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

    champion_path = args.submission_dir / CHAMPION_SUBMISSION
    if not champion_path.exists():
        champion_path = args.submission_dir / FALLBACK_CHAMPION
    champion_test = read_submission_coords(champion_path)
    champion_oof = np.load(args.champion_oof)["champion_oof"]
    champion_hit = distance_summary(champion_oof, y)["r_hit_1cm"]

    spec = find_spec()
    folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED)
    print(f"Building OOF recursive predictions for {SPEC_NAME}", flush=True)
    rec_oof_all = build_oof_recursive(train_coords, spec, folds)

    rec_test_all = full_recursive_predictions(train_coords, test_coords, spec)
    gain_oof_cache = {}
    gain_test_cache = {}

    rows = []
    written = []
    for rank, config in enumerate(CONFIGS, start=1):
        candidate = Candidate("gate", SPEC_NAME, config.mult, config.weight, frac=config.frac)
        rec_oof = rec_oof_all[config.mult]
        pair = (SPEC_NAME, config.mult)
        if pair not in gain_oof_cache:
            print(f"Training OOF gain selector for {SPEC_NAME} {mult_slug(config.mult)}", flush=True)
            gain_oof_cache[pair] = build_gain_proba_oof(train_coords, y, champion_oof, rec_oof, folds)
        blended_oof = apply_candidate(champion_oof, rec_oof, candidate, gain_oof_cache[pair])

        rec_test = rec_test_all[config.mult]
        if pair not in gain_test_cache:
            print(f"Training test gain selector for {SPEC_NAME} {mult_slug(config.mult)}", flush=True)
            gain_test_cache[pair] = fit_gain_proba_test(
                train_coords,
                y,
                champion_oof,
                rec_oof,
                test_coords,
                champion_test,
                rec_test,
            )
        pred = apply_candidate(champion_test, rec_test, candidate, gain_test_cache[pair])
        path = args.submission_dir / output_name(rank, config)
        write_submission(sample_submission, pred, path)
        written.append(path)

        metrics = distance_summary(blended_oof, y)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "config": config.name,
                "mult": mult_slug(config.mult),
                "frac": config.frac,
                "weight": config.weight,
                "oof_hit": metrics["r_hit_1cm"],
                "oof_delta_hit_vs_champion": metrics["r_hit_1cm"] - champion_hit,
                **delta_summary(blended_oof, champion_oof, "oof_vs_champion"),
                **delta_summary(pred, champion_test, "test_vs_champion"),
            }
        )

    df = pd.DataFrame(rows)
    report = [
        "# 2026-05-26 Recursive One-Step Gate Refine",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_before_today: `{PUBLIC_BEST_SCORE:.5f}`",
        f"- public_feedback: `{PUBLIC_FEEDBACK}`",
        f"- successful_axis: `recursive one-step gain gate, {SPEC_NAME}, {mult_slug(WINNER_MULT)}, top080_b40 = 0.69180`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Global recursive blends failed, but the narrow gain-gated candidate improved to 0.69180.",
        "- Keep the same late one-step dynamics model and refine only the public-positive gate region.",
        "- Primary knobs: selected fraction around top 8% and blend strength around 40%.",
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(df),
        "",
        "## Recommended Public Order",
        "",
        "1. `recstepgate_refine_rank1_top08_b45_f100s100u100_top080_b45.csv`",
        "2. `recstepgate_refine_rank2_top06_b40_f100s100u100_top060_b40.csv`",
        "3. `recstepgate_refine_rank3_top10_b40_f100s100u100_top100_b40.csv`",
        "4. `recstepgate_refine_rank4_top08_b35_f100s100u100_top080_b35.csv`",
        "5. `recstepgate_refine_rank5_top08_b50_f100s100u100_top080_b50.csv`",
        "",
        "## Decision Rule",
        "",
        "- If rank1 improves, continue strength search upward around 0.45-0.55.",
        "- If rank2 improves, the selected region should be narrowed below 8%.",
        "- If rank3 improves, the selected region should be widened above 8%.",
        "- If all tie or drop, keep `top080_b40` as the new champion and search a different selector feature set.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
