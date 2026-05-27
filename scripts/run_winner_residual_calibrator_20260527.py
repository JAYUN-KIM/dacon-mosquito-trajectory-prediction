from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


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
from run_hit_weighted_local_frame import make_features  # noqa: E402
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
)


PUBLIC_WINNER = "recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv"
PUBLIC_WINNER_SCORE = 0.69200
SPEC_NAME = "os_c89_b005_late"
WINNER_MULT = (1.00, 1.00, 1.00)
WINNER_FRAC = 0.090
WINNER_WEIGHT = 0.450
R_HIT = 0.01


@dataclass(frozen=True)
class CalibConfig:
    name: str
    top_frac: float
    shrink: float
    cap: float


CONFIGS = [
    # Conservative mm-level corrections on the likely-miss tail.
    CalibConfig("top03_s010_c0008", 0.030, 0.10, 0.0008),
    CalibConfig("top05_s010_c0010", 0.050, 0.10, 0.0010),
    CalibConfig("top05_s015_c0010", 0.050, 0.15, 0.0010),
    CalibConfig("top08_s010_c0012", 0.080, 0.10, 0.0012),
    CalibConfig("top08_s015_c0012", 0.080, 0.15, 0.0012),
    CalibConfig("top10_s010_c0015", 0.100, 0.10, 0.0015),
    CalibConfig("top04_s020_c0010", 0.040, 0.20, 0.0010),
    CalibConfig("top06_s020_c0012", 0.060, 0.20, 0.0012),
    CalibConfig("top03_s025_c0015", 0.030, 0.25, 0.0015),
    CalibConfig("top10_s015_c0015", 0.100, 0.15, 0.0015),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def find_spec():
    from run_recursive_onestep_dynamics_20260526 import SPECS  # noqa: PLC0415

    return next(spec for spec in SPECS if spec.name == SPEC_NAME)


def output_name(rank: int, config: CalibConfig) -> str:
    return f"wincal27_rank{rank}_{slug(config.name)}.csv"


def norm_columns(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True)


def make_calibration_features(
    coords: np.ndarray,
    base_pred: np.ndarray,
    winner_pred: np.ndarray,
    recursive_pred: np.ndarray,
    gain_proba: np.ndarray,
) -> np.ndarray:
    base_features, _ = make_features(coords)
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    last = coords[:, -1, :]
    winner_disp = winner_pred - last
    base_disp = base_pred - last
    rec_disp = recursive_pred - last
    win_minus_base = winner_pred - base_pred
    rec_minus_winner = recursive_pred - winner_pred
    dot_win_rec = np.sum(winner_disp * rec_minus_winner, axis=1, keepdims=True)
    cos_win_rec = dot_win_rec / (norm_columns(winner_disp) * norm_columns(rec_minus_winner) + 1e-12)
    handcrafted = np.hstack(
        [
            speed[:, -5:],
            accel[:, -5:],
            speed[:, -1:] - speed[:, -2:-1],
            winner_disp,
            np.abs(winner_disp),
            norm_columns(winner_disp),
            base_disp,
            rec_disp,
            win_minus_base,
            rec_minus_winner,
            np.abs(rec_minus_winner),
            norm_columns(win_minus_base),
            norm_columns(rec_minus_winner),
            gain_proba[:, None],
            dot_win_rec,
            cos_win_rec,
        ]
    ).astype(np.float32)
    return np.hstack([base_features, handcrafted]).astype(np.float32)


def make_residual_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=220,
        learning_rate=0.030,
        num_leaves=31,
        min_child_samples=28,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.04,
        reg_lambda=0.42,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_miss_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.030,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.04,
        reg_lambda=0.42,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_weights(dist: np.ndarray) -> np.ndarray:
    boundary = np.exp(-0.5 * ((dist - R_HIT) / 0.0035) ** 2)
    near_miss = ((dist > R_HIT) & (dist <= 0.020)).astype(np.float64)
    weights = 1.0 + 5.0 * boundary + 2.5 * near_miss
    return weights / np.mean(weights)


def cap_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / (norms + 1e-12))
    return vectors * scale


def apply_calibration(
    winner_pred: np.ndarray,
    residual_pred: np.ndarray,
    miss_proba: np.ndarray,
    config: CalibConfig,
) -> np.ndarray:
    cutoff = np.quantile(miss_proba, 1.0 - config.top_frac)
    mask = miss_proba >= cutoff
    correction = config.shrink * cap_vectors(residual_pred, config.cap)
    pred = winner_pred.copy()
    pred[mask] = pred[mask] + correction[mask]
    return pred


def fit_oof_calibrator(
    features: np.ndarray,
    residual_target: np.ndarray,
    winner_dist: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    residual_oof = np.zeros_like(residual_target)
    miss_proba_oof = np.zeros(len(features), dtype=np.float64)
    miss_labels = (winner_dist > R_HIT).astype(np.int8)
    weights = make_weights(winner_dist)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(features), dtype=bool)
        train_mask[val_idx] = False
        print(f"calibrator OOF fold {fold_idx}/{len(folds)}", flush=True)
        miss_model = make_miss_model(927000 + fold_idx)
        miss_model.fit(features[train_mask], miss_labels[train_mask], sample_weight=weights[train_mask])
        miss_proba_oof[val_idx] = miss_model.predict_proba(features[val_idx])[:, 1]
        for axis in range(3):
            model = make_residual_model(927500 + fold_idx * 11 + axis)
            model.fit(
                features[train_mask],
                residual_target[train_mask, axis],
                sample_weight=weights[train_mask],
            )
            residual_oof[val_idx, axis] = model.predict(features[val_idx])
    return residual_oof, miss_proba_oof


def fit_full_calibrator(
    train_features: np.ndarray,
    residual_target: np.ndarray,
    winner_dist: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    miss_labels = (winner_dist > R_HIT).astype(np.int8)
    weights = make_weights(winner_dist)
    miss_probas = []
    residual_preds = []
    for seed in [42, 777]:
        print(f"full calibrator seed={seed}", flush=True)
        miss_model = make_miss_model(928000 + seed)
        miss_model.fit(train_features, miss_labels, sample_weight=weights)
        miss_probas.append(miss_model.predict_proba(test_features)[:, 1])
        axis_preds = []
        for axis in range(3):
            model = make_residual_model(928500 + seed * 5 + axis)
            model.fit(train_features, residual_target[:, axis], sample_weight=weights)
            axis_preds.append(model.predict(test_features))
        residual_preds.append(np.vstack(axis_preds).T)
    return np.mean(residual_preds, axis=0), np.mean(miss_probas, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual calibrator around the 0.692 recursive gate winner.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_winner_residual_calibrator_20260527.md")
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

    base_champion_path = args.submission_dir / CHAMPION_SUBMISSION
    if not base_champion_path.exists():
        base_champion_path = args.submission_dir / FALLBACK_CHAMPION
    base_champion_test = read_submission_coords(base_champion_path)
    winner_test = read_submission_coords(args.submission_dir / PUBLIC_WINNER)
    base_champion_oof = np.load(args.champion_oof)["champion_oof"]

    spec = find_spec()
    folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED)
    print(f"Building recursive predictions for {SPEC_NAME}", flush=True)
    rec_oof = build_oof_recursive(train_coords, spec, folds)[WINNER_MULT]
    rec_test = full_recursive_predictions(train_coords, test_coords, spec)[WINNER_MULT]

    print("Building gain selectors for winner proxy", flush=True)
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
    winner_candidate = Candidate("gate", SPEC_NAME, WINNER_MULT, WINNER_WEIGHT, frac=WINNER_FRAC)
    winner_oof = apply_candidate(base_champion_oof, rec_oof, winner_candidate, gain_oof)
    winner_dist = np.linalg.norm(winner_oof - y, axis=1)
    winner_hit = distance_summary(winner_oof, y)["r_hit_1cm"]

    print("Building calibration feature matrices", flush=True)
    train_features = make_calibration_features(train_coords, base_champion_oof, winner_oof, rec_oof, gain_oof)
    test_features = make_calibration_features(test_coords, base_champion_test, winner_test, rec_test, gain_test)
    residual_target = y - winner_oof

    residual_oof, miss_proba_oof = fit_oof_calibrator(train_features, residual_target, winner_dist, folds)
    residual_test, miss_proba_test = fit_full_calibrator(train_features, residual_target, winner_dist, test_features)

    rows = []
    predictions = {}
    for config in CONFIGS:
        pred_oof = apply_calibration(winner_oof, residual_oof, miss_proba_oof, config)
        pred_test = apply_calibration(winner_test, residual_test, miss_proba_test, config)
        metrics = distance_summary(pred_oof, y)
        row = {
            "config": config.name,
            "top_frac": config.top_frac,
            "shrink": config.shrink,
            "cap": config.cap,
            "oof_hit": metrics["r_hit_1cm"],
            "oof_delta_vs_winner": metrics["r_hit_1cm"] - winner_hit,
            **delta_summary(pred_oof, winner_oof, "oof_vs_winner"),
            **delta_summary(pred_test, winner_test, "test_vs_winner"),
        }
        row["selection_score"] = (
            row["oof_delta_vs_winner"]
            - 0.12 * max(0.0, row["test_vs_winner_mean_delta"] - 0.00006)
            - 0.0002 * max(0.0, config.shrink - 0.15)
        )
        rows.append(row)
        predictions[config.name] = pred_test

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "oof_delta_vs_winner", "test_vs_winner_mean_delta"],
        ascending=[False, False, True],
    )
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        config = next(config for config in CONFIGS if config.name == row["config"])
        pred = predictions[config.name]
        path = args.submission_dir / output_name(rank, config)
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "config": config.name,
                "top_frac": config.top_frac,
                "shrink": config.shrink,
                "cap": config.cap,
                "oof_delta_vs_winner": float(row["oof_delta_vs_winner"]),
                "selection_score": float(row["selection_score"]),
                "test_vs_winner_mean_delta": float(row["test_vs_winner_mean_delta"]),
                "test_vs_winner_p95_delta": float(row["test_vs_winner_p95_delta"]),
                "test_vs_winner_max_delta": float(row["test_vs_winner_max_delta"]),
            }
        )

    outputs_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-27 Winner Residual Calibrator",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{PUBLIC_WINNER} = {PUBLIC_WINNER_SCORE:.5f}`",
        f"- oof_winner_hit_proxy: `{winner_hit:.6f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Stop fraction/strength tuning around the recursive gate plateau.",
        "- Treat the 0.692 winner as a fixed anchor.",
        "- Train a tiny residual calibrator on OOF winner errors.",
        "- Apply only to predicted miss-risk top samples with sub-2mm capped corrections.",
        "",
        "## Leaderboard",
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
    for _, out_row in outputs_df.head(4).iterrows():
        report.append(f"{int(out_row['rank'])}. `{out_row['submission']}`")
    report.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- If rank1 is below 0.6920, stop residual calibrator immediately.",
            "- If rank1 ties, try rank2 only if it has lower movement or different top fraction.",
            "- If any candidate improves, continue with miss-risk feature engineering rather than recursive gate strength tuning.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
