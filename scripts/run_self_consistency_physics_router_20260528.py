from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


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
from run_hit_weighted_local_frame import make_features, safe_scale  # noqa: E402
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


PHYS_VELOCITY_SCALES = [0.94, 0.97, 1.00, 1.03, 1.06]
PHYS_ACCEL_SCALES = [-0.08, -0.04, 0.00, 0.08, 0.15, 0.22, 0.30, 0.40]
POLY_CONFIGS = [(3, 1), (4, 1), (5, 1), (7, 1), (9, 1), (3, 2), (5, 2), (7, 2), (9, 2), (11, 2)]
WDIFF_CONFIGS = [(3, 0.55), (5, 0.50), (5, 0.75), (7, 0.60), (9, 0.65), (11, 0.70)]
BACKTEST_CUTOFFS = [6, 7, 8]


@dataclass(frozen=True)
class RouterConfig:
    name: str
    top_frac: float
    blend: float
    cap: float
    temperature: float


CONFIGS = [
    RouterConfig("top03_b018_c0015_t090", 0.030, 0.18, 0.0015, 0.90),
    RouterConfig("top05_b016_c0018_t090", 0.050, 0.16, 0.0018, 0.90),
    RouterConfig("top06_b018_c0020_t100", 0.060, 0.18, 0.0020, 1.00),
    RouterConfig("top08_b014_c0020_t110", 0.080, 0.14, 0.0020, 1.10),
    RouterConfig("top04_b024_c0022_t090", 0.040, 0.24, 0.0022, 0.90),
    RouterConfig("top10_b012_c0018_t120", 0.100, 0.12, 0.0018, 1.20),
    RouterConfig("top025_b030_c0025_t085", 0.025, 0.30, 0.0025, 0.85),
    RouterConfig("top06_b026_c0030_t100", 0.060, 0.26, 0.0030, 1.00),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def find_spec():
    from run_recursive_onestep_dynamics_20260526 import SPECS  # noqa: PLC0415

    return next(spec for spec in SPECS if spec.name == SPEC_NAME)


def output_name(rank: int, config: RouterConfig) -> str:
    return f"selfcons28_rank{rank}_{slug(config.name)}.csv"


def norm_columns(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True)


def cap_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norms = norm_columns(vectors)
    return vectors * np.minimum(1.0, cap / (norms + 1e-12))


def physics_prediction(coords: np.ndarray, velocity_scale: float, accel_scale: float) -> np.ndarray:
    last = coords[:, -1, :]
    d_last = coords[:, -1, :] - coords[:, -2, :]
    d_prev = coords[:, -2, :] - coords[:, -3, :]
    return last + 2.0 * velocity_scale * d_last + 2.0 * accel_scale * (d_last - d_prev)


def poly_weights(window: int, degree: int, pred_step: float = 2.0) -> np.ndarray:
    t = np.arange(-(window - 1), 1, dtype=float)
    powers = np.arange(degree + 1)
    design = t[:, None] ** powers[None, :]
    pred_row = pred_step ** powers
    return pred_row @ np.linalg.pinv(design)


def poly_prediction(coords: np.ndarray, window: int, degree: int) -> np.ndarray:
    use_window = min(window, coords.shape[1])
    if use_window <= degree:
        use_window = degree + 1
    weights = poly_weights(use_window, degree)
    return np.einsum("w,nwc->nc", weights, coords[:, -use_window:, :])


def weighted_diff_prediction(coords: np.ndarray, window: int, decay: float) -> np.ndarray:
    use_window = min(window, coords.shape[1])
    diffs = np.diff(coords[:, -use_window:, :], axis=1)
    weights = decay ** np.arange(diffs.shape[1] - 1, -1, -1, dtype=float)
    weights = weights / weights.sum()
    step = np.einsum("w,nwc->nc", weights, diffs)
    return coords[:, -1, :] + 2.0 * step


def candidate_predictions(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    preds = []
    names = []

    for velocity_scale in PHYS_VELOCITY_SCALES:
        for accel_scale in PHYS_ACCEL_SCALES:
            preds.append(physics_prediction(coords, velocity_scale, accel_scale))
            names.append(f"phys_v{velocity_scale:.2f}_a{accel_scale:.2f}")

    for window, degree in POLY_CONFIGS:
        if coords.shape[1] >= degree + 1:
            preds.append(poly_prediction(coords, window, degree))
            names.append(f"poly_w{window}_d{degree}")

    for window, decay in WDIFF_CONFIGS:
        preds.append(weighted_diff_prediction(coords, window, decay))
        names.append(f"wdiff_w{window}_d{decay:.2f}")

    return np.stack(preds, axis=1).astype(np.float64), names


def internal_backtest_errors(coords: np.ndarray, expected_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    error_blocks = []
    weights = []
    for cutoff in BACKTEST_CUTOFFS:
        prefix = coords[:, : cutoff + 1, :]
        target = coords[:, cutoff + 2, :]
        pred, names = candidate_predictions(prefix)
        if names != expected_names:
            raise ValueError("candidate names changed across backtest prefixes")
        error_blocks.append(np.linalg.norm(pred - target[:, None, :], axis=2))
        weights.append(1.0 + 0.35 * (cutoff - min(BACKTEST_CUTOFFS)))
    stacked = np.stack(error_blocks, axis=2)
    weight_arr = np.asarray(weights, dtype=np.float64)
    weighted_errors = np.average(stacked, axis=2, weights=weight_arr)
    return weighted_errors, stacked


def soft_self_consistency_prediction(
    coords: np.ndarray,
    temperature_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    full_preds, names = candidate_predictions(coords)
    errors, error_cube = internal_backtest_errors(coords, names)
    scale = safe_scale(coords)
    tau = np.maximum(0.25 * scale, 0.0006) * temperature_mult
    logits = -errors / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=1, keepdims=True)
    pred = np.einsum("nk,nkc->nc", weights, full_preds)
    stats = np.hstack(
        [
            errors.min(axis=1, keepdims=True),
            np.median(errors, axis=1, keepdims=True),
            errors.mean(axis=1, keepdims=True),
            errors.std(axis=1, keepdims=True),
            np.sort(errors, axis=1)[:, :5],
            error_cube.mean(axis=2).min(axis=1, keepdims=True),
        ]
    )
    return pred, weights, stats.astype(np.float32), names


def make_router_features(
    coords: np.ndarray,
    winner_pred: np.ndarray,
    self_pred: np.ndarray,
    self_stats: np.ndarray,
    self_weights: np.ndarray,
) -> np.ndarray:
    base_features, _ = make_features(coords)
    scale = safe_scale(coords)
    last = coords[:, -1, :]
    delta = self_pred - winner_pred
    winner_disp = winner_pred - last
    self_disp = self_pred - last
    dot = np.sum(delta * winner_disp, axis=1, keepdims=True)
    cos = dot / (norm_columns(delta) * norm_columns(winner_disp) + 1e-12)
    top_weight_mass = np.sort(self_weights, axis=1)[:, -5:].sum(axis=1, keepdims=True)
    handcrafted = np.hstack(
        [
            self_stats / scale,
            top_weight_mass,
            delta / scale,
            np.abs(delta) / scale,
            norm_columns(delta) / scale,
            winner_disp / scale,
            self_disp / scale,
            norm_columns(winner_disp) / scale,
            norm_columns(self_disp) / scale,
            dot / np.maximum(scale**2, 1e-8),
            cos,
        ]
    )
    handcrafted[~np.isfinite(handcrafted)] = 0.0
    return np.hstack([base_features, handcrafted]).astype(np.float32)


def make_gain_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.04,
        reg_lambda=0.45,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def label_weights(winner_dist: np.ndarray, self_dist: np.ndarray) -> np.ndarray:
    boundary = np.exp(-0.5 * ((np.minimum(winner_dist, self_dist) - R_HIT) / 0.0033) ** 2)
    disagreement = np.clip(np.abs(winner_dist - self_dist) / 0.006, 0.0, 2.5)
    weights = 1.0 + 4.5 * boundary + disagreement
    return weights / np.mean(weights)


def fit_oof_gain(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    folds: list[np.ndarray],
) -> np.ndarray:
    oof = np.zeros(len(features), dtype=np.float64)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(features), dtype=bool)
        train_mask[val_idx] = False
        print(f"self-consistency gain OOF fold {fold_idx}/{len(folds)}", flush=True)
        model = make_gain_model(528900 + fold_idx)
        model.fit(features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        oof[val_idx] = model.predict_proba(features[val_idx])[:, 1]
    return oof


def fit_full_gain(
    train_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    preds = []
    for seed in [42, 777, 2026]:
        print(f"full self-consistency gain seed={seed}", flush=True)
        model = make_gain_model(529700 + seed)
        model.fit(train_features, labels, sample_weight=weights)
        preds.append(model.predict_proba(test_features)[:, 1])
    return np.mean(preds, axis=0)


def apply_router(
    winner_pred: np.ndarray,
    self_pred: np.ndarray,
    gain_proba: np.ndarray,
    config: RouterConfig,
) -> np.ndarray:
    cutoff = np.quantile(gain_proba, 1.0 - config.top_frac)
    mask = gain_proba >= cutoff
    soft = np.clip((gain_proba - cutoff) / (gain_proba.max() - cutoff + 1e-12), 0.0, 1.0)
    movement = cap_vectors(self_pred - winner_pred, config.cap)
    pred = winner_pred.copy()
    pred[mask] = pred[mask] + config.blend * soft[mask, None] * movement[mask]
    return pred


def build_winner_proxy(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    y: np.ndarray,
    base_champion_oof: np.ndarray,
    base_champion_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    return (
        apply_candidate(base_champion_oof, rec_oof, winner_candidate, gain_oof),
        apply_candidate(base_champion_test, rec_test, winner_candidate, gain_test),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-consistency physics router around the 0.692 winner.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_self_consistency_physics_router_20260528.md")
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
    public_winner_test = read_submission_coords(args.submission_dir / PUBLIC_WINNER)
    base_champion_oof = np.load(args.champion_oof)["champion_oof"]

    winner_oof, winner_proxy_test = build_winner_proxy(
        train_coords,
        test_coords,
        y,
        base_champion_oof,
        base_champion_test,
    )
    # Use the exact submitted public winner for output coordinates.
    winner_test = public_winner_test
    winner_hit = distance_summary(winner_oof, y)["r_hit_1cm"]
    winner_dist = np.linalg.norm(winner_oof - y, axis=1)

    rows = []
    predictions = {}
    details = []
    folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED + 17)

    for config in CONFIGS:
        print(f"Self-consistency router config={config.name}", flush=True)
        self_oof, self_weights_oof, self_stats_oof, candidate_names = soft_self_consistency_prediction(
            train_coords,
            config.temperature,
        )
        self_test, self_weights_test, self_stats_test, _ = soft_self_consistency_prediction(
            test_coords,
            config.temperature,
        )
        self_dist = np.linalg.norm(self_oof - y, axis=1)
        gain_labels = (self_dist + 0.00025 < winner_dist).astype(np.int8)
        gain_weights = label_weights(winner_dist, self_dist)
        train_features = make_router_features(train_coords, winner_oof, self_oof, self_stats_oof, self_weights_oof)
        test_features = make_router_features(test_coords, winner_test, self_test, self_stats_test, self_weights_test)
        gain_oof = fit_oof_gain(train_features, gain_labels, gain_weights, folds)
        gain_test = fit_full_gain(train_features, gain_labels, gain_weights, test_features)

        pred_oof = apply_router(winner_oof, self_oof, gain_oof, config)
        pred_test = apply_router(winner_test, self_test, gain_test, config)
        metrics = distance_summary(pred_oof, y)
        row = {
            "config": config.name,
            "top_frac": config.top_frac,
            "blend": config.blend,
            "cap": config.cap,
            "temperature": config.temperature,
            "self_oof_hit": distance_summary(self_oof, y)["r_hit_1cm"],
            "gain_positive_rate": float(gain_labels.mean()),
            "oof_hit": metrics["r_hit_1cm"],
            "oof_delta_vs_winner": metrics["r_hit_1cm"] - winner_hit,
            **delta_summary(pred_oof, winner_oof, "oof_vs_winner"),
            **delta_summary(pred_test, winner_test, "test_vs_winner"),
        }
        row["selection_score"] = (
            row["oof_delta_vs_winner"]
            - 0.08 * max(0.0, row["test_vs_winner_mean_delta"] - 0.00009)
            - 0.03 * max(0.0, row["test_vs_winner_p95_delta"] - 0.0010)
        )
        rows.append(row)
        predictions[config.name] = pred_test
        details.append(
            {
                "config": config.name,
                "candidate_count": len(candidate_names),
                "best_internal_candidate_top1": candidate_names[int(np.argmax(self_weights_test.mean(axis=0)))],
            }
        )

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
                "oof_delta_vs_winner": float(row["oof_delta_vs_winner"]),
                "selection_score": float(row["selection_score"]),
                "test_vs_winner_mean_delta": float(row["test_vs_winner_mean_delta"]),
                "test_vs_winner_p95_delta": float(row["test_vs_winner_p95_delta"]),
                "test_vs_winner_max_delta": float(row["test_vs_winner_max_delta"]),
            }
        )

    outputs_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-28 Self-Consistency Physics Router",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{PUBLIC_WINNER} = {PUBLIC_WINNER_SCORE:.5f}`",
        f"- oof_winner_hit_proxy: `{winner_hit:.6f}`",
        "",
        "## Idea",
        "",
        "- Stop tuning the recursive gate itself.",
        "- Build many physics/poly/smoothed-difference candidates.",
        "- For each trajectory, score candidates by how well they predict recent observed endpoints from earlier prefixes.",
        "- Move the 0.692 winner slightly toward this self-consistent physics estimate only on learned gain-positive samples.",
        "",
        "## Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Candidate Notes",
        "",
        dataframe_to_markdown(pd.DataFrame(details)),
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
            "- If rank1 is below 0.6920, self-consistency physics is not complementary enough.",
            "- If rank1 improves, refine candidate families and internal backtest cutoffs before touching recursive gate strength again.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
