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
from run_constant_turn_curvature_20260518 import TurnConfig, constant_turn_prediction  # noqa: E402
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
WINNER_NAME = "winner_recgate09_b45"


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    mode: str
    threshold: float
    blend: float
    cap: float
    top_frac: float = 1.0


CONFIGS = [
    SelectorConfig("soft_blend08", "soft_blend", 0.00, 0.08, 0.0040),
    SelectorConfig("soft_blend12", "soft_blend", 0.00, 0.12, 0.0050),
    SelectorConfig("hard_conf22_b35_cap006", "hard_gate", 0.22, 0.35, 0.0060),
    SelectorConfig("hard_conf28_b45_cap007", "hard_gate", 0.28, 0.45, 0.0070),
    SelectorConfig("hard_conf35_b60_cap008", "hard_gate", 0.35, 0.60, 0.0080),
    SelectorConfig("top08_conf18_b40_cap007", "top_gate", 0.18, 0.40, 0.0070, top_frac=0.080),
    SelectorConfig("top05_conf20_b55_cap008", "top_gate", 0.20, 0.55, 0.0080, top_frac=0.050),
    SelectorConfig("top03_conf25_b70_cap010", "top_gate", 0.25, 0.70, 0.0100, top_frac=0.030),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def find_spec():
    from run_recursive_onestep_dynamics_20260526 import SPECS  # noqa: PLC0415

    return next(spec for spec in SPECS if spec.name == SPEC_NAME)


def output_name(rank: int, config: SelectorConfig) -> str:
    return f"wideact28_rank{rank}_{slug(config.name)}.csv"


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
    weights = poly_weights(window, degree)
    return np.einsum("w,nwc->nc", weights, coords[:, -window:, :])


def weighted_diff_prediction(coords: np.ndarray, window: int, decay: float) -> np.ndarray:
    diffs = np.diff(coords[:, -window:, :], axis=1)
    weights = decay ** np.arange(diffs.shape[1] - 1, -1, -1, dtype=float)
    weights = weights / weights.sum()
    step = np.einsum("w,nwc->nc", weights, diffs)
    return coords[:, -1, :] + 2.0 * step


def analytic_candidates(coords: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    names = []
    preds = []

    for velocity_scale in [0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10]:
        for accel_scale in [-0.12, -0.06, 0.00, 0.08, 0.15, 0.22, 0.30, 0.40, 0.52]:
            names.append(f"phys_v{velocity_scale:.2f}_a{accel_scale:.2f}")
            preds.append(physics_prediction(coords, velocity_scale, accel_scale))

    for window, degree in [(3, 1), (4, 1), (5, 1), (7, 1), (9, 1), (11, 1), (3, 2), (5, 2), (7, 2), (9, 2), (11, 2)]:
        names.append(f"poly_w{window}_d{degree}")
        preds.append(poly_prediction(coords, window, degree))

    for window, decay in [(3, 0.50), (3, 0.80), (5, 0.50), (5, 0.75), (7, 0.60), (9, 0.65), (11, 0.70)]:
        names.append(f"wdiff_w{window}_d{decay:.2f}")
        preds.append(weighted_diff_prediction(coords, window, decay))

    turn_configs = [
        TurnConfig("tm025s05d098", 1, -0.25, 0.50, 0.98),
        TurnConfig("tm020s04d098", 1, -0.20, 0.40, 0.98),
        TurnConfig("tp020s04d098", 1, 0.20, 0.40, 0.98),
        TurnConfig("tm030s06d096", 1, -0.30, 0.60, 0.96),
        TurnConfig("tm015s03d100", 1, -0.15, 0.30, 1.00),
    ]
    for cfg in turn_configs:
        names.append(f"turn_{cfg.name}")
        preds.append(constant_turn_prediction(coords, cfg))

    return names, preds


def select_candidate_subset(
    all_names: list[str],
    all_oof: np.ndarray,
    y: np.ndarray,
    keep_top_hit: int = 28,
    keep_oracle: int = 28,
) -> list[int]:
    distances = np.linalg.norm(all_oof - y[:, None, :], axis=2)
    hit_rates = np.mean(distances <= R_HIT, axis=0)
    labels = np.argmin(distances, axis=1)
    counts = np.bincount(labels, minlength=len(all_names))

    selected = {0}
    selected.update(np.argsort(hit_rates)[-keep_top_hit:])
    selected.update(np.argsort(counts)[-keep_oracle:])
    return sorted(selected)


def make_selector_features(coords: np.ndarray, candidate_preds: np.ndarray, winner_index: int) -> np.ndarray:
    base_features, _ = make_features(coords)
    scale = safe_scale(coords)
    winner = candidate_preds[:, winner_index, :]
    deltas = candidate_preds - winner[:, None, :]
    delta_norms = np.linalg.norm(deltas, axis=2)
    spread = np.std(candidate_preds, axis=1)
    last = coords[:, -1, :]
    disp = candidate_preds - last[:, None, :]
    disp_norms = np.linalg.norm(disp, axis=2)
    order = np.argsort(delta_norms, axis=1)
    nearest = np.take_along_axis(delta_norms, order[:, :10], axis=1)
    farthest = np.take_along_axis(delta_norms, order[:, -10:], axis=1)
    handcrafted = np.hstack(
        [
            nearest / scale,
            farthest / scale,
            delta_norms.mean(axis=1, keepdims=True) / scale,
            delta_norms.std(axis=1, keepdims=True) / scale,
            disp_norms.mean(axis=1, keepdims=True) / scale,
            disp_norms.std(axis=1, keepdims=True) / scale,
            spread / scale,
            norm_columns(spread) / scale,
        ]
    )
    handcrafted[~np.isfinite(handcrafted)] = 0.0
    return np.hstack([base_features, handcrafted]).astype(np.float32)


def make_action_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.05,
        reg_lambda=0.45,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def action_weights(best_dist: np.ndarray, winner_dist: np.ndarray) -> np.ndarray:
    boundary = np.exp(-0.5 * ((np.minimum(best_dist, winner_dist) - R_HIT) / 0.0035) ** 2)
    rescue = ((best_dist < R_HIT) & (winner_dist > R_HIT)).astype(np.float64)
    near_gain = np.clip((winner_dist - best_dist) / 0.006, 0.0, 3.0)
    weights = 1.0 + 5.0 * boundary + 2.0 * rescue + near_gain
    return weights / np.mean(weights)


def aligned_proba(model: LGBMClassifier, features: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(features)
    out = np.zeros((len(features), n_classes), dtype=np.float64)
    for col_idx, class_idx in enumerate(model.classes_.astype(int)):
        out[:, class_idx] = raw[:, col_idx]
    return out


def fit_oof_selector(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    folds: list[np.ndarray],
    n_classes: int,
) -> np.ndarray:
    oof = np.zeros((len(features), n_classes), dtype=np.float64)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(features), dtype=bool)
        train_mask[val_idx] = False
        print(f"wide action selector OOF fold {fold_idx}/{len(folds)}", flush=True)
        model = make_action_model(528800 + fold_idx)
        model.fit(features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        oof[val_idx] = aligned_proba(model, features[val_idx], n_classes)
    return oof


def fit_full_selector(
    train_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    test_features: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    probas = []
    for seed in [42, 777, 2026]:
        print(f"full wide action selector seed={seed}", flush=True)
        model = make_action_model(529500 + seed)
        model.fit(train_features, labels, sample_weight=weights)
        probas.append(aligned_proba(model, test_features, n_classes))
    return np.mean(probas, axis=0)


def soft_topn_prediction(proba: np.ndarray, candidates: np.ndarray, topn: int = 5) -> np.ndarray:
    order = np.argsort(proba, axis=1)[:, -topn:]
    rows = np.arange(len(proba))[:, None]
    top_proba = proba[rows, order]
    top_proba = top_proba / (top_proba.sum(axis=1, keepdims=True) + 1e-12)
    top_candidates = candidates[rows, order]
    return np.einsum("nk,nkc->nc", top_proba, top_candidates)


def apply_selector(
    winner_pred: np.ndarray,
    candidate_preds: np.ndarray,
    proba: np.ndarray,
    winner_index: int,
    config: SelectorConfig,
) -> np.ndarray:
    if config.mode == "soft_blend":
        action_pred = soft_topn_prediction(proba, candidate_preds, topn=5)
        movement = cap_vectors(action_pred - winner_pred, config.cap)
        return winner_pred + config.blend * movement

    chosen = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    action_pred = candidate_preds[np.arange(len(candidate_preds)), chosen]
    if config.mode == "hard_gate":
        mask = (chosen != winner_index) & (conf >= config.threshold)
    elif config.mode == "top_gate":
        score = conf * (chosen != winner_index)
        cutoff = np.quantile(score, 1.0 - config.top_frac)
        mask = (score >= cutoff) & (conf >= config.threshold) & (chosen != winner_index)
    else:
        raise ValueError(f"unknown selector mode: {config.mode}")
    movement = config.blend * cap_vectors(action_pred - winner_pred, config.cap)
    pred = winner_pred.copy()
    pred[mask] = pred[mask] + movement[mask]
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wide analytic physics action selector around the 0.692 winner.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_wide_physics_action_selector_20260528.md")
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
    winner_hit = distance_summary(winner_oof, y)["r_hit_1cm"]

    print("Building wide analytic candidate pool", flush=True)
    analytic_names, analytic_oof_list = analytic_candidates(train_coords)
    _, analytic_test_list = analytic_candidates(test_coords)
    all_names = [WINNER_NAME, "base_champion", "recursive_onestep"] + analytic_names
    all_oof = np.stack([winner_oof, base_champion_oof, rec_oof] + analytic_oof_list, axis=1)
    all_test = np.stack([winner_test, base_champion_test, rec_test] + analytic_test_list, axis=1)

    selected = select_candidate_subset(all_names, all_oof, y)
    names = [all_names[idx] for idx in selected]
    candidate_oof = all_oof[:, selected, :]
    candidate_test = all_test[:, selected, :]
    winner_index = names.index(WINNER_NAME)

    distances = np.linalg.norm(candidate_oof - y[:, None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    best_dist = distances[np.arange(len(distances)), labels]
    winner_dist = distances[:, winner_index]
    weights = action_weights(best_dist, winner_dist)
    candidate_hit = pd.DataFrame(
        {
            "candidate": names,
            "oof_hit": np.mean(distances <= R_HIT, axis=0),
            "oracle_count": np.bincount(labels, minlength=len(names)),
        }
    ).sort_values(["oof_hit", "oracle_count"], ascending=[False, False])

    print(f"selected candidate count={len(names)} winner_index={winner_index}", flush=True)
    train_features = make_selector_features(train_coords, candidate_oof, winner_index)
    test_features = make_selector_features(test_coords, candidate_test, winner_index)
    selector_folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED + 41)
    proba_oof = fit_oof_selector(train_features, labels, weights, selector_folds, len(names))
    proba_test = fit_full_selector(train_features, labels, weights, test_features, len(names))

    rows = []
    predictions = {}
    for config in CONFIGS:
        pred_oof = apply_selector(winner_oof, candidate_oof, proba_oof, winner_index, config)
        pred_test = apply_selector(winner_test, candidate_test, proba_test, winner_index, config)
        metrics = distance_summary(pred_oof, y)
        row = {
            "config": config.name,
            "mode": config.mode,
            "threshold": config.threshold,
            "blend": config.blend,
            "cap": config.cap,
            "top_frac": config.top_frac,
            "oof_hit": metrics["r_hit_1cm"],
            "oof_delta_vs_winner": metrics["r_hit_1cm"] - winner_hit,
            **delta_summary(pred_oof, winner_oof, "oof_vs_winner"),
            **delta_summary(pred_test, winner_test, "test_vs_winner"),
        }
        row["selection_score"] = (
            row["oof_delta_vs_winner"]
            - 0.05 * max(0.0, row["test_vs_winner_mean_delta"] - 0.00020)
            - 0.02 * max(0.0, row["test_vs_winner_p95_delta"] - 0.0020)
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
        path = args.submission_dir / output_name(rank, config)
        write_submission(sample_submission, predictions[config.name], path)
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
        "# 2026-05-28 Wide Physics Action Selector",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{PUBLIC_WINNER} = {PUBLIC_WINNER_SCORE:.5f}`",
        f"- oof_winner_hit_proxy: `{winner_hit:.6f}`",
        f"- selected_candidate_count: `{len(names)}`",
        "",
        "## Idea",
        "",
        "- Restart from a broad analytic physics candidate pool.",
        "- Include the current 0.692 winner as just one action among many.",
        "- Train a multiclass action selector to choose the candidate that best optimizes hit@1cm per sample.",
        "- Submit both soft and confidence-gated moves from the winner toward selected actions.",
        "",
        "## Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Candidate Pool",
        "",
        dataframe_to_markdown(candidate_hit.head(30)),
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
            "- If rank1 collapses, wide action selection is overfitting OOF and should be abandoned.",
            "- If rank1 ties or improves, inspect which analytic actions dominate the selected pool and refine that family.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
