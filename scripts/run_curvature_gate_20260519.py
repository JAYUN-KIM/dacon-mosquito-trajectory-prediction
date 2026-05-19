from __future__ import annotations

import argparse
import re
import sys
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
from run_aggressive_experiments import (  # noqa: E402
    dataframe_to_markdown,
    distance_summary,
    physics_prediction,
    split_mask,
    stack_samples,
)
from run_constant_turn_curvature_20260518 import (  # noqa: E402
    TurnConfig,
    constant_turn_prediction,
    rotation_vectors,
    unit_vectors,
)
from run_direct_multiplier_selector_20260510 import (  # noqa: E402
    CURRENT_BEST_NAME,
    aligned_predict_proba,
    build_oof_direct_local,
    candidate_index,
    label_weights,
    make_candidate_predictions,
    make_folds,
    make_selector,
    make_selector_features,
)
from run_hit_probability_router_20260516 import build_oof_selector_proba, soft_predict  # noqa: E402
from run_hit_weighted_local_frame import make_features  # noqa: E402
from run_temporal_backcast_augmentation_20260516 import (  # noqa: E402
    SPECS,
    direct_prediction,
    fit_predict_augmented,
)


SELECTOR_SOFT_ANCHOR = "direct_selector_rank2_selectorsoft.csv"
TEMPORAL_BEST_ANCHOR = "temporalbc_refine_r1f102s100u100_w55.csv"
CURRENT_PUBLIC_BEST = "turncurve_refine_temporalbest_w1tm0p25s0p5d0p98_a09.csv"
CURRENT_PUBLIC_SCORE = 0.69000

TEMPORAL_SPEC_NAME = "tb_c678_w020"
TEMPORAL_MULT = (1.02, 1.00, 1.00)
TEMPORAL_BLEND_WEIGHT = 0.55
BEST_TURN_CFG = TurnConfig("w1_tm0p25_s0p5_d0p98", 1, -0.25, 0.50, 0.98)

R_HIT = 0.01
CV_SEEDS = [42, 777, 2026]
TEMPORAL_OOF_FOLDS = 5
TEMPORAL_OOF_SEED = 20260519
FIXED_ALPHAS = [0.000, 0.060, 0.075, 0.085, 0.090, 0.095, 0.105, 0.120, 0.140]
GATE_THRESHOLDS = [0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62]
GATE_ALPHAS = [0.075, 0.085, 0.090, 0.095, 0.105]
SOFT_RULES = [
    (0.35, 1.30, 0.090),
    (0.40, 1.20, 0.090),
    (0.45, 1.10, 0.090),
    (0.50, 1.00, 0.090),
    (0.35, 1.30, 0.095),
    (0.40, 1.20, 0.095),
]
REG_SHRINKS = [0.35, 0.50, 0.70, 1.00]
REG_CAPS = [0.12, 0.14, 0.16]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_submission_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def build_oof_temporal_rank1(coords: np.ndarray, y: np.ndarray, features: np.ndarray) -> np.ndarray:
    specs = {spec.name: spec for spec in SPECS}
    if TEMPORAL_SPEC_NAME not in specs:
        raise ValueError(f"unknown temporal spec: {TEMPORAL_SPEC_NAME}")

    folds = make_folds(len(coords), TEMPORAL_OOF_FOLDS, TEMPORAL_OOF_SEED)
    oof_local = np.zeros((len(coords), 3), dtype=np.float64)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(coords), dtype=bool)
        train_mask[val_idx] = False
        print(
            f"temporal OOF fold {fold_idx}/{TEMPORAL_OOF_FOLDS}: "
            f"train={int(train_mask.sum())} val={len(val_idx)}",
            flush=True,
        )
        oof_local[val_idx] = fit_predict_augmented(
            coords[train_mask],
            y[train_mask],
            features[train_mask],
            features[val_idx],
            specs[TEMPORAL_SPEC_NAME],
            219000 + fold_idx * 37,
        )
    return direct_prediction(coords, oof_local, TEMPORAL_MULT)


def build_oof_selector_soft(coords: np.ndarray, y: np.ndarray, features: np.ndarray) -> np.ndarray:
    print("building OOF direct local for selector-soft anchor", flush=True)
    oof_local = build_oof_direct_local(coords, y, features)
    candidate_preds = make_candidate_predictions(coords, oof_local)
    distances = np.linalg.norm(candidate_preds - y[:, None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    best_dist = np.min(distances, axis=1)
    current_idx = candidate_index(CURRENT_BEST_NAME)
    current_dist = distances[:, current_idx]
    selector_weights = label_weights(best_dist, current_dist)
    selector_features = make_selector_features(features, coords, oof_local)

    print("building OOF selector probabilities", flush=True)
    selector_proba = build_oof_selector_proba(selector_features, labels, selector_weights)
    return soft_predict(selector_proba, candidate_preds)


def build_or_load_oof_cache(
    cache_path: Path,
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        print(f"loading OOF cache: {cache_path}", flush=True)
        data = np.load(cache_path)
        temporal_oof = data["temporal_oof"]
        selector_oof = data["selector_oof"]
        anchor_oof = data["anchor_oof"]
        if len(anchor_oof) == len(coords):
            return temporal_oof, selector_oof, anchor_oof
        print("cache row count mismatch; rebuilding", flush=True)

    print("building temporal rank1 OOF prediction", flush=True)
    temporal_oof = build_oof_temporal_rank1(coords, y, features)
    print("building selector-soft OOF prediction", flush=True)
    selector_oof = build_oof_selector_soft(coords, y, features)
    anchor_oof = (1.0 - TEMPORAL_BLEND_WEIGHT) * selector_oof + TEMPORAL_BLEND_WEIGHT * temporal_oof
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        temporal_oof=temporal_oof,
        selector_oof=selector_oof,
        anchor_oof=anchor_oof,
    )
    print(f"saved OOF cache: {cache_path}", flush=True)
    return temporal_oof, selector_oof, anchor_oof


def curvature_correction(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv = physics_prediction(coords, 1.0, 0.0)
    turn = constant_turn_prediction(coords, BEST_TURN_CFG)
    return cv, turn, turn - cv


def make_gate_features(coords: np.ndarray, base_features: np.ndarray, anchor_pred: np.ndarray, correction: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    rotvecs = rotation_vectors(diffs)
    rot_norm = np.linalg.norm(rotvecs, axis=2)

    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    d_mean = diffs[:, -3:, :].mean(axis=1)
    _, speed_last = unit_vectors(d_last)
    speed_prev = np.linalg.norm(d_prev, axis=1)
    speed_mean = np.mean(speed[:, -4:], axis=1)
    speed_std = np.std(speed[:, -4:], axis=1)
    speed_delta = speed_last - speed_prev

    anchor_delta = anchor_pred - coords[:, -1, :]
    anchor_norm = np.linalg.norm(anchor_delta, axis=1, keepdims=True)
    correction_norm = np.linalg.norm(correction, axis=1, keepdims=True)
    last_norm = np.linalg.norm(d_last, axis=1, keepdims=True)
    dot_anchor_corr = np.sum(anchor_delta * correction, axis=1, keepdims=True)
    cos_anchor_corr = dot_anchor_corr / (anchor_norm * correction_norm + 1e-12)
    dot_last_corr = np.sum(d_last * correction, axis=1, keepdims=True)
    cos_last_corr = dot_last_corr / (last_norm * correction_norm + 1e-12)
    cross_last_corr = np.linalg.norm(np.cross(d_last, correction), axis=1, keepdims=True)

    weighted_rot_1 = rotvecs[:, -1, :]
    weighted_rot_2 = np.average(rotvecs[:, -2:, :], axis=1, weights=np.array([1.0, 2.0]))
    weighted_rot_3 = np.average(rotvecs[:, -3:, :], axis=1, weights=np.array([1.0, 1.5, 2.0]))

    handcrafted = np.hstack(
        [
            d_last,
            d_prev,
            d_mean,
            np.abs(d_last),
            np.abs(d_prev),
            speed[:, -5:],
            accel[:, -5:],
            rot_norm[:, -5:],
            weighted_rot_1,
            weighted_rot_2,
            weighted_rot_3,
            speed_last[:, None],
            speed_prev[:, None],
            speed_mean[:, None],
            speed_std[:, None],
            speed_delta[:, None],
            np.abs(speed_delta)[:, None],
            anchor_delta,
            np.abs(anchor_delta),
            anchor_norm,
            correction,
            np.abs(correction),
            correction_norm,
            correction_norm / (last_norm + 1e-12),
            dot_anchor_corr,
            cos_anchor_corr,
            dot_last_corr,
            cos_last_corr,
            cross_last_corr,
        ]
    ).astype(np.float32)

    out = np.hstack([base_features, handcrafted]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def boundary_weights(anchor_dist: np.ndarray, fixed_dist: np.ndarray) -> np.ndarray:
    best = np.minimum(anchor_dist, fixed_dist)
    weights = 1.0 + 7.5 * np.exp(-0.5 * ((best - R_HIT) / 0.0032) ** 2)
    weights += 2.0 * ((anchor_dist <= R_HIT) != (fixed_dist <= R_HIT))
    weights += 1.0 * np.clip(np.abs(anchor_dist - fixed_dist) / 0.003, 0.0, 1.0)
    weights = np.clip(weights, 0.5, 12.0)
    return weights / np.mean(weights)


def optimal_alpha(anchor: np.ndarray, correction: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum((y - anchor) * correction, axis=1)
    denom = np.sum(correction * correction, axis=1) + 1e-12
    alpha = numerator / denom
    return np.clip(alpha, 0.0, 0.16)


def make_gate_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.020,
        num_leaves=45,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.08,
        reg_lambda=0.75,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_alpha_regressor(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=640,
        learning_rate=0.018,
        num_leaves=39,
        min_child_samples=26,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.08,
        reg_lambda=0.90,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def predict_gate_proba(model: LGBMClassifier, features: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(features)
    if raw.shape[1] == 1:
        positive_class = int(model.classes_[0]) == 1
        return np.ones(len(features), dtype=np.float64) if positive_class else np.zeros(len(features), dtype=np.float64)
    class_to_col = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    return raw[:, class_to_col.get(1, 0)].astype(np.float64)


def strategy_predictions(
    anchor: np.ndarray,
    correction: np.ndarray,
    gate_proba: np.ndarray,
    reg_alpha: np.ndarray,
) -> dict[str, tuple[np.ndarray, float, float]]:
    strategies: dict[str, tuple[np.ndarray, float, float]] = {
        "temporal_anchor": (anchor, 0.0, 0.0),
    }
    for alpha in FIXED_ALPHAS:
        strategies[f"fixed_a{int(round(alpha * 1000)):03d}"] = (anchor + alpha * correction, alpha, 1.0)

    for alpha in GATE_ALPHAS:
        for threshold in GATE_THRESHOLDS:
            mask = gate_proba >= threshold
            pred = anchor.copy()
            pred[mask] = anchor[mask] + alpha * correction[mask]
            strategies[f"gate_t{int(round(threshold * 100)):02d}_a{int(round(alpha * 1000)):03d}"] = (
                pred,
                alpha,
                float(np.mean(mask)),
            )

    for floor, scale, alpha in SOFT_RULES:
        factor = np.clip(floor + scale * gate_proba, 0.0, 1.45)
        pred = anchor + (alpha * factor)[:, None] * correction
        strategies[
            f"soft_f{int(round(floor * 100)):02d}_s{int(round(scale * 100)):03d}_a{int(round(alpha * 1000)):03d}"
        ] = (pred, float(alpha * np.mean(factor)), float(np.mean(factor)))

    for cap in REG_CAPS:
        clipped = np.clip(reg_alpha, 0.0, cap)
        for shrink in REG_SHRINKS:
            alpha_vec = 0.090 + shrink * (clipped - 0.090)
            alpha_vec = np.clip(alpha_vec, 0.0, cap)
            pred = anchor + alpha_vec[:, None] * correction
            strategies[
                f"reg_cap{int(round(cap * 1000)):03d}_sh{int(round(shrink * 100)):03d}"
            ] = (pred, float(np.mean(alpha_vec)), float(np.mean(alpha_vec > 0.090)))
    return strategies


def evaluate_gate_cv(
    features: np.ndarray,
    labels: np.ndarray,
    alpha_target: np.ndarray,
    weights: np.ndarray,
    anchor: np.ndarray,
    correction: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for seed in CV_SEEDS:
        print(f"gate CV seed={seed}", flush=True)
        val_mask = split_mask(len(y), 0.2, seed)
        train_mask = ~val_mask

        clf = make_gate_classifier(931000 + seed)
        clf.fit(features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        gate_proba = predict_gate_proba(clf, features[val_mask])

        reg = make_alpha_regressor(932000 + seed)
        reg.fit(features[train_mask], alpha_target[train_mask], sample_weight=weights[train_mask])
        reg_alpha = np.clip(reg.predict(features[val_mask]), 0.0, 0.18)

        strategies = strategy_predictions(anchor[val_mask], correction[val_mask], gate_proba, reg_alpha)
        for name, (pred, mean_alpha, route_fraction) in strategies.items():
            rows.append(
                {
                    "seed": seed,
                    "strategy": name,
                    "mean_alpha": mean_alpha,
                    "route_fraction": route_fraction,
                    **distance_summary(pred, y[val_mask]),
                }
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("strategy", as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
            mean_alpha=("mean_alpha", "mean"),
            route_fraction=("route_fraction", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    fixed_a090_hit = float(summary.loc[summary["strategy"] == "fixed_a090", "mean_r_hit"].iloc[0])
    summary["delta_vs_fixed_a090"] = summary["mean_r_hit"] - fixed_a090_hit
    return summary


def fit_full_gate(
    features: np.ndarray,
    labels: np.ndarray,
    alpha_target: np.ndarray,
    weights: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probas = []
    alphas = []
    for seed in [1301, 2718, 3141]:
        print(f"full gate seed={seed}", flush=True)
        clf = make_gate_classifier(941000 + seed)
        clf.fit(features, labels, sample_weight=weights)
        probas.append(predict_gate_proba(clf, test_features))

        reg = make_alpha_regressor(942000 + seed)
        reg.fit(features, alpha_target, sample_weight=weights)
        alphas.append(np.clip(reg.predict(test_features), 0.0, 0.18))
    return np.mean(probas, axis=0), np.mean(alphas, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample-wise curvature correction gate.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_curvature_gate_20260519.md")
    parser.add_argument("--top-k", type=int, default=5)
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

    print("Building base feature matrices", flush=True)
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)

    temporal_oof, selector_oof, anchor_oof = build_or_load_oof_cache(args.cache_path, train_coords, y, train_features)
    print("Building curvature corrections", flush=True)
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)

    fixed_oof = anchor_oof + 0.090 * train_correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    labels = (fixed_dist < anchor_dist).astype(np.int8)
    weights = boundary_weights(anchor_dist, fixed_dist)
    alpha_target = optimal_alpha(anchor_oof, train_correction, y)

    print(
        "OOF anchors: "
        f"selector={distance_summary(selector_oof, y)['r_hit_1cm']:.6f} "
        f"temporal={distance_summary(temporal_oof, y)['r_hit_1cm']:.6f} "
        f"blend={distance_summary(anchor_oof, y)['r_hit_1cm']:.6f} "
        f"fixed_a090={distance_summary(fixed_oof, y)['r_hit_1cm']:.6f}",
        flush=True,
    )
    print(
        f"gate labels positive={labels.mean():.4f} "
        f"alpha_target_mean={alpha_target.mean():.5f} alpha_target_p90={np.quantile(alpha_target, 0.90):.5f}",
        flush=True,
    )

    print("Building gate features", flush=True)
    train_gate_features = make_gate_features(train_coords, train_features, anchor_oof, train_correction)
    print("Evaluating curvature gate CV", flush=True)
    leaderboard = evaluate_gate_cv(train_gate_features, labels, alpha_target, weights, anchor_oof, train_correction, y)
    print(leaderboard.head(60).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Preparing test gate features", flush=True)
    temporal_anchor_test = read_submission_coords(args.submission_dir / TEMPORAL_BEST_ANCHOR)
    current_public_best = read_submission_coords(args.submission_dir / CURRENT_PUBLIC_BEST)
    test_gate_features = make_gate_features(test_coords, test_features, temporal_anchor_test, test_correction)

    print("Training full gate models", flush=True)
    test_gate_proba, test_reg_alpha = fit_full_gate(train_gate_features, labels, alpha_target, weights, test_gate_features)
    test_strategies = strategy_predictions(temporal_anchor_test, test_correction, test_gate_proba, test_reg_alpha)

    output_rows = []
    written = []
    skip_names = {"temporal_anchor", "fixed_a090"}
    for _, row in leaderboard.iterrows():
        name = str(row["strategy"])
        if name in skip_names:
            continue
        if name not in test_strategies:
            continue
        pred, mean_alpha, route_fraction = test_strategies[name]
        rank = len(written) + 1
        path = args.submission_dir / f"curvgate_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "strategy": name,
                "cv_mean_r_hit": float(row["mean_r_hit"]),
                "cv_delta_vs_fixed_a090": float(row["delta_vs_fixed_a090"]),
                "test_mean_alpha": mean_alpha,
                "test_route_fraction": route_fraction,
                **delta_summary(pred, current_public_best, "vs_public_best"),
                **delta_summary(pred, temporal_anchor_test, "vs_temporal_anchor"),
            }
        )
        if len(written) >= args.top_k:
            break

    diagnostics = pd.DataFrame(
        [
            {"name": "selector_oof", **distance_summary(selector_oof, y)},
            {"name": "temporal_rank1_oof", **distance_summary(temporal_oof, y)},
            {"name": "temporal55_oof_anchor", **distance_summary(anchor_oof, y)},
            {"name": "temporal55_fixed_a090", **distance_summary(fixed_oof, y)},
        ]
    )

    report = [
        "# 2026-05-19 Curvature Gate",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- current_public_best: `{CURRENT_PUBLIC_BEST} = {CURRENT_PUBLIC_SCORE:.5f}`",
        f"- temporal_anchor: `{TEMPORAL_BEST_ANCHOR}`",
        f"- turn_config: `{BEST_TURN_CFG}`",
        f"- cache_path: `{args.cache_path}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Previous best applies the same constant-turn curvature correction to every sample.",
        "- This experiment trains a sample-wise gate from train OOF outcomes: when does alpha=0.09 improve over the temporal anchor?",
        "- The gate is optimized around the 1cm boundary, then used to build threshold, soft-alpha, and regressed-alpha submissions.",
        "",
        "## OOF Anchor Diagnostics",
        "",
        dataframe_to_markdown(diagnostics),
        "",
        "## Gate CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- If fixed alpha candidates still dominate, the public signal is probably global-alpha sensitive rather than gate-sensitive.",
        "- If gated candidates improve public, expand with multiple curvature configs and a multi-class alpha policy.",
        "- No test labels or external data are used; test submissions only apply models trained on train OOF labels.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
