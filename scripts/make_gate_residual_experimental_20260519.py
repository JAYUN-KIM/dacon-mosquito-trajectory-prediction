from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


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
    TEMPORAL_BEST_ANCHOR,
    boundary_weights,
    curvature_correction,
    fit_full_gate,
    make_gate_classifier,
    make_gate_features,
    optimal_alpha,
    predict_gate_proba,
    read_submission_coords,
    write_submission,
)
from run_direct_multiplier_selector_20260510 import make_folds  # noqa: E402
from run_hit_weighted_local_frame import make_features  # noqa: E402


BASE_THRESHOLD = 0.52
BASE_ALPHA = 0.105
OOF_FOLDS = 5
OOF_SEED = 20260519
SHRINK_GRID = [0.015, 0.025, 0.040, 0.060, 0.085, 0.110]
CAP_GRID = [0.0006, 0.0009, 0.0012, 0.0016, 0.0022]


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def clip_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / (norm + 1e-12))
    return vectors * scale


def make_residual_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=520,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=32,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.12,
        reg_lambda=1.20,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_residual(
    train_x: np.ndarray,
    train_y: np.ndarray,
    pred_x: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = make_residual_model(seed + axis * 1009)
        model.fit(train_x, train_y[:, axis], sample_weight=weights)
        preds.append(model.predict(pred_x))
    return np.vstack(preds).T.astype(np.float64)


def build_gate_oof_proba(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    proba = np.zeros(len(labels), dtype=np.float64)
    folds = make_folds(len(labels), OOF_FOLDS, OOF_SEED)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[val_idx] = False
        print(f"gate residual OOF gate fold {fold_idx}/{OOF_FOLDS}", flush=True)
        model = make_gate_classifier(951000 + fold_idx * 37)
        model.fit(features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        proba[val_idx] = predict_gate_proba(model, features[val_idx])
    return proba


def build_residual_features(gate_features: np.ndarray, gate_proba: np.ndarray, base_pred: np.ndarray, correction: np.ndarray) -> np.ndarray:
    correction_norm = np.linalg.norm(correction, axis=1, keepdims=True)
    gate_extra = np.hstack(
        [
            gate_proba[:, None],
            (gate_proba >= BASE_THRESHOLD).astype(np.float32)[:, None],
            base_pred,
            np.abs(base_pred),
            correction,
            np.abs(correction),
            correction_norm,
        ]
    ).astype(np.float32)
    out = np.hstack([gate_features, gate_extra]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def residual_weights(base_dist: np.ndarray) -> np.ndarray:
    weights = 1.0 + 10.0 * np.exp(-0.5 * ((base_dist - 0.010) / 0.0026) ** 2)
    weights += 2.0 * (base_dist <= 0.010)
    weights = np.clip(weights, 0.5, 14.0)
    return weights / np.mean(weights)


def evaluate_residual_cv(
    features: np.ndarray,
    residual_target: np.ndarray,
    weights: np.ndarray,
    base_pred: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    folds = make_folds(len(y), OOF_FOLDS, OOF_SEED + 17)
    raw_residual = np.zeros_like(base_pred)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        print(f"residual OOF fold {fold_idx}/{OOF_FOLDS}", flush=True)
        raw_residual[val_idx] = fit_predict_residual(
            features[train_mask],
            residual_target[train_mask],
            features[val_idx],
            weights[train_mask],
            961000 + fold_idx * 53,
        )

    rows = [{"strategy": "base_gate_t52_a105", "shrink": 0.0, "cap": 0.0, **distance_summary(base_pred, y)}]
    for shrink in SHRINK_GRID:
        for cap in CAP_GRID:
            pred = base_pred + shrink * clip_vectors(raw_residual, cap)
            rows.append(
                {
                    "strategy": f"residual_sh{int(shrink * 1000):03d}_cap{int(cap * 10000):04d}",
                    "shrink": shrink,
                    "cap": cap,
                    **distance_summary(pred, y),
                }
            )
    df = pd.DataFrame(rows).sort_values(["r_hit_1cm", "mean_distance"], ascending=[False, True])
    return df, raw_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental residual correction on top of curvature-gate best.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_gate_residual_experimental_20260519.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    if not args.cache_path.exists():
        raise FileNotFoundError(args.cache_path)

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

    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)

    fixed_oof = anchor_oof + 0.090 * train_correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    gate_labels = (fixed_dist < anchor_dist).astype(np.int8)
    gate_weights = boundary_weights(anchor_dist, fixed_dist)
    alpha_target = optimal_alpha(anchor_oof, train_correction, y)

    print("Building gate features", flush=True)
    train_gate_features = make_gate_features(train_coords, train_features, anchor_oof, train_correction)
    gate_oof_proba = build_gate_oof_proba(train_gate_features, gate_labels, gate_weights)
    gate_oof_mask = gate_oof_proba >= BASE_THRESHOLD
    base_train = anchor_oof.copy()
    base_train[gate_oof_mask] = anchor_oof[gate_oof_mask] + BASE_ALPHA * train_correction[gate_oof_mask]
    residual_target = y - base_train
    base_dist = np.linalg.norm(base_train - y, axis=1)
    weights = residual_weights(base_dist)
    residual_features = build_residual_features(train_gate_features, gate_oof_proba, base_train, train_correction)

    print("Evaluating residual-on-gate CV", flush=True)
    leaderboard, raw_residual_oof = evaluate_residual_cv(residual_features, residual_target, weights, base_train, y)
    print(leaderboard.head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    best = leaderboard[leaderboard["strategy"] != "base_gate_t52_a105"].iloc[0]
    shrink = float(best["shrink"])
    cap = float(best["cap"])

    print("Preparing test base and full residual model", flush=True)
    temporal_anchor_test = read_submission_coords(args.submission_dir / TEMPORAL_BEST_ANCHOR)
    test_gate_features = make_gate_features(test_coords, test_features, temporal_anchor_test, test_correction)
    test_gate_proba, _ = fit_full_gate(train_gate_features, gate_labels, alpha_target, gate_weights, test_gate_features)
    test_mask = test_gate_proba >= BASE_THRESHOLD
    base_test = temporal_anchor_test.copy()
    base_test[test_mask] = temporal_anchor_test[test_mask] + BASE_ALPHA * test_correction[test_mask]
    test_residual_features = build_residual_features(test_gate_features, test_gate_proba, base_test, test_correction)

    full_residuals = []
    for seed in [2718, 3141, 5772]:
        print(f"full residual seed={seed}", flush=True)
        full_residuals.append(fit_predict_residual(residual_features, residual_target, test_residual_features, weights, 971000 + seed))
    full_residual = np.mean(full_residuals, axis=0)
    pred = base_test + shrink * clip_vectors(full_residual, cap)

    output_path = args.submission_dir / f"gate_residual_exp_sh{int(shrink * 1000):03d}_cap{int(cap * 10000):04d}.csv"
    write_submission(sample_submission, pred, output_path)

    report = [
        "# 2026-05-19 Gate Residual Experimental",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- public feedback before this: `gate_t50_a105 = 0.6910`, `gate_t52_a105 = 0.6912`, `low025 = 0.6904`",
        "- idea: train a tiny residual-vector correction on top of the curvature-gate base, focused on 1cm-boundary samples.",
        f"- selected_output: `{output_path}`",
        f"- selected_strategy: `{best['strategy']}`",
        "",
        "## CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard.head(40)),
        "",
        "## Output Diagnostics",
        "",
        dataframe_to_markdown(
            pd.DataFrame(
                [
                    {
                        "submission": output_path.name,
                        "shrink": shrink,
                        "cap": cap,
                        "test_route_fraction": float(np.mean(test_mask)),
                        **delta_summary(pred, base_test, "vs_gate_base"),
                        **delta_summary(pred, temporal_anchor_test, "vs_temporal_anchor"),
                    }
                ]
            )
        ),
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    print(f"Wrote submission: {output_path}", flush=True)


if __name__ == "__main__":
    main()
