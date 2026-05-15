from __future__ import annotations

import argparse
import re
import sys
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
from run_direct_multiplier_selector_20260510 import (  # noqa: E402
    CURRENT_BEST_NAME,
    SOURCE_SUBMISSION,
    aligned_predict_proba,
    build_oof_direct_local,
    candidate_index,
    label_weights,
    make_candidate_predictions,
    make_folds,
    make_selector,
    make_selector_features,
    pick_by_indices,
    recover_test_local_from_source,
)
from run_hit_weighted_local_frame import make_features  # noqa: E402


PUBLIC_BEST_SUBMISSION = "direct_selector_rank2_selectorsoft.csv"
PUBLIC_BEST_SCORE = 0.68440
CURRENT_SUBMISSION = "direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv"
CONF45_SUBMISSION = "direct_selector_rank4_selectorconf045.csv"
N_FOLDS = 5
FOLD_SEED = 20260515
GAIN_SEEDS = [515, 2027, 88010]
THRESHOLDS = [0.42, 0.46, 0.50, 0.54, 0.58]


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


def soft_predict(proba: np.ndarray, candidate_preds: np.ndarray) -> np.ndarray:
    return np.einsum("nc,ncd->nd", proba, candidate_preds)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def binary_gain_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=520,
        learning_rate=0.025,
        num_leaves=47,
        min_child_samples=18,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=0.45,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def proba_diagnostics(proba: np.ndarray, candidate_preds: np.ndarray, current_pred: np.ndarray) -> np.ndarray:
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    top1 = sorted_proba[:, 0:1]
    top2 = sorted_proba[:, 1:2]
    top3 = sorted_proba[:, 2:3]
    entropy = -np.sum(np.clip(proba, 1e-12, 1.0) * np.log(np.clip(proba, 1e-12, 1.0)), axis=1, keepdims=True)
    soft_pred = soft_predict(proba, candidate_preds)
    hard_pred = pick_by_indices(candidate_preds, np.argmax(proba, axis=1))
    delta_soft = np.linalg.norm(soft_pred - current_pred, axis=1, keepdims=True)
    delta_hard = np.linalg.norm(hard_pred - current_pred, axis=1, keepdims=True)
    dispersion = np.sqrt(np.sum(proba[:, :, None] * (candidate_preds - soft_pred[:, None, :]) ** 2, axis=(1, 2)))[:, None]
    return np.hstack([top1, top2, top3, top1 - top2, top2 - top3, entropy, delta_soft, delta_hard, dispersion]).astype(np.float32)


def gain_features(selector_features: np.ndarray, proba: np.ndarray, candidate_preds: np.ndarray, current_pred: np.ndarray) -> np.ndarray:
    out = np.hstack([selector_features, proba_diagnostics(proba, candidate_preds, current_pred)]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def build_oof_selector_proba(
    selector_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    folds = make_folds(len(labels), N_FOLDS, FOLD_SEED)
    oof = np.zeros((len(labels), int(np.max(labels)) + 1), dtype=np.float64)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[val_idx] = False
        model = make_selector(60000 + fold_idx * 19)
        model.fit(selector_features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        fold_proba = aligned_predict_proba(model, selector_features[val_idx])
        if fold_proba.shape[1] > oof.shape[1]:
            expanded = np.zeros((len(labels), fold_proba.shape[1]), dtype=np.float64)
            expanded[:, : oof.shape[1]] = oof
            oof = expanded
        oof[val_idx, : fold_proba.shape[1]] = fold_proba
        print(f"selector OOF fold {fold_idx}/{N_FOLDS}: val={len(val_idx)}", flush=True)
    row_sum = oof.sum(axis=1, keepdims=True)
    return np.divide(oof, row_sum, out=np.full_like(oof, 1.0 / oof.shape[1]), where=row_sum > 1e-12)


def fit_predict_gain(train_x: np.ndarray, labels: np.ndarray, weights: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    probas = []
    for seed in GAIN_SEEDS:
        model = binary_gain_model(seed)
        model.fit(train_x, labels, sample_weight=weights)
        probas.append(model.predict_proba(test_x)[:, 1])
    return np.mean(probas, axis=0)


def route_by_probability(
    soft_anchor: np.ndarray,
    fallback: np.ndarray,
    soft_win_proba: np.ndarray,
    threshold: float,
    mode: str,
) -> np.ndarray:
    low_conf = soft_win_proba < threshold
    pred = soft_anchor.copy()
    if mode == "replace":
        pred[low_conf] = fallback[low_conf]
    elif mode == "blend50":
        pred[low_conf] = 0.5 * soft_anchor[low_conf] + 0.5 * fallback[low_conf]
    elif mode == "blend25":
        pred[low_conf] = 0.75 * soft_anchor[low_conf] + 0.25 * fallback[low_conf]
    else:
        raise ValueError(mode)
    return pred


def evaluate_route_cv(
    current_pred: np.ndarray,
    soft_pred: np.ndarray,
    conf45_pred: np.ndarray,
    y: np.ndarray,
    soft_win_proba: np.ndarray,
) -> pd.DataFrame:
    rows = [
        {"strategy": "current", **distance_summary(current_pred, y)},
        {"strategy": "soft_anchor", **distance_summary(soft_pred, y)},
        {"strategy": "conf45", **distance_summary(conf45_pred, y)},
    ]
    for threshold in THRESHOLDS:
        for fallback_name, fallback in [("current", current_pred), ("conf45", conf45_pred)]:
            for mode in ["blend25", "blend50", "replace"]:
                pred = route_by_probability(soft_pred, fallback, soft_win_proba, threshold, mode)
                rows.append(
                    {
                        "strategy": f"{fallback_name}_{mode}_p{threshold:.2f}",
                        "route_fraction": float(np.mean(soft_win_proba < threshold)),
                        **distance_summary(pred, y),
                    }
                )
    return pd.DataFrame(rows).sort_values(["r_hit_1cm", "mean_distance"], ascending=[False, True])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route-gain model for boundary-aware selector soft fallback.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_route_gain_model_20260515.md")
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
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)

    print("Building OOF direct-step local predictions", flush=True)
    oof_local = build_oof_direct_local(train_coords, y, train_features)
    train_candidate_preds = make_candidate_predictions(train_coords, oof_local)
    distances = np.linalg.norm(train_candidate_preds - y[:, None, :], axis=2)
    selector_labels = np.argmin(distances, axis=1)
    best_dist = np.min(distances, axis=1)
    current_idx = candidate_index(CURRENT_BEST_NAME)
    current_dist = distances[:, current_idx]
    selector_weights = label_weights(best_dist, current_dist)
    selector_features = make_selector_features(train_features, train_coords, oof_local)

    print("Building OOF selector probabilities", flush=True)
    train_proba = build_oof_selector_proba(selector_features, selector_labels, selector_weights)
    current_pred = train_candidate_preds[:, current_idx]
    soft_pred = soft_predict(train_proba, train_candidate_preds)
    hard_pred = pick_by_indices(train_candidate_preds, np.argmax(train_proba, axis=1))
    max_prob = np.max(train_proba, axis=1)
    conf45_pred = current_pred.copy()
    route45 = max_prob >= 0.45
    conf45_pred[route45] = hard_pred[route45]

    soft_dist = np.linalg.norm(soft_pred - y, axis=1)
    current_dist_actual = np.linalg.norm(current_pred - y, axis=1)
    gain_label = (soft_dist <= current_dist_actual).astype(np.int32)
    # Focus the gain model on samples near the hit boundary or where the two routes differ.
    gain_weight = 1.0 + 3.0 * np.exp(-0.5 * ((np.minimum(soft_dist, current_dist_actual) - 0.010) / 0.004) ** 2)
    gain_weight += 2.0 * np.clip(np.abs(soft_dist - current_dist_actual) / 0.004, 0.0, 1.0)
    gain_weight = np.clip(gain_weight, 0.5, 8.0)
    gain_weight = gain_weight / np.mean(gain_weight)
    train_gain_features = gain_features(selector_features, train_proba, train_candidate_preds, current_pred)

    print("OOF route diagnostics", flush=True)
    oof_gain_proba = fit_predict_gain(train_gain_features, gain_label, gain_weight, train_gain_features)
    cv_table = evaluate_route_cv(current_pred, soft_pred, conf45_pred, y, oof_gain_proba)
    print(cv_table.head(12).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Preparing test probabilities", flush=True)
    source_pred = read_submission_coords(args.submission_dir / SOURCE_SUBMISSION)
    test_local = recover_test_local_from_source(test_coords, source_pred)
    test_candidate_preds = make_candidate_predictions(test_coords, test_local)
    test_selector_features = make_selector_features(test_features, test_coords, test_local)
    full_selector = make_selector(88010)
    full_selector.fit(selector_features, selector_labels, sample_weight=selector_weights)
    test_proba = aligned_predict_proba(full_selector, test_selector_features)
    test_current = read_submission_coords(args.submission_dir / CURRENT_SUBMISSION)
    test_soft = read_submission_coords(args.submission_dir / PUBLIC_BEST_SUBMISSION)
    test_conf45 = read_submission_coords(args.submission_dir / CONF45_SUBMISSION)
    test_gain_features = gain_features(test_selector_features, test_proba, test_candidate_preds, test_candidate_preds[:, current_idx])
    test_gain_proba = fit_predict_gain(train_gain_features, gain_label, gain_weight, test_gain_features)

    output_specs = [
        ("current_blend25_p046", test_current, 0.46, "blend25"),
        ("current_blend50_p046", test_current, 0.46, "blend50"),
        ("current_blend25_p050", test_current, 0.50, "blend25"),
        ("conf45_blend25_p046", test_conf45, 0.46, "blend25"),
        ("conf45_blend50_p046", test_conf45, 0.46, "blend50"),
        ("current_replace_p042", test_current, 0.42, "replace"),
    ]
    rows = []
    written = []
    for rank, (name, fallback, threshold, mode) in enumerate(output_specs, start=1):
        pred = route_by_probability(test_soft, fallback, test_gain_proba, threshold, mode)
        path = args.submission_dir / f"routegain_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "strategy": name,
                "fallback": "current" if fallback is test_current else "conf45",
                "threshold": threshold,
                "mode": mode,
                "route_fraction": float(np.mean(test_gain_proba < threshold)),
                "mean_soft_win_proba": float(np.mean(test_gain_proba)),
                **delta_summary(pred, test_soft, "vs_soft_anchor"),
            }
        )

    report = [
        "# 2026-05-15 Route Gain Model",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## OOF Route Diagnostics",
        "",
        dataframe_to_markdown(cv_table),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(rows)),
        "",
        "## Notes",
        "",
        "- The model predicts whether selector_soft is closer than the current multiplier route on OOF data.",
        "- Candidates keep selector_soft as the anchor and only fall back for samples predicted as soft-loss risks.",
        "- This tests the next documented axis: route gain binary model and boundary-aware soft routing.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
