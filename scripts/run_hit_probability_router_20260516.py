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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, split_mask, stack_samples  # noqa: E402
from run_direct_multiplier_selector_20260510 import (  # noqa: E402
    CANDIDATES,
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
R_HIT = 0.01
N_SELECTOR_FOLDS = 5
SELECTOR_FOLD_SEED = 20260516
CV_SEEDS = [42, 777, 2026]
FULL_HIT_SEEDS = [1618, 2718, 31415]
SOFT_POWERS = [1.0, 2.0, 4.0]
BLEND_WEIGHTS = [0.15, 0.30, 0.45]
ROUTE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]


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


def proba_diagnostics(proba: np.ndarray, candidate_preds: np.ndarray) -> np.ndarray:
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    top1 = sorted_proba[:, 0:1]
    top2 = sorted_proba[:, 1:2]
    top3 = sorted_proba[:, 2:3]
    entropy = -np.sum(np.clip(proba, 1e-12, 1.0) * np.log(np.clip(proba, 1e-12, 1.0)), axis=1, keepdims=True)
    soft_pred = soft_predict(proba, candidate_preds)
    dispersion = np.sqrt(np.sum(proba[:, :, None] * (candidate_preds - soft_pred[:, None, :]) ** 2, axis=(1, 2)))[:, None]
    return np.hstack([top1, top2, top3, top1 - top2, top2 - top3, entropy, dispersion]).astype(np.float32)


def build_oof_selector_proba(selector_features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    folds = make_folds(len(labels), N_SELECTOR_FOLDS, SELECTOR_FOLD_SEED)
    probas: np.ndarray | None = None
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[val_idx] = False
        model = make_selector(73000 + fold_idx * 29)
        model.fit(selector_features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        fold_proba = aligned_predict_proba(model, selector_features[val_idx])
        if probas is None:
            probas = np.zeros((len(labels), fold_proba.shape[1]), dtype=np.float64)
        if fold_proba.shape[1] > probas.shape[1]:
            expanded = np.zeros((len(labels), fold_proba.shape[1]), dtype=np.float64)
            expanded[:, : probas.shape[1]] = probas
            probas = expanded
        probas[val_idx, : fold_proba.shape[1]] = fold_proba
        print(f"selector OOF fold {fold_idx}/{N_SELECTOR_FOLDS}: val={len(val_idx)}", flush=True)
    if probas is None:
        raise RuntimeError("failed to build selector OOF probabilities")
    row_sum = probas.sum(axis=1, keepdims=True)
    return np.divide(probas, row_sum, out=np.full_like(probas, 1.0 / probas.shape[1]), where=row_sum > 1e-12)


def candidate_metadata() -> np.ndarray:
    rows = []
    denom = max(len(CANDIDATES) - 1, 1)
    for idx, candidate in enumerate(CANDIDATES):
        mult = np.asarray(candidate.mult, dtype=np.float32)
        rows.append(np.r_[idx / denom, mult, np.abs(mult - 1.0), mult[1] - mult[2], mult[0] - 1.02])
    return np.vstack(rows).astype(np.float32)


def build_hit_rows(
    selector_features: np.ndarray,
    selector_proba: np.ndarray,
    candidate_preds: np.ndarray,
    pred_local_scaled: np.ndarray,
) -> np.ndarray:
    n_rows, n_candidates, _ = candidate_preds.shape
    sample_context = np.hstack([selector_features, selector_proba, proba_diagnostics(selector_proba, candidate_preds)]).astype(np.float32)
    sample_context[~np.isfinite(sample_context)] = 0.0
    repeated_context = np.repeat(sample_context, n_candidates, axis=0)

    mults = np.asarray([candidate.mult for candidate in CANDIDATES], dtype=np.float32)
    candidate_local = pred_local_scaled[:, None, :] * mults[None, :, :]
    current_local = candidate_local[:, candidate_index(CURRENT_BEST_NAME), :]
    local_delta_current = candidate_local - current_local[:, None, :]
    local_abs = np.abs(candidate_local)
    local_norm = np.linalg.norm(candidate_local, axis=2, keepdims=True)
    current_delta_norm = np.linalg.norm(local_delta_current, axis=2, keepdims=True)
    meta = np.broadcast_to(candidate_metadata()[None, :, :], (n_rows, n_candidates, candidate_metadata().shape[1]))

    candidate_context = np.concatenate(
        [
            candidate_local,
            local_abs,
            local_norm,
            local_delta_current,
            current_delta_norm,
            meta,
        ],
        axis=2,
    ).reshape(n_rows * n_candidates, -1)
    out = np.hstack([repeated_context, candidate_context]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def hit_sample_weights(distances: np.ndarray) -> np.ndarray:
    weights = 1.0 + 6.0 * np.exp(-0.5 * ((distances - R_HIT) / 0.0035) ** 2)
    weights += 1.0 * (distances <= R_HIT)
    weights = np.clip(weights, 0.5, 10.0)
    return (weights / np.mean(weights)).reshape(-1)


def make_hit_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=620,
        learning_rate=0.023,
        num_leaves=63,
        min_child_samples=28,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.06,
        reg_lambda=0.55,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_hit(train_x: np.ndarray, labels: np.ndarray, weights: np.ndarray, pred_x: np.ndarray, seed: int) -> np.ndarray:
    model = make_hit_model(seed)
    model.fit(train_x, labels, sample_weight=weights)
    return model.predict_proba(pred_x)[:, 1]


def fit_predict_hit_ensemble(train_x: np.ndarray, labels: np.ndarray, weights: np.ndarray, pred_x: np.ndarray) -> np.ndarray:
    preds = []
    for seed in FULL_HIT_SEEDS:
        print(f"  full hit-router seed={seed}", flush=True)
        preds.append(fit_predict_hit(train_x, labels, weights, pred_x, seed))
    return np.mean(preds, axis=0)


def weighted_candidate_average(proba: np.ndarray, candidate_preds: np.ndarray, power: float, top_k: int | None = None) -> np.ndarray:
    weights = np.clip(proba, 1e-8, 1.0) ** power
    if top_k is not None and top_k < weights.shape[1]:
        keep_idx = np.argpartition(weights, -top_k, axis=1)[:, -top_k:]
        mask = np.zeros_like(weights, dtype=bool)
        rows = np.arange(len(weights))[:, None]
        mask[rows, keep_idx] = True
        weights = np.where(mask, weights, 0.0)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return np.einsum("nc,ncd->nd", weights, candidate_preds)


def strategy_predictions(
    hit_proba: np.ndarray,
    candidate_preds: np.ndarray,
    anchor_pred: np.ndarray,
    current_pred: np.ndarray,
) -> dict[str, tuple[np.ndarray, float | None]]:
    hard_idx = np.argmax(hit_proba, axis=1)
    hard = pick_by_indices(candidate_preds, hard_idx)
    max_proba = np.max(hit_proba, axis=1)

    strategies: dict[str, tuple[np.ndarray, float | None]] = {
        "current_candidate": (current_pred, None),
        "hitprob_hard": (hard, None),
    }
    for power in SOFT_POWERS:
        soft = weighted_candidate_average(hit_proba, candidate_preds, power, top_k=None)
        top3 = weighted_candidate_average(hit_proba, candidate_preds, power, top_k=3)
        strategies[f"hitprob_soft_p{power:g}"] = (soft, None)
        strategies[f"hitprob_top3_p{power:g}"] = (top3, None)
        for blend in BLEND_WEIGHTS:
            strategies[f"anchor_blend_softp{power:g}_w{blend:.2f}"] = ((1.0 - blend) * anchor_pred + blend * soft, None)
            strategies[f"anchor_blend_top3p{power:g}_w{blend:.2f}"] = ((1.0 - blend) * anchor_pred + blend * top3, None)

    for threshold in ROUTE_THRESHOLDS:
        route_mask = max_proba >= threshold
        replace = anchor_pred.copy()
        replace[route_mask] = hard[route_mask]
        blend25 = anchor_pred.copy()
        blend25[route_mask] = 0.75 * anchor_pred[route_mask] + 0.25 * hard[route_mask]
        blend50 = anchor_pred.copy()
        blend50[route_mask] = 0.50 * anchor_pred[route_mask] + 0.50 * hard[route_mask]
        route_fraction = float(np.mean(route_mask))
        strategies[f"anchor_route_replace_p{threshold:.2f}"] = (replace, route_fraction)
        strategies[f"anchor_route_blend25_p{threshold:.2f}"] = (blend25, route_fraction)
        strategies[f"anchor_route_blend50_p{threshold:.2f}"] = (blend50, route_fraction)
    return strategies


def evaluate_hit_router_cv(
    hit_rows: np.ndarray,
    hit_labels: np.ndarray,
    hit_weights: np.ndarray,
    candidate_preds: np.ndarray,
    anchor_pred: np.ndarray,
    current_pred: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    sample_indices = np.repeat(np.arange(len(y)), len(CANDIDATES))
    rows = [{"seed": -1, "strategy": "selector_soft_anchor", "route_fraction": np.nan, **distance_summary(anchor_pred, y)}]
    for seed in CV_SEEDS:
        print(f"hit-router CV seed={seed}", flush=True)
        val_mask = split_mask(len(y), 0.2, seed)
        train_row_mask = ~val_mask[sample_indices]
        val_row_mask = val_mask[sample_indices]
        proba_flat = fit_predict_hit(
            hit_rows[train_row_mask],
            hit_labels[train_row_mask],
            hit_weights[train_row_mask],
            hit_rows[val_row_mask],
            81000 + seed,
        )
        val_indices = np.where(val_mask)[0]
        proba = proba_flat.reshape(len(val_indices), len(CANDIDATES))
        strategies = strategy_predictions(
            proba,
            candidate_preds[val_mask],
            anchor_pred[val_mask],
            current_pred[val_mask],
        )
        for name, (pred, route_fraction) in strategies.items():
            rows.append({"seed": seed, "strategy": name, "route_fraction": route_fraction, **distance_summary(pred, y[val_mask])})
    df = pd.DataFrame(rows)
    summary = (
        df[df["strategy"] != "selector_soft_anchor"]
        .groupby("strategy", as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
            route_fraction=("route_fraction", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    anchor_hit = float(distance_summary(anchor_pred, y)["r_hit_1cm"])
    summary["anchor_r_hit"] = anchor_hit
    summary["delta_hit_vs_anchor"] = summary["mean_r_hit"] - anchor_hit
    return summary


def candidate_hit_table(distances: np.ndarray) -> pd.DataFrame:
    rows = []
    labels = np.argmin(distances, axis=1)
    for idx, candidate in enumerate(CANDIDATES):
        rows.append(
            {
                "idx": idx,
                "candidate": candidate.name,
                "forward": candidate.mult[0],
                "side": candidate.mult[1],
                "up": candidate.mult[2],
                "best_distance_label_count": int(np.sum(labels == idx)),
                "candidate_hit_rate": float(np.mean(distances[:, idx] <= R_HIT)),
                "candidate_mean_distance": float(np.mean(distances[:, idx])),
                "candidate_median_distance": float(np.median(distances[:, idx])),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_hit_rate", "candidate_mean_distance"], ascending=[False, True])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate-aware hit probability router.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_hit_probability_router_20260516.md")
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

    print("Building feature matrices", flush=True)
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)

    print("Building 5-fold OOF direct-step local predictions", flush=True)
    oof_local = build_oof_direct_local(train_coords, y, train_features)
    train_candidate_preds = make_candidate_predictions(train_coords, oof_local)
    distances = np.linalg.norm(train_candidate_preds - y[:, None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    best_dist = np.min(distances, axis=1)
    current_idx = candidate_index(CURRENT_BEST_NAME)
    current_dist = distances[:, current_idx]
    selector_weights = label_weights(best_dist, current_dist)
    selector_features = make_selector_features(train_features, train_coords, oof_local)

    print("Building OOF selector-soft anchor", flush=True)
    selector_proba = build_oof_selector_proba(selector_features, labels, selector_weights)
    anchor_train = soft_predict(selector_proba, train_candidate_preds)
    current_train = train_candidate_preds[:, current_idx, :]

    print("Building candidate-aware hit probability rows", flush=True)
    hit_rows = build_hit_rows(selector_features, selector_proba, train_candidate_preds, oof_local)
    hit_labels = (distances <= R_HIT).astype(np.int8).reshape(-1)
    hit_weights = hit_sample_weights(distances)

    print("Evaluating hit-probability router CV", flush=True)
    leaderboard = evaluate_hit_router_cv(hit_rows, hit_labels, hit_weights, train_candidate_preds, anchor_train, current_train, y)
    print(leaderboard.head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Preparing test candidates", flush=True)
    source_pred = read_submission_coords(args.submission_dir / SOURCE_SUBMISSION)
    test_local = recover_test_local_from_source(test_coords, source_pred)
    test_candidate_preds = make_candidate_predictions(test_coords, test_local)
    test_selector_features = make_selector_features(test_features, test_coords, test_local)

    full_selector = make_selector(91616)
    full_selector.fit(selector_features, labels, sample_weight=selector_weights)
    test_selector_proba = aligned_predict_proba(full_selector, test_selector_features)
    test_hit_rows = build_hit_rows(test_selector_features, test_selector_proba, test_candidate_preds, test_local)
    print("Training full hit-probability router", flush=True)
    test_hit_proba = fit_predict_hit_ensemble(hit_rows, hit_labels, hit_weights, test_hit_rows).reshape(len(test_coords), len(CANDIDATES))

    public_anchor = read_submission_coords(args.submission_dir / PUBLIC_BEST_SUBMISSION)
    current_test = test_candidate_preds[:, current_idx, :]
    test_strategies = strategy_predictions(test_hit_proba, test_candidate_preds, public_anchor, current_test)
    output_rows = []
    written = []
    used = 0
    for rank, (_, row) in enumerate(leaderboard.iterrows(), start=1):
        name = str(row["strategy"])
        if name in {"selector_soft_anchor", "current_candidate"}:
            continue
        if name not in test_strategies:
            continue
        pred, route_fraction = test_strategies[name]
        used += 1
        path = args.submission_dir / f"hitprob_rank{used}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": used,
                "strategy": name,
                "submission": path.name,
                "cv_mean_r_hit": float(row["mean_r_hit"]),
                "cv_delta_vs_anchor": float(row["delta_hit_vs_anchor"]),
                "route_fraction": route_fraction if route_fraction is not None else row.get("route_fraction", np.nan),
                **delta_summary(pred, public_anchor, "vs_public_best"),
            }
        )
        if used >= args.top_k:
            break

    report = [
        "# 2026-05-16 Hit Probability Router",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Problem Redefinition",
        "",
        "- Official target: predict the +80ms 3D sensor-local coordinate from 11 historical coordinates sampled every 40ms.",
        "- Official metric: R-Hit@1cm, so a prediction is useful only when its 3D distance is <= 0.01m.",
        "- Previous selector models mostly learned the nearest candidate label. This experiment learns each candidate's hit probability directly.",
        "- No test labels or external trajectory data are used. Test features are only passed through models fitted on train data.",
        "",
        "References:",
        "",
        "- https://dacon.io/competitions/official/236716/overview/description",
        "- https://dacon.io/competitions/official/236716/overview/evaluation",
        "- https://dacon.io/competitions/official/236716/overview/rules",
        "",
        "## CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Candidate OOF Hit Table",
        "",
        dataframe_to_markdown(candidate_hit_table(distances)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- This is a metric-aligned router: rows are sample-candidate pairs and the binary label is candidate_hit = distance <= 0.01m.",
        "- Boundary samples around 1cm receive extra weight because a tiny movement can flip the leaderboard hit.",
        "- If public still trails 0.68440, the next reset should move away from candidate routing and into sequence/regime representation learning.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
