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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, physics_prediction, stack_samples  # noqa: E402
from run_constant_turn_curvature_20260518 import TurnConfig, constant_turn_prediction  # noqa: E402
from run_curvature_gate_20260519 import make_gate_features, read_submission_coords, write_submission  # noqa: E402
from run_direct_multiplier_selector_20260510 import make_folds  # noqa: E402
from run_hit_weighted_local_frame import make_features  # noqa: E402


R_HIT = 0.01
TEMPORAL_ANCHOR = "temporalbc_refine_r1f102s100u100_w55.csv"
CURRENT_BEST = "curvgate_refine_rank2_gatet52a105.csv"
CURRENT_BEST_SCORE = 0.69120
N_FOLDS = 5
FOLD_SEED = 20260520


@dataclass(frozen=True)
class Action:
    name: str
    cfg: TurnConfig | None
    alpha: float


CURVATURE_CONFIGS = [
    TurnConfig("w1_tm0p25_s0p5_d0p98", 1, -0.25, 0.50, 0.98),
    TurnConfig("w1_tm0p25_s0p25_d0p98", 1, -0.25, 0.25, 0.98),
    TurnConfig("w1_tm0p25_s0p5_d1p00", 1, -0.25, 0.50, 1.00),
    TurnConfig("w1_tm0p25_s0p0_d0p98", 1, -0.25, 0.00, 0.98),
    TurnConfig("w2_tm0p25_s0p5_d0p98", 2, -0.25, 0.50, 0.98),
    TurnConfig("w1_t0p25_s0p5_d0p98", 1, 0.25, 0.50, 0.98),
    TurnConfig("w1_t0p50_s0p5_d0p98", 1, 0.50, 0.50, 0.98),
]
ALPHAS = [0.075, 0.095, 0.105, 0.120]
ACTIONS = [Action("none", None, 0.0)] + [
    Action(f"{cfg.name}_a{int(round(alpha * 1000)):03d}", cfg, alpha)
    for cfg in CURVATURE_CONFIGS
    for alpha in ALPHAS
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def unit_or_zero(values: np.ndarray, scale: float) -> np.ndarray:
    return values / scale if scale != 0 else values


def build_action_deltas(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv = physics_prediction(coords, 1.0, 0.0)
    correction_by_name = {
        cfg.name: constant_turn_prediction(coords, cfg) - cv
        for cfg in CURVATURE_CONFIGS
    }
    deltas = []
    meta = []
    for action in ACTIONS:
        if action.cfg is None:
            deltas.append(np.zeros((len(coords), 3), dtype=np.float64))
            meta.append([0.0, 0.0, 0.0, 1.0, 0.0])
        else:
            deltas.append(action.alpha * correction_by_name[action.cfg.name])
            meta.append(
                [
                    action.cfg.rot_window / 3.0,
                    action.cfg.turn_scale,
                    action.cfg.speed_scale,
                    action.cfg.disp_scale,
                    action.alpha,
                ]
            )
    return np.stack(deltas, axis=1), np.asarray(meta, dtype=np.float32)


def action_hit_weights(distances: np.ndarray) -> np.ndarray:
    weights = 1.0 + 8.0 * np.exp(-0.5 * ((distances - R_HIT) / 0.0030) ** 2)
    weights += 1.5 * (distances <= R_HIT)
    weights = np.clip(weights, 0.5, 12.0)
    return (weights / np.mean(weights)).reshape(-1)


def make_router_rows(
    coords: np.ndarray,
    sample_context: np.ndarray,
    anchor: np.ndarray,
    action_deltas: np.ndarray,
    action_meta: np.ndarray,
) -> np.ndarray:
    n_rows, n_actions, _ = action_deltas.shape
    repeated_context = np.repeat(sample_context.astype(np.float32), n_actions, axis=0)

    diffs = np.diff(coords, axis=1)
    d_last = diffs[:, -1, :]
    anchor_delta = anchor - coords[:, -1, :]
    pred_delta = anchor_delta[:, None, :] + action_deltas

    action_norm = np.linalg.norm(action_deltas, axis=2, keepdims=True)
    pred_norm = np.linalg.norm(pred_delta, axis=2, keepdims=True)
    last_norm = np.linalg.norm(d_last, axis=1, keepdims=True)[:, None, :]
    anchor_norm = np.linalg.norm(anchor_delta, axis=1, keepdims=True)[:, None, :]
    dot_last = np.sum(action_deltas * d_last[:, None, :], axis=2, keepdims=True)
    dot_anchor = np.sum(action_deltas * anchor_delta[:, None, :], axis=2, keepdims=True)
    cos_last = dot_last / (action_norm * last_norm + 1e-12)
    cos_anchor = dot_anchor / (action_norm * anchor_norm + 1e-12)
    meta = np.broadcast_to(action_meta[None, :, :], (n_rows, n_actions, action_meta.shape[1]))

    action_context = np.concatenate(
        [
            action_deltas,
            np.abs(action_deltas),
            action_norm,
            pred_delta,
            pred_norm,
            dot_last,
            cos_last,
            dot_anchor,
            cos_anchor,
            meta,
        ],
        axis=2,
    ).reshape(n_rows * n_actions, -1)
    out = np.hstack([repeated_context, action_context]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def make_hit_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=560,
        learning_rate=0.020,
        num_leaves=47,
        min_child_samples=42,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.10,
        reg_lambda=0.90,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_hit(train_x: np.ndarray, labels: np.ndarray, weights: np.ndarray, pred_x: np.ndarray, seed: int) -> np.ndarray:
    model = make_hit_model(seed)
    model.fit(train_x, labels, sample_weight=weights)
    return model.predict_proba(pred_x)[:, 1]


def pick_by_indices(action_preds: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return action_preds[np.arange(len(indices)), indices]


def weighted_average(proba: np.ndarray, action_preds: np.ndarray, power: float, top_k: int | None) -> np.ndarray:
    weights = np.clip(proba, 1e-8, 1.0) ** power
    if top_k is not None and top_k < weights.shape[1]:
        keep_idx = np.argpartition(weights, -top_k, axis=1)[:, -top_k:]
        mask = np.zeros_like(weights, dtype=bool)
        rows = np.arange(len(weights))[:, None]
        mask[rows, keep_idx] = True
        weights = np.where(mask, weights, 0.0)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return np.einsum("na,nad->nd", weights, action_preds)


def strategy_predictions(proba: np.ndarray, action_preds: np.ndarray, current_best: np.ndarray | None = None) -> dict[str, np.ndarray]:
    hard_idx = np.argmax(proba, axis=1)
    hard = pick_by_indices(action_preds, hard_idx)
    top3_p2 = weighted_average(proba, action_preds, power=2.0, top_k=3)
    top5_p2 = weighted_average(proba, action_preds, power=2.0, top_k=5)
    top5_p4 = weighted_average(proba, action_preds, power=4.0, top_k=5)
    strategies = {
        "action_hard": hard,
        "action_top3_p2": top3_p2,
        "action_top5_p2": top5_p2,
        "action_top5_p4": top5_p4,
    }
    if current_best is not None:
        for name, pred in list(strategies.items()):
            for blend in [0.25, 0.35, 0.50]:
                strategies[f"currentblend{int(blend * 100):02d}_{name}"] = (1.0 - blend) * current_best + blend * pred
    return strategies


def evaluate_router_cv(
    rows: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    action_preds: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    sample_indices = np.repeat(np.arange(len(y)), len(ACTIONS))
    folds = make_folds(len(y), N_FOLDS, FOLD_SEED)
    out_rows = []
    for fold_idx, val_idx in enumerate(folds, start=1):
        val_mask = np.zeros(len(y), dtype=bool)
        val_mask[val_idx] = True
        train_row_mask = ~val_mask[sample_indices]
        val_row_mask = val_mask[sample_indices]
        print(f"multi-curvature action CV fold {fold_idx}/{N_FOLDS}", flush=True)
        proba_flat = fit_predict_hit(
            rows[train_row_mask],
            labels[train_row_mask],
            weights[train_row_mask],
            rows[val_row_mask],
            720000 + fold_idx * 41,
        )
        proba = proba_flat.reshape(len(val_idx), len(ACTIONS))
        strategies = strategy_predictions(proba, action_preds[val_idx], current_best=None)
        for name, pred in strategies.items():
            out_rows.append({"fold": fold_idx, "strategy": name, **distance_summary(pred, y[val_idx])})
    df = pd.DataFrame(out_rows)
    return (
        df.groupby("strategy", as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )


def fit_predict_full_router(rows: np.ndarray, labels: np.ndarray, weights: np.ndarray, test_rows: np.ndarray) -> np.ndarray:
    preds = []
    for seed in [1301, 2718, 3141]:
        print(f"full multi-curvature action seed={seed}", flush=True)
        preds.append(fit_predict_hit(rows, labels, weights, test_rows, 730000 + seed))
    return np.mean(preds, axis=0)


def action_table(distances: np.ndarray) -> pd.DataFrame:
    rows = []
    hit_rate = np.mean(distances <= R_HIT, axis=0)
    mean_distance = np.mean(distances, axis=0)
    label_counts = np.bincount(np.argmin(distances, axis=1), minlength=len(ACTIONS))
    for idx, action in enumerate(ACTIONS):
        rows.append(
            {
                "idx": idx,
                "action": action.name,
                "alpha": action.alpha,
                "label_count": int(label_counts[idx]),
                "oof_hit_rate": float(hit_rate[idx]),
                "oof_mean_distance": float(mean_distance[idx]),
            }
        )
    return pd.DataFrame(rows).sort_values(["oof_hit_rate", "oof_mean_distance"], ascending=[False, True])


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-curvature action hit-probability router.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_multi_curvature_action_router_20260520.md")
    parser.add_argument("--top-k", type=int, default=6)
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
    print("Building features and action candidates", flush=True)
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    train_action_deltas, action_meta = build_action_deltas(train_coords)
    test_action_deltas, _ = build_action_deltas(test_coords)

    # Use the known best correction context as the sample-level feature scaffold.
    best_delta_train = train_action_deltas[:, [action.name for action in ACTIONS].index("w1_tm0p25_s0p5_d0p98_a105"), :]
    best_delta_test = test_action_deltas[:, [action.name for action in ACTIONS].index("w1_tm0p25_s0p5_d0p98_a105"), :]
    train_context = make_gate_features(train_coords, train_features, anchor_oof, best_delta_train)
    temporal_anchor_test = read_submission_coords(args.submission_dir / TEMPORAL_ANCHOR)
    current_best_test = read_submission_coords(args.submission_dir / CURRENT_BEST)
    test_context = make_gate_features(test_coords, test_features, temporal_anchor_test, best_delta_test)

    train_action_preds = anchor_oof[:, None, :] + train_action_deltas
    train_distances = np.linalg.norm(train_action_preds - y[:, None, :], axis=2)
    labels = (train_distances <= R_HIT).astype(np.int8).reshape(-1)
    weights = action_hit_weights(train_distances)
    train_rows = make_router_rows(train_coords, train_context, anchor_oof, train_action_deltas, action_meta)

    print("Evaluating multi-curvature action router CV", flush=True)
    leaderboard = evaluate_router_cv(train_rows, labels, weights, train_action_preds, y)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Training full action router", flush=True)
    test_rows = make_router_rows(test_coords, test_context, temporal_anchor_test, test_action_deltas, action_meta)
    test_proba = fit_predict_full_router(train_rows, labels, weights, test_rows).reshape(len(test_coords), len(ACTIONS))
    test_action_preds = temporal_anchor_test[:, None, :] + test_action_deltas
    test_strategies = strategy_predictions(test_proba, test_action_preds, current_best=current_best_test)

    preferred = [
        "currentblend35_action_top3_p2",
        "currentblend25_action_top3_p2",
        "currentblend35_action_top5_p4",
        "currentblend50_action_top3_p2",
        "currentblend35_action_hard",
        "action_top3_p2",
    ]
    output_rows = []
    written: list[Path] = []
    for rank, strategy in enumerate(preferred, start=1):
        pred = test_strategies[strategy]
        path = args.submission_dir / f"multicurv_action_rank{rank}_{slug(strategy)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "strategy": strategy,
                **delta_summary(pred, current_best_test, "vs_current_best"),
                **delta_summary(pred, temporal_anchor_test, "vs_temporal_anchor"),
            }
        )
        if len(written) >= args.top_k:
            break

    report = [
        "# 2026-05-20 Multi-Curvature Action Router",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- current_best: `{CURRENT_BEST} = {CURRENT_BEST_SCORE:.5f}`",
        f"- action_count: `{len(ACTIONS)}`",
        "- idea: instead of a single curvature gate, learn hit probability for multiple curvature config/alpha actions.",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Action OOF Table",
        "",
        dataframe_to_markdown(action_table(train_distances)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- This is a new curvature policy axis, not threshold-only tuning.",
        "- Public risk is controlled by blending action-router outputs back toward the current best.",
        "- If blended outputs improve, expand action pool and train an alpha-bucket policy.",
        "- If pure action output improves, the sample-wise curvature choice is stronger than the previous binary gate.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
