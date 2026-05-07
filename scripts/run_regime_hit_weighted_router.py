from __future__ import annotations

import argparse
import re
import sys
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

from mosquito_trajectory.data import (
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, physics_prediction, split_mask, stack_samples
from run_hit_weighted_breakthrough_refine import sample_weights_param
from run_hit_weighted_local_frame import fit_predict_axes_weighted, make_features
from run_local_frame_residual import local_basis, to_global, to_local


CV_SEEDS = [42]
FULL_SEEDS = [42, 777, 2026]
WEIGHT_CONFIG = ("base_a5_s0045", "base", 5.0, 0.0045, 0.0100)
REGIME_NAMES = {
    0: "vertical_high",
    1: "sharp_turn",
    2: "speed_high",
    3: "speed_low",
    4: "default",
}
ROUTE_ALPHAS = [0.25, 0.50, 0.75, 1.00]
SHRINKS = [
    (0.46, 0.58, 0.70),
    (0.48, 0.58, 0.70),
    (0.52, 0.58, 0.70),
    (0.52, 0.60, 0.70),
    (0.52, 0.58, 0.78),
    (0.56, 0.58, 0.70),
]
MIN_REGIME_SAMPLES = 500


def motion_scores(coords: np.ndarray) -> dict[str, np.ndarray]:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    basis = local_basis(coords)
    diffs_local = np.einsum("nwc,nkc->nwk", diffs, basis)

    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    dot = np.sum(d_last * d_prev, axis=1)
    denom = np.linalg.norm(d_last, axis=1) * np.linalg.norm(d_prev, axis=1) + 1e-12
    turn_cos = dot / denom

    last_speed = speed[:, -1]
    speed_ratio = last_speed / (speed.mean(axis=1) + 1e-8)
    accel_ratio = accel[:, -1] / (accel.mean(axis=1) + 1e-8)
    last_norm = np.linalg.norm(d_last, axis=1) + 1e-8
    vertical_ratio = np.abs(d_last[:, 2]) / last_norm
    prev_local_norm = np.linalg.norm(diffs_local[:, -2, :], axis=1) + 1e-8
    side_ratio = np.abs(diffs_local[:, -2, 1]) / prev_local_norm
    path_len = speed.sum(axis=1)
    displacement = np.linalg.norm(coords[:, -1, :] - coords[:, 0, :], axis=1)
    tortuosity = path_len / (displacement + 1e-8)

    return {
        "last_speed": last_speed,
        "speed_ratio": speed_ratio,
        "accel_ratio": accel_ratio,
        "turn_cos": turn_cos,
        "vertical_ratio": vertical_ratio,
        "side_ratio": side_ratio,
        "tortuosity": tortuosity,
    }


def fit_regime_thresholds(coords: np.ndarray) -> dict[str, float]:
    scores = motion_scores(coords)
    return {
        "speed_high": float(np.quantile(scores["speed_ratio"], 0.75)),
        "speed_low": float(np.quantile(scores["speed_ratio"], 0.25)),
        "sharp_turn": float(np.quantile(scores["turn_cos"], 0.25)),
        "vertical_high": float(np.quantile(scores["vertical_ratio"], 0.75)),
    }


def assign_regimes(coords: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    scores = motion_scores(coords)
    labels = np.full(len(coords), 4, dtype=np.int64)

    labels[scores["speed_ratio"] <= thresholds["speed_low"]] = 3
    labels[scores["speed_ratio"] >= thresholds["speed_high"]] = 2
    labels[scores["turn_cos"] <= thresholds["sharp_turn"]] = 1
    labels[scores["vertical_ratio"] >= thresholds["vertical_high"]] = 0
    return labels


def regime_count_table(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, name in REGIME_NAMES.items():
        rows.append({"regime": name, "count": int(np.sum(labels == label))})
    return pd.DataFrame(rows)


def fit_predict_regime_components(
    train_coords: np.ndarray,
    train_x: np.ndarray,
    train_target: np.ndarray,
    train_residual_local: np.ndarray,
    pred_coords: np.ndarray,
    pred_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    thresholds = fit_regime_thresholds(train_coords)
    train_labels = assign_regimes(train_coords, thresholds)
    pred_labels = assign_regimes(pred_coords, thresholds)
    weights = sample_weights_param(train_coords, train_target, WEIGHT_CONFIG)

    global_pred = fit_predict_axes_weighted(train_x, train_residual_local, pred_x, seed, "l2", weights)
    regime_pred = global_pred.copy()

    for label in sorted(REGIME_NAMES):
        train_mask = train_labels == label
        pred_mask = pred_labels == label
        if not pred_mask.any() or int(train_mask.sum()) < MIN_REGIME_SAMPLES:
            continue

        regime_pred_label = fit_predict_axes_weighted(
            train_x[train_mask],
            train_residual_local[train_mask],
            pred_x[pred_mask],
            seed + 100 + label * 17,
            "l2",
            weights[train_mask],
        )
        regime_pred[pred_mask] = regime_pred_label

    return global_pred, regime_pred, thresholds


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = physics_prediction(coords, 1.0, 0.275)
    residual_local = to_local(y - base, local_basis(coords))
    rows = []
    regime_rows = []

    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        y_val = y[val_mask]
        base_val = base[val_mask]
        basis_val = local_basis(coords[val_mask])

        global_local, regime_local, thresholds = fit_predict_regime_components(
            coords[train_mask],
            features[train_mask],
            y[train_mask],
            residual_local[train_mask],
            coords[val_mask],
            features[val_mask],
            seed,
        )
        val_labels = assign_regimes(coords[val_mask], thresholds)
        for _, row in regime_count_table(val_labels).iterrows():
            regime_rows.append({"seed": seed, **row.to_dict()})

        for forward, side, up in SHRINKS:
            shrink = np.array([forward, side, up], dtype=np.float64)
            global_pred = base_val + to_global(global_local * shrink[None, :], basis_val)
            rows.append(
                {
                    "strategy": "global_hit_weighted",
                    "route_alpha": 0.0,
                    "forward_shrink": forward,
                    "side_shrink": side,
                    "up_shrink": up,
                    "seed": seed,
                    **distance_summary(global_pred, y_val),
                }
            )

            for route_alpha in ROUTE_ALPHAS:
                routed_local = (1.0 - route_alpha) * global_local + route_alpha * regime_local
                routed_pred = base_val + to_global(routed_local * shrink[None, :], basis_val)
                rows.append(
                    {
                        "strategy": "regime_routed",
                        "route_alpha": route_alpha,
                        "forward_shrink": forward,
                        "side_shrink": side,
                        "up_shrink": up,
                        "seed": seed,
                        **distance_summary(routed_pred, y_val),
                    }
                )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(["strategy", "route_alpha", "forward_shrink", "side_shrink", "up_shrink"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    leaderboard["risk_adjusted_hit"] = leaderboard["mean_r_hit"] - 0.25 * leaderboard["std_r_hit"].fillna(0.0)
    return leaderboard, pd.DataFrame(regime_rows)


def full_regime_components(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    test_coords: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = physics_prediction(coords, 1.0, 0.275)
    residual_local = to_local(y - base, local_basis(coords))
    global_seed_preds = []
    regime_seed_preds = []
    for seed in FULL_SEEDS:
        global_local, regime_local, _ = fit_predict_regime_components(
            coords,
            features,
            y,
            residual_local,
            test_coords,
            test_features,
            seed,
        )
        global_seed_preds.append(global_local)
        regime_seed_preds.append(regime_local)
    return np.mean(global_seed_preds, axis=0), np.mean(regime_seed_preds, axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motion-regime routed hit-weighted local-frame models.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_regime_hit_weighted_router.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
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

    print("Building hit-weighted regime features", flush=True)
    train_features, candidate_names = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    print(f"feature_count={train_features.shape[1]}", flush=True)

    full_thresholds = fit_regime_thresholds(train_coords)
    full_train_regimes = assign_regimes(train_coords, full_thresholds)
    full_test_regimes = assign_regimes(test_coords, full_thresholds)
    train_regime_counts = regime_count_table(full_train_regimes)
    test_regime_counts = regime_count_table(full_test_regimes)

    print("Evaluating regime-routed CV", flush=True)
    leaderboard, cv_regime_counts = evaluate_cv(train_coords, y, train_features, args.val_frac)
    top = leaderboard.head(30)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    base_test = physics_prediction(test_coords, 1.0, 0.275)
    test_basis = local_basis(test_coords)
    print("Training full regime components", flush=True)
    full_global_local, full_regime_local = full_regime_components(train_coords, y, train_features, test_coords, test_features)
    written = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        route_alpha = float(row["route_alpha"])
        routed_local = (1.0 - route_alpha) * full_global_local + route_alpha * full_regime_local
        shrink = np.array([row["forward_shrink"], row["side_shrink"], row["up_shrink"]], dtype=np.float64)
        pred = base_test + to_global(routed_local * shrink[None, :], test_basis)
        path = (
            args.output_dir
            / (
                f"regime_hit_rank{rank}_{slug(row['strategy'])}_a{route_alpha:.2f}_"
                f"f{shrink[0]:.2f}_s{shrink[1]:.2f}_u{shrink[2]:.2f}.csv"
            )
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    best = top.iloc[0].to_dict()
    report = [
        "# Regime Hit-Weighted Router 실험",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 데이터 경로: `{data_dir}`",
        "- Public 앵커: `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv = 0.6722`",
        f"- feature 수: `{train_features.shape[1]}`",
        f"- hit-boundary weight 설정: `{WEIGHT_CONFIG}`",
        f"- CV seed: `{CV_SEEDS}`",
        f"- 전체 학습 ensemble seed: `{FULL_SEEDS}`",
        f"- regime blend alpha 후보: `{ROUTE_ALPHAS}`",
        f"- 축별 shrink 후보: `{SHRINKS}`",
        f"- regime threshold: `{full_thresholds}`",
        f"- candidate feature family: `{candidate_names}`",
        f"- CV best: `{best['strategy']}`, hit=`{best['mean_r_hit']:.6f}`, shrink=(`{best['forward_shrink']:.2f}`, `{best['side_shrink']:.2f}`, `{best['up_shrink']:.2f}`), alpha=`{best['route_alpha']:.2f}`",
        f"- 생성한 제출 파일: `{[str(path) for path in written]}`",
        "",
        "## Train Regime 분포",
        "",
        dataframe_to_markdown(train_regime_counts),
        "",
        "## Test Regime 분포",
        "",
        dataframe_to_markdown(test_regime_counts),
        "",
        "## CV Regime 분포",
        "",
        dataframe_to_markdown(cv_regime_counts.head(30)) if not cv_regime_counts.empty else "",
        "",
        "## Top 30 CV 결과",
        "",
        dataframe_to_markdown(top),
        "",
        "## 해석",
        "",
        "- 하나의 global hit-weighted local-frame 모델이 motion regime별로 너무 거친지 확인한 실험입니다.",
        "- 전역 z축 기준 vertical ratio, turn cosine, speed ratio로 regime을 나누고, regime 모델을 global 모델에 alpha만큼 섞었습니다.",
        "- 의미 있는 regime 기준으로 정정한 뒤에는 regime routing이 global 모델을 이기지 못했습니다.",
        "- 따라서 현재 결론은 regime별 독립 모델보다 hit-boundary weight와 shrink 조합을 더 밀어보는 쪽이 안전합니다.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
