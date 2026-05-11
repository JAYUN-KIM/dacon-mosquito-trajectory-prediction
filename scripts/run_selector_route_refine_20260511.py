from __future__ import annotations

import argparse
import itertools
import re
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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, split_mask, stack_samples  # noqa: E402
from run_direct_multiplier_selector_20260510 import (  # noqa: E402
    CA_A6_SPEC,
    SOURCE_SUBMISSION,
    build_oof_direct_local,
    make_selector_features,
    recover_test_local_from_source,
)
from run_direct_step_refine_20260509 import sample_weights  # noqa: E402
from run_hit_weighted_local_frame import make_features, safe_scale  # noqa: E402
from run_local_frame_residual import local_basis, to_global  # noqa: E402


BASE_SUBMISSION = "direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv"
SELECTOR_CONF55_SUBMISSION = "direct_selector_rank1_selectorconf055.csv"
SELECTOR_CONF45_SUBMISSION = "direct_selector_rank4_selectorconf045.csv"
PUBLIC_BEST_SUBMISSION = "selector_adjust_rank4_conf45pull015.csv"
PUBLIC_BEST_SCORE = 0.68360
CURRENT_MULT = np.asarray([1.02, 1.06, 0.94], dtype=np.float64)

ROUTE_PULL_WEIGHTS = [0.10, 0.20, 0.25, 0.30]
CONT_ALPHAS = [0.20, 0.35, 0.50]
CONT_BLEND_WEIGHTS = [0.15, 0.25]
CV_SEEDS = [42, 777, 2026]
REG_SEEDS = [113, 2027, 88011]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def coords(df: pd.DataFrame) -> np.ndarray:
    return df[COORD_COLUMNS].to_numpy(dtype=np.float64)


def write_submission(template: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = template[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    distances = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(distances)),
        f"{prefix}_median_delta": float(np.median(distances)),
        f"{prefix}_p95_delta": float(np.quantile(distances, 0.95)),
        f"{prefix}_max_delta": float(np.max(distances)),
    }


def direct_prediction_sample_mult(coords_arr: np.ndarray, pred_local_scaled: np.ndarray, mults: np.ndarray) -> np.ndarray:
    pred_local = pred_local_scaled * safe_scale(coords_arr) * mults
    return coords_arr[:, -1, :] + to_global(pred_local, local_basis(coords_arr))


def route_pull_grid(submission_dir: Path, sample_submission: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    selector = coords(read_submission(submission_dir / SELECTOR_CONF55_SUBMISSION))
    conf45 = coords(read_submission(submission_dir / SELECTOR_CONF45_SUBMISSION))
    public_best = coords(read_submission(submission_dir / PUBLIC_BEST_SUBMISSION))
    base = coords(read_submission(submission_dir / BASE_SUBMISSION))

    rows = []
    written = []
    for rank, weight in enumerate(ROUTE_PULL_WEIGHTS, start=1):
        pred = selector + weight * (conf45 - selector)
        path = submission_dir / f"route_refine_rank{rank}_conf45pull{int(weight * 1000):03d}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "route_pull_weight": weight,
                **delta_summary(pred, public_best, "vs_public_best"),
                **delta_summary(pred, base, "vs_base"),
            }
        )
    return pd.DataFrame(rows), written


def multiplier_grid() -> np.ndarray:
    forward = [1.00, 1.01, 1.02, 1.03, 1.04]
    side = [1.00, 1.04, 1.06, 1.08, 1.10, 1.12]
    up = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    return np.asarray(list(itertools.product(forward, side, up)), dtype=np.float64)


def make_regressor(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=520,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.06,
        reg_lambda=0.40,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def build_grid_oracle(
    train_coords: np.ndarray,
    y: np.ndarray,
    oof_local_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = multiplier_grid()
    preds = np.stack([direct_prediction_sample_mult(train_coords, oof_local_scaled, np.repeat(mult[None, :], len(train_coords), axis=0)) for mult in grid], axis=1)
    distances = np.linalg.norm(preds - y[:, None, :], axis=2)
    best_idx = np.argmin(distances, axis=1)
    best_dist = distances[np.arange(len(train_coords)), best_idx]
    current_pred = direct_prediction_sample_mult(train_coords, oof_local_scaled, np.repeat(CURRENT_MULT[None, :], len(train_coords), axis=0))
    current_dist = np.linalg.norm(current_pred - y, axis=1)
    labels = grid[best_idx].copy()
    improvement = current_dist - best_dist
    labels[improvement < 0.00035] = CURRENT_MULT
    return labels, best_dist, current_dist, current_pred


def multiplier_label_weights(best_dist: np.ndarray, current_dist: np.ndarray) -> np.ndarray:
    improvement = current_dist - best_dist
    weights = 1.0 + 4.5 * np.exp(-0.5 * ((current_dist - 0.010) / 0.0045) ** 2)
    weights += 3.0 * np.clip(improvement / 0.004, 0.0, 1.0)
    weights = np.clip(weights, 0.5, 9.0)
    return weights / np.mean(weights)


def fit_predict_mult(
    train_x: np.ndarray,
    train_y: np.ndarray,
    pred_x: np.ndarray,
    weights: np.ndarray,
    seeds: list[int],
) -> np.ndarray:
    seed_preds = []
    for seed in seeds:
        axis_preds = []
        for axis in range(3):
            model = make_regressor(seed + axis * 17)
            model.fit(train_x, train_y[:, axis], sample_weight=weights)
            axis_preds.append(model.predict(pred_x))
        seed_preds.append(np.vstack(axis_preds).T)
    pred = np.mean(seed_preds, axis=0)
    lo = np.asarray([0.98, 0.98, 0.86], dtype=np.float64)
    hi = np.asarray([1.06, 1.14, 1.02], dtype=np.float64)
    return np.clip(pred, lo[None, :], hi[None, :])


def evaluate_continuous_cv(
    selector_features: np.ndarray,
    label_mults: np.ndarray,
    weights: np.ndarray,
    train_coords: np.ndarray,
    oof_local_scaled: np.ndarray,
    y: np.ndarray,
    current_pred: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for seed in CV_SEEDS:
        val_mask = split_mask(len(y), 0.2, seed)
        train_mask = ~val_mask
        pred_mult = fit_predict_mult(
            selector_features[train_mask],
            label_mults[train_mask],
            selector_features[val_mask],
            weights[train_mask],
            [seed],
        )
        y_val = y[val_mask]
        current_val = current_pred[val_mask]
        rows.append({"seed": seed, "strategy": "current_mult", **distance_summary(current_val, y_val)})
        for alpha in CONT_ALPHAS:
            mult = CURRENT_MULT[None, :] + alpha * (pred_mult - CURRENT_MULT[None, :])
            pred = direct_prediction_sample_mult(train_coords[val_mask], oof_local_scaled[val_mask], mult)
            rows.append({"seed": seed, "strategy": f"cont_alpha{alpha:.2f}", **distance_summary(pred, y_val)})

    df = pd.DataFrame(rows)
    summary = (
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
    summary["risk_adjusted_hit"] = summary["mean_r_hit"] - 0.25 * summary["std_r_hit"].fillna(0.0)
    return summary


def continuous_multiplier_outputs(
    data_dir: Path,
    submission_dir: Path,
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    print("Loading train/test data for continuous multiplier regression", flush=True)
    train_samples = read_trajectory_folder(data_dir / "train")
    test_samples = read_trajectory_folder(data_dir / "test")
    targets = read_targets(data_dir / "train_labels.csv")
    ids, missing = aligned_ids(train_samples, targets[ID_COLUMN].tolist())
    if missing:
        raise ValueError(f"{len(missing)} train ids are missing trajectory files, examples: {missing[:5]}")

    train_coords = stack_samples(train_samples, ids)
    test_coords = stack_samples(test_samples, sample_submission[ID_COLUMN].tolist())
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)

    print("Building OOF local direct predictions", flush=True)
    oof_local_scaled = build_oof_direct_local(train_coords, y, train_features)
    print("Building grid-oracle multiplier labels", flush=True)
    label_mults, best_dist, current_dist, current_pred = build_grid_oracle(train_coords, y, oof_local_scaled)
    weights = multiplier_label_weights(best_dist, current_dist)
    selector_features = make_selector_features(train_features, train_coords, oof_local_scaled)

    print("Evaluating continuous multiplier CV", flush=True)
    cv_summary = evaluate_continuous_cv(selector_features, label_mults, weights, train_coords, oof_local_scaled, y, current_pred)
    print(cv_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    source_pred = coords(read_submission(submission_dir / SOURCE_SUBMISSION))
    test_local_scaled = recover_test_local_from_source(test_coords, source_pred)
    test_selector_features = make_selector_features(test_features, test_coords, test_local_scaled)
    pred_mult_raw = fit_predict_mult(selector_features, label_mults, test_selector_features, weights, REG_SEEDS)

    public_best = coords(read_submission(submission_dir / PUBLIC_BEST_SUBMISSION))
    rows = []
    written = []
    rank = 1
    for alpha in CONT_ALPHAS:
        mult = CURRENT_MULT[None, :] + alpha * (pred_mult_raw - CURRENT_MULT[None, :])
        cont_pred = direct_prediction_sample_mult(test_coords, test_local_scaled, mult)
        for blend_weight in CONT_BLEND_WEIGHTS:
            pred = (1.0 - blend_weight) * public_best + blend_weight * cont_pred
            path = submission_dir / f"contmult_rank{rank}_a{int(alpha * 100):02d}_blend{int(blend_weight * 100):02d}.csv"
            write_submission(sample_submission, pred, path)
            written.append(path)
            rows.append(
                {
                    "rank": rank,
                    "submission": path.name,
                    "alpha": alpha,
                    "blend_with_continuous": blend_weight,
                    "mean_mult_f": float(np.mean(mult[:, 0])),
                    "mean_mult_s": float(np.mean(mult[:, 1])),
                    "mean_mult_u": float(np.mean(mult[:, 2])),
                    **delta_summary(pred, public_best, "vs_public_best"),
                }
            )
            rank += 1
    return cv_summary, pd.DataFrame(rows), written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2026-05-11 selector route refinement and continuous multiplier probes.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_selector_route_refine_20260511.md")
    parser.add_argument("--skip-continuous", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")

    print("Building route pull grid", flush=True)
    route_df, route_paths = route_pull_grid(args.submission_dir, sample_submission)
    cont_cv = None
    cont_df = pd.DataFrame()
    cont_paths: list[Path] = []
    if not args.skip_continuous:
        cont_cv, cont_df, cont_paths = continuous_multiplier_outputs(data_dir, args.submission_dir, sample_submission)

    report = [
        "# 2026-05-11 Selector Route Refine",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- route_grid_outputs: `{[str(path) for path in route_paths]}`",
        f"- continuous_outputs: `{[str(path) for path in cont_paths]}`",
        "",
        "## Route Pull Grid",
        "",
        dataframe_to_markdown(route_df),
        "",
    ]
    if cont_cv is not None:
        report.extend(
            [
                "## Continuous Multiplier CV",
                "",
                dataframe_to_markdown(cont_cv),
                "",
                "## Continuous Multiplier Outputs",
                "",
                dataframe_to_markdown(cont_df),
                "",
            ]
        )
    report.extend(
        [
            "## Notes",
            "",
            "- Route grid keeps the 2026-05-10 selector_conf0.55 anchor and changes only the pull toward selector_conf0.45.",
            "- Continuous multiplier regression predicts per-sample forward/side/up multipliers from OOF grid-oracle labels, then blends cautiously with the public-best anchor.",
            "- If continuous CV is below current_mult, prioritize route grid submissions and treat continuous outputs as exploratory only.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote report: {args.report_path}", flush=True)
    for path in route_paths + cont_paths:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
