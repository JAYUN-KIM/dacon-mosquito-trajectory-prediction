from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


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
from run_hit_weighted_local_frame import fit_predict_axes_weighted, make_features, make_lgbm, safe_scale
from run_local_frame_residual import local_basis, to_global, to_local


warnings.filterwarnings("ignore", message="X does not have valid feature names")


CV_SEEDS = [42]
FULL_SEEDS = [42, 777, 2026]
BOUNDARY_WEIGHT = ("base_a5_s0045", "base", 5.0, 0.0045, 0.0100)
PUBLIC_ANCHOR = "hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv"

DIRECT_MULTS = [
    (0.94, 0.95, 0.95),
    (0.98, 0.95, 0.95),
    (0.98, 1.00, 1.00),
    (1.00, 1.00, 1.00),
    (1.02, 1.00, 1.00),
    (1.04, 1.00, 1.00),
    (1.00, 1.05, 1.05),
]
RESIDUAL_MULTS = [
    (0.44, 0.58, 0.70),
    (0.46, 0.58, 0.70),
    (0.48, 0.58, 0.70),
    (0.52, 0.58, 0.70),
    (0.56, 0.58, 0.70),
    (0.52, 0.58, 0.78),
    (0.52, 0.60, 0.70),
]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    base_kind: str
    scaled_target: bool
    model_kind: str
    weight_kind: str
    mults: tuple[tuple[float, float, float], ...]


EXPERIMENTS = [
    ExperimentSpec("direct_step_scaled_lgbm_uniform", "last", True, "lgbm", "uniform", tuple(DIRECT_MULTS)),
    ExperimentSpec("direct_step_scaled_lgbm_boundary", "last", True, "lgbm", "boundary", tuple(DIRECT_MULTS)),
    ExperimentSpec("direct_step_scaled_cat_uniform", "last", True, "catboost", "uniform", tuple(DIRECT_MULTS)),
    ExperimentSpec("cv_delta_scaled_lgbm_boundary", "cv", True, "lgbm", "boundary", tuple(RESIDUAL_MULTS)),
    ExperimentSpec("ca_delta_scaled_lgbm_boundary", "ca", True, "lgbm", "boundary", tuple(RESIDUAL_MULTS)),
    ExperimentSpec("ca_delta_scaled_cat_boundary", "ca", True, "catboost", "boundary", tuple(RESIDUAL_MULTS)),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def base_prediction(coords: np.ndarray, base_kind: str) -> np.ndarray:
    if base_kind == "last":
        return coords[:, -1, :]
    if base_kind == "cv":
        return physics_prediction(coords, 1.0, 0.0)
    if base_kind == "ca":
        return physics_prediction(coords, 1.0, 0.275)
    raise ValueError(f"unknown base_kind: {base_kind}")


def local_target(coords: np.ndarray, y: np.ndarray, spec: ExperimentSpec) -> np.ndarray:
    basis = local_basis(coords)
    target = to_local(y - base_prediction(coords, spec.base_kind), basis)
    if spec.scaled_target:
        target = target / safe_scale(coords)
    return target


def materialize_prediction(coords: np.ndarray, local_pred: np.ndarray, spec: ExperimentSpec, mult: tuple[float, float, float]) -> np.ndarray:
    basis = local_basis(coords)
    pred_local = local_pred.copy()
    if spec.scaled_target:
        pred_local = pred_local * safe_scale(coords)
    pred_local = pred_local * np.asarray(mult, dtype=np.float64)[None, :]
    return base_prediction(coords, spec.base_kind) + to_global(pred_local, basis)


def sample_weights(coords: np.ndarray, y: np.ndarray, spec: ExperimentSpec) -> np.ndarray:
    if spec.weight_kind == "uniform":
        return np.ones(len(coords), dtype=np.float64)
    if spec.weight_kind == "boundary":
        return sample_weights_param(coords, y, BOUNDARY_WEIGHT)
    raise ValueError(f"unknown weight_kind: {spec.weight_kind}")


def make_catboost(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=520,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=4.5,
        random_seed=seed,
        thread_count=-1,
        bootstrap_type="Bernoulli",
        subsample=0.90,
        rsm=0.92,
        allow_writing_files=False,
        verbose=False,
    )


def fit_predict_catboost_axes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    pred_x: np.ndarray,
    seed: int,
    weights: np.ndarray,
) -> np.ndarray:
    preds = []
    for axis in range(3):
        model = make_catboost(seed + axis)
        model.fit(train_x, train_y[:, axis], sample_weight=weights)
        preds.append(model.predict(pred_x))
    return np.vstack(preds).T


def fit_predict_axes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    pred_x: np.ndarray,
    seed: int,
    spec: ExperimentSpec,
    weights: np.ndarray,
) -> np.ndarray:
    if spec.model_kind == "lgbm":
        return fit_predict_axes_weighted(train_x, train_y, pred_x, seed, "l2", weights)
    if spec.model_kind == "catboost":
        return fit_predict_catboost_axes(train_x, train_y, pred_x, seed, weights)
    raise ValueError(f"unknown model_kind: {spec.model_kind}")


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    rows = []
    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        y_val = y[val_mask]

        for spec in EXPERIMENTS:
            print(f"  fitting {spec.name}", flush=True)
            target = local_target(coords[train_mask], y[train_mask], spec)
            weights = sample_weights(coords[train_mask], y[train_mask], spec)
            pred_local = fit_predict_axes(features[train_mask], target, features[val_mask], seed, spec, weights)
            for mult in spec.mults:
                pred = materialize_prediction(coords[val_mask], pred_local, spec, mult)
                rows.append(
                    {
                        "experiment": spec.name,
                        "base_kind": spec.base_kind,
                        "scaled_target": spec.scaled_target,
                        "model_kind": spec.model_kind,
                        "weight_kind": spec.weight_kind,
                        "seed": seed,
                        "forward_mult": mult[0],
                        "side_mult": mult[1],
                        "up_mult": mult[2],
                        **distance_summary(pred, y_val),
                    }
                )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(
            [
                "experiment",
                "base_kind",
                "scaled_target",
                "model_kind",
                "weight_kind",
                "forward_mult",
                "side_mult",
                "up_mult",
            ],
            as_index=False,
        )
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
    return leaderboard


def full_local_prediction(coords: np.ndarray, y: np.ndarray, features: np.ndarray, test_features: np.ndarray, spec: ExperimentSpec) -> np.ndarray:
    target = local_target(coords, y, spec)
    weights = sample_weights(coords, y, spec)
    seed_preds = []
    for seed in FULL_SEEDS:
        print(f"  full seed={seed} {spec.name}", flush=True)
        seed_preds.append(fit_predict_axes(features, target, test_features, seed, spec, weights))
    return np.mean(seed_preds, axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct local future-step geometry experiments.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_direct_step_geometry.md")
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

    print("Building direct-step feature matrix", flush=True)
    train_features, candidate_names = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    print(f"feature_count={train_features.shape[1]}", flush=True)

    print("Evaluating direct-step CV", flush=True)
    leaderboard = evaluate_cv(train_coords, y, train_features, args.val_frac)
    top = leaderboard.head(30)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    specs = {spec.name: spec for spec in EXPERIMENTS}
    cache: dict[str, np.ndarray] = {}
    written = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        spec_name = str(row["experiment"])
        spec = specs[spec_name]
        if spec_name not in cache:
            print(f"Training full model: {spec_name}", flush=True)
            cache[spec_name] = full_local_prediction(train_coords, y, train_features, test_features, spec)
        mult = (float(row["forward_mult"]), float(row["side_mult"]), float(row["up_mult"]))
        pred = materialize_prediction(test_coords, cache[spec_name], spec, mult)
        path = (
            args.output_dir
            / (
                f"direct_step_rank{rank}_{slug(spec_name)}_"
                f"f{mult[0]:.2f}_s{mult[1]:.2f}_u{mult[2]:.2f}.csv"
            )
        )
        write_submission(sample_submission, pred, path)
        written.append(path)

    best = top.iloc[0].to_dict()
    report = [
        "# Direct Step Geometry 실험",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 데이터 경로: `{data_dir}`",
        f"- 실험 전 Public 앵커: `{PUBLIC_ANCHOR} = 0.6722`",
        f"- feature 수: `{train_features.shape[1]}`",
        f"- CV seed: `{CV_SEEDS}`",
        f"- 전체 학습 ensemble seed: `{FULL_SEEDS}`",
        f"- 실험 후보: `{EXPERIMENTS}`",
        f"- candidate feature family: `{candidate_names}`",
        f"- CV best: `{best['experiment']}`, hit=`{best['mean_r_hit']:.6f}`, mult=(`{best['forward_mult']:.2f}`, `{best['side_mult']:.2f}`, `{best['up_mult']:.2f}`)",
        f"- 생성한 제출 파일: `{[str(path) for path in written]}`",
        "",
        "## Top 30 CV 결과",
        "",
        dataframe_to_markdown(top),
        "",
        "## 해석",
        "",
        "- residual 보정만 계속 밀지 않고, `+80ms 미래 step` 자체를 local frame에서 직접 예측하는 target 전환 실험입니다.",
        "- `direct_step_*`는 마지막 관측 좌표 기준 displacement를 직접 예측합니다.",
        "- `cv_delta_*`, `ca_delta_*`는 scale-normalized target을 유지하면서 물리 origin을 쓰는 편이 나은지 비교합니다.",
        "- pure direct-step 후보가 public에서 강하면 이 branch를 다음 메인 축으로 확장합니다.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
