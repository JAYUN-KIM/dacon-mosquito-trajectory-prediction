from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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

from mosquito_trajectory.data import (  # noqa: E402
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, physics_prediction, split_mask, stack_samples  # noqa: E402
from run_hit_weighted_local_frame import fit_predict_axes_weighted, make_features, physics_poly_candidates, safe_scale  # noqa: E402
from run_local_frame_residual import local_basis, to_global, to_local  # noqa: E402


CV_SEEDS = [42, 777]
FULL_SEEDS = [42, 777, 2026, 3407, 10007]
OLD_DIRECT_BEST = "direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv"
OLD_RESIDUAL_BEST = "hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv"

FORWARD_MULTS = [0.98, 1.00, 1.02, 1.04, 1.06, 1.08]
SIDE_MULTS = [0.96, 1.00, 1.04]
UP_MULTS = [0.96, 1.00, 1.04]
BLEND_OLD_DIRECT_WEIGHTS = [0.70, 0.80, 0.90]
BLEND_RESIDUAL_WEIGHTS = [0.85, 0.92]


@dataclass(frozen=True)
class WeightSpec:
    name: str
    source: str
    amplitude: float
    sigma: float
    center: float


WEIGHT_SPECS = [
    WeightSpec("uniform", "uniform", 0.0, 1.0, 0.0100),
    WeightSpec("ca_a5_s0045_c0100", "ca", 5.0, 0.0045, 0.0100),
    WeightSpec("ca_a6_s0055_c0105", "ca", 6.0, 0.0055, 0.0105),
    WeightSpec("ca_a4_s0035_c0095", "ca", 4.0, 0.0035, 0.0095),
    WeightSpec("cv_a5_s0045_c0100", "cv", 5.0, 0.0045, 0.0100),
    WeightSpec("cand_a5_s0035_c0100", "candidate", 5.0, 0.0035, 0.0100),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def multiplier_grid() -> list[tuple[float, float, float]]:
    return [(f, s, u) for f in FORWARD_MULTS for s in SIDE_MULTS for u in UP_MULTS]


def direct_target_local(coords: np.ndarray, y: np.ndarray) -> np.ndarray:
    target = to_local(y - coords[:, -1, :], local_basis(coords))
    return target / safe_scale(coords)


def direct_prediction(coords: np.ndarray, pred_local_scaled: np.ndarray, mult: tuple[float, float, float]) -> np.ndarray:
    pred_local = pred_local_scaled * safe_scale(coords)
    pred_local = pred_local * np.asarray(mult, dtype=np.float64)[None, :]
    return coords[:, -1, :] + to_global(pred_local, local_basis(coords))


def reference_error(coords: np.ndarray, y: np.ndarray, source: str) -> np.ndarray:
    if source == "ca":
        pred = physics_prediction(coords, 1.0, 0.275)
        return np.linalg.norm(pred - y, axis=1)
    if source == "cv":
        pred = physics_prediction(coords, 1.0, 0.0)
        return np.linalg.norm(pred - y, axis=1)
    if source == "candidate":
        candidates = physics_poly_candidates(coords)
        return np.linalg.norm(candidates - y[:, None, :], axis=2).min(axis=1)
    raise ValueError(f"unknown weight source: {source}")


def sample_weights(coords: np.ndarray, y: np.ndarray, spec: WeightSpec) -> np.ndarray:
    if spec.source == "uniform":
        return np.ones(len(coords), dtype=np.float64)

    dist = reference_error(coords, y, spec.source)
    weights = 1.0 + spec.amplitude * np.exp(-0.5 * ((dist - spec.center) / spec.sigma) ** 2)
    weights = np.clip(weights, 0.5, 8.0)
    return weights / np.mean(weights)


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray, val_frac: float) -> pd.DataFrame:
    rows = []
    grid = multiplier_grid()

    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(coords), val_frac, seed)
        train_mask = ~val_mask
        y_val = y[val_mask]

        for spec in WEIGHT_SPECS:
            print(f"  fitting {spec.name}", flush=True)
            train_y = direct_target_local(coords[train_mask], y[train_mask])
            weights = sample_weights(coords[train_mask], y[train_mask], spec)
            pred_local = fit_predict_axes_weighted(
                features[train_mask],
                train_y,
                features[val_mask],
                seed,
                "l2",
                weights,
            )
            for mult in grid:
                pred = direct_prediction(coords[val_mask], pred_local, mult)
                rows.append(
                    {
                        "weight_spec": spec.name,
                        "seed": seed,
                        "forward_mult": mult[0],
                        "side_mult": mult[1],
                        "up_mult": mult[2],
                        **distance_summary(pred, y_val),
                    }
                )

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(["weight_spec", "forward_mult", "side_mult", "up_mult"], as_index=False)
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


def full_direct_local_prediction(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    test_features: np.ndarray,
    spec: WeightSpec,
) -> np.ndarray:
    train_y = direct_target_local(coords, y)
    weights = sample_weights(coords, y, spec)
    seed_preds = []
    for seed in FULL_SEEDS:
        print(f"  full seed={seed} {spec.name}", flush=True)
        seed_preds.append(fit_predict_axes_weighted(features, train_y, test_features, seed, "l2", weights))
    return np.mean(seed_preds, axis=0)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def read_submission_coords(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine the 0.678 pure direct-step branch.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_direct_step_refine_20260509.md")
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

    print("Evaluating pure direct-step refine CV", flush=True)
    leaderboard = evaluate_cv(train_coords, y, train_features, args.val_frac)
    top = leaderboard.head(40)
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    specs = {spec.name: spec for spec in WEIGHT_SPECS}
    cache: dict[str, np.ndarray] = {}
    written: list[Path] = []
    pred_cache: dict[str, np.ndarray] = {}

    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        spec_name = str(row["weight_spec"])
        spec = specs[spec_name]
        if spec_name not in cache:
            print(f"Training full direct-step model: {spec_name}", flush=True)
            cache[spec_name] = full_direct_local_prediction(train_coords, y, train_features, test_features, spec)

        mult = (float(row["forward_mult"]), float(row["side_mult"]), float(row["up_mult"]))
        pred = direct_prediction(test_coords, cache[spec_name], mult)
        path = (
            args.output_dir
            / f"direct_refine_rank{rank}_{slug(spec_name)}_f{mult[0]:.2f}_s{mult[1]:.2f}_u{mult[2]:.2f}_5seed.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)
        pred_cache[path.name] = pred

    old_direct = read_submission_coords(args.output_dir / OLD_DIRECT_BEST)
    old_residual = read_submission_coords(args.output_dir / OLD_RESIDUAL_BEST)
    best_pred = next(iter(pred_cache.values())) if pred_cache else None

    if best_pred is not None and old_direct is not None:
        for weight in BLEND_OLD_DIRECT_WEIGHTS:
            pred = weight * best_pred + (1.0 - weight) * old_direct
            path = args.output_dir / f"direct_refine_blend_oldbest_w{weight:.2f}.csv"
            write_submission(sample_submission, pred, path)
            written.append(path)

    if best_pred is not None and old_residual is not None:
        for weight in BLEND_RESIDUAL_WEIGHTS:
            pred = weight * best_pred + (1.0 - weight) * old_residual
            path = args.output_dir / f"direct_refine_blend_residual_w{weight:.2f}.csv"
            write_submission(sample_submission, pred, path)
            written.append(path)

    best = top.iloc[0].to_dict()
    report = [
        "# 2026-05-09 Direct-Step Refine",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 데이터 경로: `{data_dir}`",
        "- Public anchor: `direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv = 0.67800`",
        f"- feature 수: `{train_features.shape[1]}`",
        f"- CV seed: `{CV_SEEDS}`",
        f"- full seed: `{FULL_SEEDS}`",
        f"- weight specs: `{WEIGHT_SPECS}`",
        f"- multiplier grid: forward=`{FORWARD_MULTS}`, side=`{SIDE_MULTS}`, up=`{UP_MULTS}`",
        f"- CV best: `{best['weight_spec']}`, hit=`{best['mean_r_hit']:.6f}`, mult=(`{best['forward_mult']:.2f}`, `{best['side_mult']:.2f}`, `{best['up_mult']:.2f}`)",
        f"- candidate feature family: `{candidate_names}`",
        f"- 생성한 제출 파일: `{[str(path) for path in written]}`",
        "",
        "## Top 40 CV 결과",
        "",
        dataframe_to_markdown(top),
        "",
        "## 해석",
        "",
        "- 어제 새 최고점이 나온 pure direct-step branch를 버리지 않고, multiplier와 hit-boundary weight를 확장했습니다.",
        "- full prediction은 기존 3seed보다 강한 5seed로 생성했습니다.",
        "- top 후보 외에도 기존 0.678 direct-step best 및 0.6722 residual anchor와의 blend 후보를 함께 만들었습니다.",
        "- public 결과가 좋으면 다음 단계는 direct-step 전용 feature와 weight center/sigma를 더 촘촘히 파는 방향입니다.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
