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
from run_direct_multiplier_selector_20260510 import build_oof_direct_local  # noqa: E402
from run_direct_step_refine_20260509 import direct_prediction  # noqa: E402
from run_hit_weighted_local_frame import make_features, safe_scale  # noqa: E402
from run_local_frame_residual import local_basis, normalize, to_global, to_local  # noqa: E402


SOURCE_SUBMISSION = "direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv"
SOURCE_MULT = (1.02, 1.00, 1.00)
CURRENT_BEST_SUBMISSION = "direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv"
CURRENT_BEST_MULT = (1.02, 1.06, 0.94)
CURRENT_BEST_SCORE = 0.68300

SMOOTH_CONFIGS = [
    ("w631_rawscale", [0.60, 0.30, 0.10], "raw"),
    ("w631_mixscale", [0.60, 0.30, 0.10], "mix"),
    ("w532_rawscale", [0.50, 0.30, 0.20], "raw"),
    ("w532_mixscale", [0.50, 0.30, 0.20], "mix"),
    ("w4321_rawscale", [0.40, 0.30, 0.20, 0.10], "raw"),
    ("w4321_mixscale", [0.40, 0.30, 0.20, 0.10], "mix"),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def basis_from_forward(forward_vectors: np.ndarray) -> np.ndarray:
    forward = normalize(forward_vectors, np.array([1.0, 0.0, 0.0]))
    z_ref = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(forward), axis=0)
    y_ref = np.repeat(np.array([[0.0, 1.0, 0.0]]), len(forward), axis=0)
    side_raw = np.cross(z_ref, forward)
    near_vertical = np.linalg.norm(side_raw, axis=1) < 1e-8
    side_raw[near_vertical] = np.cross(y_ref[near_vertical], forward[near_vertical])
    side = normalize(side_raw, np.array([0.0, 1.0, 0.0]))
    up = normalize(np.cross(forward, side), np.array([0.0, 0.0, 1.0]))
    return np.stack([forward, side, up], axis=1)


def smoothed_velocity(coords: np.ndarray, weights: list[float]) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    k = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    recent = diffs[:, -k:, :]
    return np.einsum("k,nkc->nc", w[::-1], recent)


def recover_scaled_local_prediction(test_coords: np.ndarray, source_pred: np.ndarray) -> np.ndarray:
    delta_local = to_local(source_pred - test_coords[:, -1, :], local_basis(test_coords))
    return delta_local / safe_scale(test_coords) / np.asarray(SOURCE_MULT, dtype=np.float64)[None, :]


def materialize_smoothed(
    coords: np.ndarray,
    scaled_local_pred: np.ndarray,
    mult: tuple[float, float, float],
    weights: list[float],
    scale_mode: str,
) -> np.ndarray:
    smooth_v = smoothed_velocity(coords, weights)
    smooth_basis = basis_from_forward(smooth_v)
    raw_scale = safe_scale(coords)
    smooth_scale = np.maximum(np.linalg.norm(smooth_v, axis=1, keepdims=True), 1e-4)
    if scale_mode == "raw":
        scale = raw_scale
    elif scale_mode == "smooth":
        scale = smooth_scale
    elif scale_mode == "mix":
        scale = 0.5 * raw_scale + 0.5 * smooth_scale
    else:
        raise ValueError(scale_mode)
    pred_local = scaled_local_pred * scale * np.asarray(mult, dtype=np.float64)[None, :]
    return coords[:, -1, :] + to_global(pred_local, smooth_basis)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def delta_summary(pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(pred - reference, axis=1)
    return {
        "mean_delta": float(np.mean(distances)),
        "median_delta": float(np.median(distances)),
        "p95_delta": float(np.quantile(distances, 0.95)),
        "max_delta": float(np.max(distances)),
    }


def evaluate_oof_smoothing(data_dir: Path) -> pd.DataFrame:
    print("Loading train data for OOF smoothing proxy", flush=True)
    train_samples = read_trajectory_folder(data_dir / "train")
    targets = read_targets(data_dir / "train_labels.csv")
    ids, missing = aligned_ids(train_samples, targets[ID_COLUMN].tolist())
    if missing:
        raise ValueError(f"{len(missing)} train ids are missing trajectory files, examples: {missing[:5]}")

    train_coords = stack_samples(train_samples, ids)
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)
    train_features, _ = make_features(train_coords)
    oof_local_scaled = build_oof_direct_local(train_coords, y, train_features)

    current_pred = direct_prediction(train_coords, oof_local_scaled, CURRENT_BEST_MULT)
    rows = [
        {
            "smooth": "current_best_basis",
            "weights": "-",
            "scale_mode": "original",
            **distance_summary(current_pred, y),
        }
    ]
    current_hit = rows[0]["r_hit_1cm"]
    current_mean = rows[0]["mean_distance"]

    for name, weights, scale_mode in SMOOTH_CONFIGS:
        pred = materialize_smoothed(train_coords, oof_local_scaled, CURRENT_BEST_MULT, weights, scale_mode)
        rows.append(
            {
                "smooth": name,
                "weights": str(weights),
                "scale_mode": scale_mode,
                **distance_summary(pred, y),
            }
        )

    df = pd.DataFrame(rows)
    df["delta_hit_vs_current"] = df["r_hit_1cm"] - current_hit
    df["delta_mean_distance_vs_current"] = df["mean_distance"] - current_mean
    return df.sort_values(["r_hit_1cm", "mean_distance"], ascending=[False, True]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Velocity-smoothing probes for the direct-step local prediction.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_direct_velocity_smoothing_probe_20260510.md")
    parser.add_argument("--skip-cv", action="store_true", help="Only generate test submissions without the OOF proxy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}", flush=True)

    test_samples = read_trajectory_folder(data_dir / "test")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    test_coords = stack_samples(test_samples, sample_submission[ID_COLUMN].tolist())

    source_path = args.output_dir / SOURCE_SUBMISSION
    best_path = args.output_dir / CURRENT_BEST_SUBMISSION
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    source_pred = pd.read_csv(source_path)[COORD_COLUMNS].to_numpy(dtype=np.float64)
    best_pred = pd.read_csv(best_path)[COORD_COLUMNS].to_numpy(dtype=np.float64)
    scaled_local_pred = recover_scaled_local_prediction(test_coords, source_pred)

    cv_df = None
    ordered_configs = SMOOTH_CONFIGS
    if not args.skip_cv:
        cv_df = evaluate_oof_smoothing(data_dir)
        print(cv_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
        config_by_name = {name: (name, weights, scale_mode) for name, weights, scale_mode in SMOOTH_CONFIGS}
        cv_names = cv_df.loc[cv_df["smooth"] != "current_best_basis", "smooth"].tolist()
        ordered_configs = [config_by_name[name] for name in cv_names if name in config_by_name]

    rows = []
    written = []
    for rank, (name, weights, scale_mode) in enumerate(ordered_configs, start=1):
        pred = materialize_smoothed(test_coords, scaled_local_pred, CURRENT_BEST_MULT, weights, scale_mode)
        path = args.output_dir / f"direct_smooth_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "smooth": name,
                "weights": str(weights),
                "scale_mode": scale_mode,
                **delta_summary(pred, best_pred),
            }
        )

    df = pd.DataFrame(rows)
    report = [
        "# 2026-05-10 Direct Velocity Smoothing Probe",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 데이터 경로: `{data_dir}`",
        f"- source submission: `{SOURCE_SUBMISSION}`",
        f"- current best: `{CURRENT_BEST_SUBMISSION} = {CURRENT_BEST_SCORE:.5f}`",
        f"- multiplier: `{CURRENT_BEST_MULT}`",
        f"- 생성한 제출 파일: `{[str(path) for path in written]}`",
        f"- OOF CV proxy: `{'skipped' if cv_df is None else 'enabled'}`",
        "",
    ]
    if cv_df is not None:
        report.extend(
            [
                "## OOF CV Proxy",
                "",
                dataframe_to_markdown(cv_df),
                "",
            ]
        )
    report.extend(
        [
            "## 제출 후보",
            "",
            dataframe_to_markdown(df),
            "",
            "## 해석",
            "",
            "- direct-step local prediction은 유지하고, local frame의 forward 방향만 최근 velocity 평균으로 부드럽게 바꾼 test-time probe입니다.",
            "- 관측 마지막 step의 방향 노이즈가 크다면 개선될 수 있지만, 현재 best 대비 이동량이 큰 후보는 public 리스크도 큽니다.",
            "- OOF CV proxy가 current_best_basis보다 낮으면 제출 우선순위를 낮추고, 높거나 비슷하면서 이동량이 작은 후보를 먼저 제출합니다.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
