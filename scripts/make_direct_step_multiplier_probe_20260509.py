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

from mosquito_trajectory.data import COORD_COLUMNS, ID_COLUMN, read_sample_submission, read_trajectory_folder, resolve_raw_data_dir  # noqa: E402
from run_aggressive_experiments import dataframe_to_markdown, stack_samples  # noqa: E402
from run_hit_weighted_local_frame import safe_scale  # noqa: E402
from run_local_frame_residual import local_basis, to_global, to_local  # noqa: E402


SOURCE_SUBMISSION = "direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv"
SOURCE_MULT = (1.02, 1.00, 1.00)
PUBLIC_BEST_SUBMISSION = "direct_refine_rank2_caa6s0055c0105_f1.02_s1.04_u0.96_5seed.csv"
PUBLIC_BEST_SCORE = 0.68260

PROBE_MULTS = [
    (1.02, 1.06, 0.94),
    (1.02, 1.08, 0.92),
    (1.02, 1.04, 0.94),
    (1.02, 1.06, 0.96),
    (1.03, 1.04, 0.96),
    (1.01, 1.04, 0.96),
    (1.02, 1.02, 0.98),
    (1.03, 1.06, 0.94),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def recover_scaled_local_prediction(test_coords: np.ndarray, source_pred: np.ndarray) -> np.ndarray:
    basis = local_basis(test_coords)
    delta_local = to_local(source_pred - test_coords[:, -1, :], basis)
    source_mult = np.asarray(SOURCE_MULT, dtype=np.float64)[None, :]
    return delta_local / safe_scale(test_coords) / source_mult


def materialize(test_coords: np.ndarray, scaled_local_pred: np.ndarray, mult: tuple[float, float, float]) -> np.ndarray:
    basis = local_basis(test_coords)
    pred_local = scaled_local_pred * safe_scale(test_coords)
    pred_local = pred_local * np.asarray(mult, dtype=np.float64)[None, :]
    return test_coords[:, -1, :] + to_global(pred_local, basis)


def delta_summary(pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(pred - reference, axis=1)
    return {
        "mean_delta": float(np.mean(distances)),
        "median_delta": float(np.median(distances)),
        "p95_delta": float(np.quantile(distances, 0.95)),
        "max_delta": float(np.max(distances)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast multiplier probes from the 0.6824 CA-boundary direct-step model.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_direct_multiplier_probe_20260509.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}", flush=True)

    test_samples = read_trajectory_folder(data_dir / "test")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    test_coords = stack_samples(test_samples, sample_submission[ID_COLUMN].tolist())

    source_path = args.output_dir / SOURCE_SUBMISSION
    best_path = args.output_dir / PUBLIC_BEST_SUBMISSION
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    source_pred = pd.read_csv(source_path)[COORD_COLUMNS].to_numpy(dtype=np.float64)
    best_pred = pd.read_csv(best_path)[COORD_COLUMNS].to_numpy(dtype=np.float64)
    scaled_local_pred = recover_scaled_local_prediction(test_coords, source_pred)

    rows = []
    written = []
    for rank, mult in enumerate(PROBE_MULTS, start=1):
        pred = materialize(test_coords, scaled_local_pred, mult)
        path = args.output_dir / f"direct_micro_rank{rank}_fromcaa6_f{mult[0]:.2f}_s{mult[1]:.2f}_u{mult[2]:.2f}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "forward_mult": mult[0],
                "side_mult": mult[1],
                "up_mult": mult[2],
                **delta_summary(pred, best_pred),
            }
        )

    df = pd.DataFrame(rows)
    report = [
        "# 2026-05-09 Direct-Step Multiplier Probe",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 데이터 경로: `{data_dir}`",
        f"- source submission: `{SOURCE_SUBMISSION}`",
        f"- source multiplier: `{SOURCE_MULT}`",
        f"- current public best: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- 생성한 제출 파일: `{[str(path) for path in written]}`",
        "",
        "## Probe 후보",
        "",
        dataframe_to_markdown(df),
        "",
        "## 해석",
        "",
        "- 기존 `0.6824` source prediction에서 local-step을 복원해, 재학습 없이 multiplier만 바꾼 후보입니다.",
        "- 현재 best인 `s1.04/u0.96` 주변에서 side를 더 키우고 up을 줄이는 방향, forward를 소폭 올리는 방향을 동시에 확인합니다.",
        "- 남은 제출 수가 제한적이면 `rank1 f1.02/s1.06/u0.94`와 `rank5 f1.03/s1.04/u0.96`을 우선 추천합니다.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
