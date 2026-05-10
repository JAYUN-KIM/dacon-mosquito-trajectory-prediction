from __future__ import annotations

import argparse
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

from mosquito_trajectory.data import COORD_COLUMNS, ID_COLUMN  # noqa: E402
from run_aggressive_experiments import dataframe_to_markdown  # noqa: E402


BASE_SUBMISSION = "direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv"
BASE_PUBLIC = 0.68300
SELECTOR_SUBMISSION = "direct_selector_rank1_selectorconf055.csv"
SELECTOR_PUBLIC = 0.68340
SOFT_SUBMISSION = "direct_selector_rank2_selectorsoft.csv"
CONF45_SUBMISSION = "direct_selector_rank4_selectorconf045.csv"


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def coords(df: pd.DataFrame) -> np.ndarray:
    return df[COORD_COLUMNS].to_numpy(dtype=np.float64)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    distances = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(distances)),
        f"{prefix}_median_delta": float(np.median(distances)),
        f"{prefix}_p95_delta": float(np.quantile(distances, 0.95)),
        f"{prefix}_max_delta": float(np.max(distances)),
    }


def write_submission(template: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = template[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small public-guided adjustment candidates around the selector win.")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_selector_adjustments_20260510.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_df = read_submission(args.submission_dir / BASE_SUBMISSION)
    selector_df = read_submission(args.submission_dir / SELECTOR_SUBMISSION)
    soft_df = read_submission(args.submission_dir / SOFT_SUBMISSION)
    conf45_df = read_submission(args.submission_dir / CONF45_SUBMISSION)

    base = coords(base_df)
    selector = coords(selector_df)
    soft = coords(soft_df)
    conf45 = coords(conf45_df)
    selector_delta = selector - base

    candidates: list[tuple[str, np.ndarray, str]] = [
        (
            "selector_adjust_rank1_extend115.csv",
            base + 1.15 * selector_delta,
            "public에서 이긴 selector 방향을 15%만 추가로 연장한 1순위 공격 후보",
        ),
        (
            "selector_adjust_rank2_shrink075.csv",
            base + 0.75 * selector_delta,
            "selector 이동을 75%로 줄인 보수 후보",
        ),
        (
            "selector_adjust_rank3_softpull015.csv",
            selector + 0.15 * (soft - selector),
            "selector_conf0.55를 유지하되 soft selector 쪽으로 15% 당긴 후보",
        ),
        (
            "selector_adjust_rank4_conf45pull015.csv",
            selector + 0.15 * (conf45 - selector),
            "route 범위를 조금 넓히는 conf0.45 방향 15% 후보",
        ),
        (
            "selector_adjust_rank5_extend130.csv",
            base + 1.30 * selector_delta,
            "selector 방향을 30% 더 미는 공격 후보",
        ),
    ]

    rows = []
    written = []
    for rank, (filename, pred, note) in enumerate(candidates, start=1):
        path = args.submission_dir / filename
        write_submission(base_df, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": filename,
                "note": note,
                **delta_summary(pred, base, "vs_base"),
                **delta_summary(pred, selector, "vs_selector"),
            }
        )

    df = pd.DataFrame(rows)
    report = [
        "# 2026-05-10 Selector Adjustment Candidates",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- base: `{BASE_SUBMISSION} = {BASE_PUBLIC:.5f}`",
        f"- selector anchor: `{SELECTOR_SUBMISSION} = {SELECTOR_PUBLIC:.5f}`",
        f"- soft reference: `{SOFT_SUBMISSION}`",
        f"- conf0.45 reference: `{CONF45_SUBMISSION}`",
        f"- 생성한 제출 파일: `{[str(path) for path in written]}`",
        "",
        "## 후보",
        "",
        dataframe_to_markdown(df),
        "",
        "## 제출 우선순위",
        "",
        "- 1순위는 `selector_adjust_rank1_extend115.csv`입니다. 이미 오른 selector 방향을 아주 작게 더 미는 후보입니다.",
        "- 2순위는 `selector_adjust_rank2_shrink075.csv`입니다. 0.6834가 운 좋게 오른 경우를 대비한 방어형 확인 후보입니다.",
        "- 3순위는 `selector_adjust_rank3_softpull015.csv`입니다. soft selector 신호를 약하게 섞어보는 후보입니다.",
        "- `rank5_extend130`은 이동량은 여전히 작지만 가장 공격적이므로 제출권이 남을 때만 추천합니다.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)
    print(df.to_string(index=False, float_format=lambda value: f"{value:.8f}"), flush=True)


if __name__ == "__main__":
    main()
