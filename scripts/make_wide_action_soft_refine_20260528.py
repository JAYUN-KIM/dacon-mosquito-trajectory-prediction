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

from mosquito_trajectory.data import COORD_COLUMNS, ID_COLUMN, read_sample_submission, resolve_raw_data_dir  # noqa: E402
from run_aggressive_experiments import dataframe_to_markdown  # noqa: E402
from run_curvature_gate_20260519 import delta_summary, read_submission_coords, write_submission  # noqa: E402


PUBLIC_WINNER = "recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv"
WIDE_SOFT08 = "wideact28_rank1_softblend08.csv"
WIDE_SOFT12 = "wideact28_rank2_softblend12.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear refine around the wide action soft blend direction.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_wide_action_soft_refine_20260528.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    winner = read_submission_coords(args.submission_dir / PUBLIC_WINNER)
    soft08 = read_submission_coords(args.submission_dir / WIDE_SOFT08)
    soft12 = read_submission_coords(args.submission_dir / WIDE_SOFT12)

    rows = []
    outputs = [
        ("wideact28_refine_softblend06.csv", winner + 0.75 * (soft08 - winner), "soft08_direction_x075"),
        ("wideact28_refine_softblend10.csv", winner + 1.25 * (soft08 - winner), "soft08_direction_x125"),
        ("wideact28_refine_softblend08_12avg.csv", 0.50 * soft08 + 0.50 * soft12, "avg_soft08_soft12"),
    ]
    for name, pred, rule in outputs:
        path = args.submission_dir / name
        write_submission(sample_submission, pred, path)
        rows.append(
            {
                "submission": name,
                "rule": rule,
                **delta_summary(pred, winner, "test_vs_winner"),
                **delta_summary(pred, soft08, "test_vs_soft08"),
            }
        )
        print(f"Wrote submission: {path}", flush=True)

    df = pd.DataFrame(rows)
    report = [
        "# 2026-05-28 Wide Action Soft Refine",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- base_winner: `{PUBLIC_WINNER}`",
        f"- soft08: `{WIDE_SOFT08}`",
        f"- soft12: `{WIDE_SOFT12}`",
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(df),
        "",
        "## Recommended Use",
        "",
        "1. Submit `wideact28_rank1_softblend08.csv` first.",
        "2. If it improves, try `wideact28_refine_softblend10.csv` or `wideact28_rank2_softblend12.csv`.",
        "3. If it drops slightly, try `wideact28_refine_softblend06.csv` as the safer same-direction version.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
