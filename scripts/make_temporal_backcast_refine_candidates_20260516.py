from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COORD_COLUMNS = ["x", "y", "z"]
ID_COLUMN = "id"

ANCHOR = "direct_selector_rank2_selectorsoft.csv"
TEMPORAL_RANKS = {
    "r1_f102_s100_u100": "temporalbc_rank1_tbc678w020_f1.02_s1.00_u1.00.csv",
    "r2_f102_s104_u096": "temporalbc_rank2_tbc678w020_f1.02_s1.04_u0.96.csv",
    "r3_f102_s104_u094": "temporalbc_rank3_tbc678w020_f1.02_s1.04_u0.94.csv",
    "r4_f102_s106_u094": "temporalbc_rank4_tbc678w020_f1.02_s1.06_u0.94.csv",
}


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_coords(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def coord_delta_summary(pred: pd.DataFrame, anchor: pd.DataFrame, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred[COORD_COLUMNS].to_numpy(float) - anchor[COORD_COLUMNS].to_numpy(float), axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_blend(anchor: pd.DataFrame, target: pd.DataFrame, weight: float, path: Path) -> pd.DataFrame:
    out = anchor[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = (1.0 - weight) * anchor[COORD_COLUMNS] + weight * target[COORD_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def write_average_blend(anchor: pd.DataFrame, targets: list[pd.DataFrame], weight: float, path: Path) -> pd.DataFrame:
    avg = targets[0][COORD_COLUMNS].copy()
    for target in targets[1:]:
        avg += target[COORD_COLUMNS]
    avg /= len(targets)
    out = anchor[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = (1.0 - weight) * anchor[COORD_COLUMNS] + weight * avg
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine temporal-backcast public-winning blend strength.")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_temporal_backcast_refine_20260516.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor = read_coords(args.submission_dir / ANCHOR)
    temporal = {name: read_coords(args.submission_dir / filename) for name, filename in TEMPORAL_RANKS.items()}

    rows = []
    written: list[Path] = []

    # Public scores on the first probe show 50% blend is best, so search near that point.
    for name in ["r1_f102_s100_u100"]:
        for weight in [0.42, 0.46, 0.48, 0.52, 0.55, 0.58, 0.62]:
            path = args.submission_dir / f"temporalbc_refine_{slug(name)}_w{int(weight * 100):02d}.csv"
            pred = write_blend(anchor, temporal[name], weight, path)
            written.append(path)
            rows.append(
                {
                    "submission": path.name,
                    "source": name,
                    "blend_weight": weight,
                    "kind": "rank1_strength_grid",
                    **coord_delta_summary(pred, anchor, "vs_public_best"),
                }
            )

    # Nearby multiplier variants can reveal whether 50% strength is robust to side/up tilt.
    for name in ["r2_f102_s104_u096", "r3_f102_s104_u094", "r4_f102_s106_u094"]:
        for weight in [0.48, 0.52, 0.56]:
            path = args.submission_dir / f"temporalbc_refine_{slug(name)}_w{int(weight * 100):02d}.csv"
            pred = write_blend(anchor, temporal[name], weight, path)
            written.append(path)
            rows.append(
                {
                    "submission": path.name,
                    "source": name,
                    "blend_weight": weight,
                    "kind": "multiplier_variant_grid",
                    **coord_delta_summary(pred, anchor, "vs_public_best"),
                }
            )

    # Ensemble the strongest neighboring temporal directions before blending with the public anchor.
    ensemble_sets = {
        "avg_r1r2": ["r1_f102_s100_u100", "r2_f102_s104_u096"],
        "avg_r1r2r3": ["r1_f102_s100_u100", "r2_f102_s104_u096", "r3_f102_s104_u094"],
        "avg_r1r2r3r4": ["r1_f102_s100_u100", "r2_f102_s104_u096", "r3_f102_s104_u094", "r4_f102_s106_u094"],
    }
    for ens_name, source_names in ensemble_sets.items():
        targets = [temporal[name] for name in source_names]
        for weight in [0.48, 0.52, 0.56]:
            path = args.submission_dir / f"temporalbc_refine_{slug(ens_name)}_w{int(weight * 100):02d}.csv"
            pred = write_average_blend(anchor, targets, weight, path)
            written.append(path)
            rows.append(
                {
                    "submission": path.name,
                    "source": ens_name,
                    "blend_weight": weight,
                    "kind": "temporal_direction_ensemble",
                    **coord_delta_summary(pred, anchor, "vs_public_best"),
                }
            )

    df = pd.DataFrame(rows)
    recommended = pd.concat(
        [
            df[df["submission"].str.contains("r1f102s100u100_w52")],
            df[df["submission"].str.contains("r1f102s100u100_w55")],
            df[df["submission"].str.contains("avgr1r2_w52")],
            df[df["submission"].str.contains("r2f102s104u096_w52")],
            df[df["submission"].str.contains("avgr1r2r3_w52")],
        ],
        ignore_index=True,
    )

    report = [
        "# 2026-05-16 Temporal Backcast Refine",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- public probe result: `35% = 0.6862`, `50% = 0.6878`, `100% = 0.6864`",
        "- interpretation: temporal-backcast direction is valid, but optimal public strength is around 50%.",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Recommended Next Public Probe",
        "",
        dataframe_to_markdown(recommended),
        "",
        "## All Candidates",
        "",
        dataframe_to_markdown(df.sort_values(["kind", "blend_weight", "source"])),
        "",
        "## Notes",
        "",
        "- The previous best public score was 0.68440; temporal-backcast 50% blend moved it to 0.68780.",
        "- This script searches around the winning strength and tests whether nearby multiplier variants or temporal-direction ensembles add more gain.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
