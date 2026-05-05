from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COORD_COLUMNS = ["x", "y", "z"]
ID_COLUMN = "id"


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in [ID_COLUMN, *COORD_COLUMNS] if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"{path} has duplicated ids")
    return df[[ID_COLUMN, *COORD_COLUMNS]].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend two coordinate submission files.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-weight", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.left_weight <= 1.0:
        raise ValueError("--left-weight must be between 0 and 1")

    left = read_submission(args.left)
    right = read_submission(args.right)
    if left[ID_COLUMN].tolist() != right[ID_COLUMN].tolist():
        right = right.set_index(ID_COLUMN).loc[left[ID_COLUMN]].reset_index()

    w = args.left_weight
    pred = w * left[COORD_COLUMNS].to_numpy(dtype=float) + (1.0 - w) * right[COORD_COLUMNS].to_numpy(dtype=float)
    out = left[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    diff = np.linalg.norm(left[COORD_COLUMNS].to_numpy(dtype=float) - right[COORD_COLUMNS].to_numpy(dtype=float), axis=1)
    print(f"Wrote blended submission: {args.output}")
    print(f"left_weight={w:.3f}")
    print(f"mean_pair_distance={diff.mean():.6f}")
    print(f"median_pair_distance={np.median(diff):.6f}")
    print(f"p95_pair_distance={np.quantile(diff, 0.95):.6f}")


if __name__ == "__main__":
    main()

