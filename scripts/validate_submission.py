import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["id", "x", "y", "z"]


def validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("path:", path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("head:")
    print(df.head())
    print("nulls:")
    print(df.isnull().sum())
    print("describe:")
    print(df.describe(include="all"))

    errors = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if df.isnull().sum().sum() > 0:
        errors.append("submission has null values")
    if not missing:
        if not np.isfinite(df[["x", "y", "z"]].to_numpy(dtype=float)).all():
            errors.append("coordinates contain non-finite values")
        if df["id"].duplicated().any():
            errors.append("duplicated id values")

    if errors:
        print("[FAIL]", " | ".join(errors))
        raise SystemExit(1)

    print("[OK] submission validation passed")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    validate(args.path)


if __name__ == "__main__":
    main()

