from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COORD_COLUMNS = ["x", "y", "z"]
ID_COLUMN = "id"
TIME_COLUMN = "timestep_ms"


@dataclass(frozen=True)
class TrajectorySample:
    sample_id: str
    timesteps_ms: np.ndarray
    coords: np.ndarray


def required_raw_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "train_dir": data_dir / "train",
        "test_dir": data_dir / "test",
        "train_labels": data_dir / "train_labels.csv",
        "sample_submission": data_dir / "sample_submission.csv",
    }


def is_raw_data_dir(data_dir: Path) -> bool:
    return all(path.exists() for path in required_raw_paths(data_dir).values())


def resolve_raw_data_dir(data_dir: Path) -> Path:
    """Find the actual DACON raw root even when open.zip was extracted into a nested folder."""
    data_dir = data_dir.resolve()
    if is_raw_data_dir(data_dir):
        return data_dir

    candidates = [path for path in data_dir.iterdir() if path.is_dir()]
    valid = [path for path in candidates if is_raw_data_dir(path)]
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        names = ", ".join(str(path) for path in valid)
        raise ValueError(f"multiple valid raw data directories found under {data_dir}: {names}")
    return data_dir


def missing_raw_paths(data_dir: Path) -> list[Path]:
    missing = []
    for path in required_raw_paths(data_dir).values():
        if not path.exists():
            missing.append(path)
    return missing


def explain_expected_layout(data_dir: Path) -> str:
    return f"""Expected raw data layout:

{data_dir}
  train/
    TRAIN_00001.csv
    TRAIN_00002.csv
    ...
  test/
    TEST_00001.csv
    TEST_00002.csv
    ...
  train_labels.csv
  sample_submission.csv

Each train/test CSV must contain: {TIME_COLUMN}, x, y, z
Label/submission CSVs must contain: id, x, y, z
"""


def canonicalize_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    rename: dict[str, str] = {}
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}

    for column in required:
        key = column.lower()
        if key not in lower_to_original:
            raise ValueError(f"missing required column '{column}' from columns {list(df.columns)}")
        original = lower_to_original[key]
        if original != column:
            rename[original] = column

    if rename:
        df = df.rename(columns=rename)
    return df


def read_trajectory_csv(path: Path) -> TrajectorySample:
    df = pd.read_csv(path)
    df = canonicalize_columns(df, [TIME_COLUMN, *COORD_COLUMNS])
    df = df.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    timesteps_ms = df[TIME_COLUMN].to_numpy(dtype=float)
    coords = df[COORD_COLUMNS].to_numpy(dtype=float)

    if len(timesteps_ms) < 2:
        raise ValueError(f"{path} has fewer than 2 timesteps")
    if np.unique(timesteps_ms).size != timesteps_ms.size:
        raise ValueError(f"{path} has duplicated timestep values")
    if not np.isfinite(coords).all():
        raise ValueError(f"{path} contains non-finite coordinates")

    return TrajectorySample(path.stem, timesteps_ms, coords)


def read_trajectory_folder(folder: Path, limit: int | None = None) -> dict[str, TrajectorySample]:
    paths = sorted(folder.glob("*.csv"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no CSV files found in {folder}")

    samples: dict[str, TrajectorySample] = {}
    for path in paths:
        sample = read_trajectory_csv(path)
        samples[sample.sample_id] = sample
    return samples


def read_targets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = canonicalize_columns(df, [ID_COLUMN, *COORD_COLUMNS])
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    if df[ID_COLUMN].duplicated().any():
        duplicated = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].head().tolist()
        raise ValueError(f"{path} has duplicated ids, examples: {duplicated}")
    if not np.isfinite(df[COORD_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite target coordinates")
    return df[[ID_COLUMN, *COORD_COLUMNS]].copy()


def read_sample_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = canonicalize_columns(df, [ID_COLUMN, *COORD_COLUMNS])
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    if df[ID_COLUMN].duplicated().any():
        duplicated = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].head().tolist()
        raise ValueError(f"{path} has duplicated ids, examples: {duplicated}")
    return df[[ID_COLUMN, *COORD_COLUMNS]].copy()


def aligned_ids(samples: dict[str, TrajectorySample], ids: Iterable[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for sample_id in ids:
        if sample_id in samples:
            present.append(sample_id)
        else:
            missing.append(sample_id)
    return present, missing
