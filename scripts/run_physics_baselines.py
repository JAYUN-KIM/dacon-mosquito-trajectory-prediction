from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mosquito_trajectory.baselines import BASELINE_METHODS, TARGET_DELTA_MS, predict_many
from mosquito_trajectory.data import (
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    explain_expected_layout,
    missing_raw_paths,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)


def r_hit(pred: np.ndarray, true: np.ndarray, radius: float = 0.01) -> float:
    distances = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=1)
    return float(np.mean(distances <= radius))


def distance_summary(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=1)
    return {
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "p90_distance": float(np.quantile(distances, 0.90)),
        "p95_distance": float(np.quantile(distances, 0.95)),
        "r_hit_1cm": float(np.mean(distances <= 0.01)),
    }


def split_ids(ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < val_frac < 1.0:
        return ids, []
    rng = np.random.default_rng(seed)
    shuffled = np.array(ids, dtype=object)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_frac)))
    val_ids = sorted(shuffled[:val_size].tolist())
    train_ids = sorted(shuffled[val_size:].tolist())
    return train_ids, val_ids


def summarize_sequences(samples: dict) -> dict[str, object]:
    lengths = np.array([sample.coords.shape[0] for sample in samples.values()])
    first_times = np.array([sample.timesteps_ms[0] for sample in samples.values()])
    last_times = np.array([sample.timesteps_ms[-1] for sample in samples.values()])
    coords = np.vstack([sample.coords for sample in samples.values()])
    step_distances = np.concatenate(
        [
            np.linalg.norm(np.diff(sample.coords, axis=0), axis=1)
            for sample in samples.values()
            if sample.coords.shape[0] >= 2
        ]
    )

    return {
        "sample_count": int(len(samples)),
        "sequence_length_min": int(lengths.min()),
        "sequence_length_max": int(lengths.max()),
        "first_timestep_ms_values": sorted(np.unique(first_times).astype(float).tolist()),
        "last_timestep_ms_values": sorted(np.unique(last_times).astype(float).tolist()),
        "coord_min": dict(zip(COORD_COLUMNS, coords.min(axis=0).round(6).tolist())),
        "coord_max": dict(zip(COORD_COLUMNS, coords.max(axis=0).round(6).tolist())),
        "step_distance_mean": float(np.mean(step_distances)),
        "step_distance_p50": float(np.quantile(step_distances, 0.50)),
        "step_distance_p90": float(np.quantile(step_distances, 0.90)),
        "step_distance_p99": float(np.quantile(step_distances, 0.99)),
    }


def evaluate_methods(
    samples: dict,
    targets: pd.DataFrame,
    ids: list[str],
    methods: list[str],
    delta_ms: float,
) -> pd.DataFrame:
    true = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=float)
    rows = []
    for method in methods:
        pred = predict_many(samples, ids, method, delta_ms)
        rows.append({"method": method, **distance_summary(pred, true)})
    return pd.DataFrame(rows).sort_values(
        ["r_hit_1cm", "mean_distance", "median_distance"],
        ascending=[False, True, True],
    )


def write_submission(
    sample_submission: pd.DataFrame,
    test_samples: dict,
    method: str,
    output_dir: Path,
    delta_ms: float,
) -> Path:
    ids = sample_submission[ID_COLUMN].astype(str).tolist()
    present, missing = aligned_ids(test_samples, ids)
    if missing:
        raise ValueError(f"{len(missing)} test ids are missing trajectory files, examples: {missing[:5]}")
    pred = predict_many(test_samples, present, method, delta_ms)

    submission = sample_submission[[ID_COLUMN]].copy()
    submission[COORD_COLUMNS] = pred

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"physics_baseline_{method}.csv"
    submission.to_csv(path, index=False)
    return path


def append_experiment_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            logs = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []
    if not isinstance(logs, list):
        logs = []
    logs.append(payload)
    path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    report_path: Path,
    payload: dict[str, object],
    train_summary: dict[str, object],
    test_summary: dict[str, object],
    leaderboard: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = leaderboard.iloc[0]
    lines = [
        "# Physics Baseline Run",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Data dir: `{payload['data_dir']}`",
        f"- Validation size: `{payload['validation_size']}`",
        f"- Best method: `{payload['best_method']}`",
        f"- Best R-Hit@1cm: `{best['r_hit_1cm']:.6f}`",
        f"- Best mean distance: `{best['mean_distance']:.6f}` m",
        f"- Submission: `{payload['submission_path']}`",
        "",
        "## Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Data Summary",
        "",
        "### Train",
        "",
        "```json",
        json.dumps(train_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "### Test",
        "",
        "```json",
        json.dumps(test_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Readout",
        "",
    ]

    if str(best["method"]) == "position":
        lines.append("- The holdout currently favors no-motion prediction, so the next useful check is noise/outlier handling before adding more dynamics.")
    elif "velocity" in str(best["method"]):
        lines.append("- The holdout favors linear short-horizon motion. Next experiments should focus on robust velocity estimates and recent-window blending.")
    elif "poly2" in str(best["method"]) or "acceleration" in str(best["method"]):
        lines.append("- The holdout rewards curvature/acceleration. Next experiments should control overfit with smoothing and threshold-aware blending.")
    else:
        lines.append("- The best method is deterministic; use this as the baseline before residual ML.")

    lines.append("- Because the official metric is thresholded at 1cm, compare methods by `r_hit_1cm` first and use mean distance only as a tie-breaker.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate physics baselines and create a DACON submission.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--experiment-log", type=Path, default=ROOT / "experiments" / "log.json")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_physics_baseline.md")
    parser.add_argument("--delta-ms", type=float, default=TARGET_DELTA_MS)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission-method", default="auto", choices=["auto", *BASELINE_METHODS.keys()])
    parser.add_argument("--methods", nargs="*", default=list(BASELINE_METHODS.keys()), choices=list(BASELINE_METHODS.keys()))
    parser.add_argument("--limit-train", type=int, default=None, help="Optional smoke-test limit for train CSV files.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional smoke-test limit for test CSV files.")
    parser.add_argument("--skip-submission", action="store_true", help="Only run EDA/evaluation; useful with file limits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_raw_data_dir(args.data_dir)
    missing = missing_raw_paths(args.data_dir)
    if missing:
        print("[DATA MISSING]")
        print("Missing paths:")
        for path in missing:
            print(f"- {path}")
        print()
        print(explain_expected_layout(args.data_dir))
        raise SystemExit(2)

    print(f"Loading raw data from {args.data_dir}")
    train_samples = read_trajectory_folder(args.data_dir / "train", limit=args.limit_train)
    test_samples = read_trajectory_folder(args.data_dir / "test", limit=args.limit_test)
    targets = read_targets(args.data_dir / "train_labels.csv")
    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")

    target_ids = targets[ID_COLUMN].astype(str).tolist()
    train_ids, missing_train = aligned_ids(train_samples, target_ids)
    if missing_train:
        if args.limit_train is None:
            raise ValueError(f"{len(missing_train)} train labels are missing trajectory files, examples: {missing_train[:5]}")
        print(f"\n[WARN] {len(missing_train)} train labels are outside --limit-train and will be ignored")
        targets = targets[targets[ID_COLUMN].isin(train_ids)].copy()

    _, val_ids = split_ids(train_ids, args.val_frac, args.seed)
    if not val_ids:
        raise ValueError("--val-frac must leave at least one validation sample")

    train_summary = summarize_sequences(train_samples)
    test_summary = summarize_sequences(test_samples)
    print("\nTrain summary:")
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))
    print("\nTest summary:")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

    print(f"\nEvaluating {len(args.methods)} baseline methods on {len(val_ids)} validation samples")
    leaderboard = evaluate_methods(train_samples, targets, val_ids, args.methods, args.delta_ms)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    best_method = str(leaderboard.iloc[0]["method"])
    submission_method = best_method if args.submission_method == "auto" else args.submission_method
    submission_path = None
    if args.skip_submission:
        print("\nSkipped submission creation because --skip-submission was set")
    else:
        submission_path = write_submission(sample_submission, test_samples, submission_method, args.output_dir, args.delta_ms)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "delta_ms": args.delta_ms,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "validation_size": len(val_ids),
        "best_method": best_method,
        "submission_method": submission_method,
        "submission_path": str(submission_path) if submission_path is not None else None,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    append_experiment_log(args.experiment_log, payload)
    write_report(args.report_path, payload, train_summary, test_summary, leaderboard)

    if submission_path is not None:
        print(f"\nWrote submission: {submission_path}")
    print(f"Appended experiment log: {args.experiment_log}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()
