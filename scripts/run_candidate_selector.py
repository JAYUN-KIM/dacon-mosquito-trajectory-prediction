from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in [SRC, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mosquito_trajectory.data import (
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import (
    SEEDS,
    dataframe_to_markdown,
    distance_summary,
    make_features,
    physics_prediction,
    split_mask,
    stack_samples,
)


warnings.filterwarnings("ignore", message="X does not have valid feature names")


CANDIDATES = [
    ("v100_a000", 1.00, 0.000),
    ("v100_a100", 1.00, 0.100),
    ("v100_a150", 1.00, 0.150),
    ("v100_a200", 1.00, 0.200),
    ("v100_a250", 1.00, 0.250),
    ("v100_a275", 1.00, 0.275),
    ("v100_a300", 1.00, 0.300),
    ("v098_a175", 0.98, 0.175),
    ("v098_a200", 0.98, 0.200),
    ("v098_a225", 0.98, 0.225),
    ("v102_a200", 1.02, 0.200),
    ("v102_a250", 1.02, 0.250),
]


def candidate_predictions(coords: np.ndarray) -> np.ndarray:
    return np.stack(
        [physics_prediction(coords, velocity_scale=velocity_scale, accel_scale=accel_scale) for _, velocity_scale, accel_scale in CANDIDATES],
        axis=1,
    )


def classifier(seed: int, objective: str = "binary") -> LGBMClassifier:
    params = {
        "objective": objective,
        "n_estimators": 320,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 28,
        "subsample": 0.88,
        "subsample_freq": 1,
        "colsample_bytree": 0.88,
        "reg_alpha": 0.04,
        "reg_lambda": 0.35,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if objective == "multiclass":
        params["num_class"] = len(CANDIDATES)
    return LGBMClassifier(**params)


def evaluate_single_candidates(cand_pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, (name, velocity_scale, accel_scale) in enumerate(CANDIDATES):
        rows.append(
            {
                "candidate": name,
                "velocity_scale": velocity_scale,
                "accel_scale": accel_scale,
                **distance_summary(cand_pred[:, idx, :], y),
            }
        )
    return pd.DataFrame(rows).sort_values(["r_hit_1cm", "mean_distance"], ascending=[False, True])


def select_predictions(cand_pred: np.ndarray, selected_idx: np.ndarray) -> np.ndarray:
    return cand_pred[np.arange(len(selected_idx)), selected_idx, :]


def evaluate_selectors(features: np.ndarray, cand_pred: np.ndarray, y: np.ndarray, val_frac: float) -> pd.DataFrame:
    dist = np.linalg.norm(cand_pred - y[:, None, :], axis=2)
    hit = dist <= 0.01
    best_idx = np.argmin(dist, axis=1)
    rows = []

    for seed in SEEDS:
        val_mask = split_mask(len(features), val_frac, seed)
        train_mask = ~val_mask
        y_val = y[val_mask]
        cand_val = cand_pred[val_mask]

        oracle_idx = np.argmin(dist[val_mask], axis=1)
        rows.append(
            {
                "selector": "oracle_distance",
                "seed": seed,
                "detail": "upper_bound",
                **distance_summary(select_predictions(cand_val, oracle_idx), y_val),
            }
        )
        oracle_hit_idx = np.argmax(hit[val_mask].astype(int), axis=1)
        no_hit = ~hit[val_mask].any(axis=1)
        oracle_hit_idx[no_hit] = oracle_idx[no_hit]
        rows.append(
            {
                "selector": "oracle_hit",
                "seed": seed,
                "detail": "upper_bound",
                **distance_summary(select_predictions(cand_val, oracle_hit_idx), y_val),
            }
        )

        multi = classifier(seed, objective="multiclass")
        multi.fit(features[train_mask], best_idx[train_mask])
        multi_idx = multi.predict(features[val_mask]).astype(int)
        rows.append(
            {
                "selector": "multiclass_best_distance",
                "seed": seed,
                "detail": "argmin_distance_label",
                **distance_summary(select_predictions(cand_val, multi_idx), y_val),
            }
        )

        prob_cols = []
        for cand_idx in range(len(CANDIDATES)):
            clf = classifier(seed + cand_idx, objective="binary")
            clf.fit(features[train_mask], hit[train_mask, cand_idx].astype(int))
            prob_cols.append(clf.predict_proba(features[val_mask])[:, 1])
        prob = np.vstack(prob_cols).T
        binary_idx = np.argmax(prob, axis=1)
        rows.append(
            {
                "selector": "binary_hit_probability",
                "seed": seed,
                "detail": "max_predicted_hit_probability",
                **distance_summary(select_predictions(cand_val, binary_idx), y_val),
            }
        )

    df = pd.DataFrame(rows)
    return (
        df.groupby(["selector", "detail"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )


def full_binary_selector_submission(features: np.ndarray, hit: np.ndarray, test_features: np.ndarray, test_cand: np.ndarray) -> np.ndarray:
    seed_probs = []
    for seed in SEEDS[:3]:
        prob_cols = []
        for cand_idx in range(len(CANDIDATES)):
            clf = classifier(seed + cand_idx, objective="binary")
            clf.fit(features, hit[:, cand_idx].astype(int))
            prob_cols.append(clf.predict_proba(test_features)[:, 1])
        seed_probs.append(np.vstack(prob_cols).T)
    prob = np.mean(seed_probs, axis=0)
    selected_idx = np.argmax(prob, axis=1)
    return select_predictions(test_cand, selected_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sample-wise selectors over physics candidates.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_candidate_selector.md")
    parser.add_argument("--val-frac", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}")

    train_samples = read_trajectory_folder(data_dir / "train")
    test_samples = read_trajectory_folder(data_dir / "test")
    targets = read_targets(data_dir / "train_labels.csv")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")

    ids, missing = aligned_ids(train_samples, targets[ID_COLUMN].tolist())
    if missing:
        raise ValueError(f"{len(missing)} train ids are missing trajectory files, examples: {missing[:5]}")

    train_coords = stack_samples(train_samples, ids)
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)
    test_ids = sample_submission[ID_COLUMN].tolist()
    test_coords = stack_samples(test_samples, test_ids)

    features = make_features(train_coords)
    test_features = make_features(test_coords)
    cand_pred = candidate_predictions(train_coords)
    test_cand = candidate_predictions(test_coords)

    single = evaluate_single_candidates(cand_pred, y)
    print("Single-candidate full-train diagnostics:")
    print(single.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    selectors = evaluate_selectors(features, cand_pred, y, args.val_frac)
    print("\nSelector CV:")
    print(selectors.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    dist = np.linalg.norm(cand_pred - y[:, None, :], axis=2)
    hit = dist <= 0.01
    selector_pred = full_binary_selector_submission(features, hit, test_features, test_cand)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selector_path = args.output_dir / "candidate_binary_hit_selector.csv"
    submission = sample_submission[[ID_COLUMN]].copy()
    submission[COORD_COLUMNS] = selector_pred
    submission.to_csv(selector_path, index=False)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Candidate Selector",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Data dir: `{data_dir}`",
        f"- Candidates: `{[name for name, _, _ in CANDIDATES]}`",
        f"- CV seeds: `{SEEDS}`",
        f"- Submission: `{selector_path}`",
        "",
        "## Single Candidates On Full Train",
        "",
        dataframe_to_markdown(single),
        "",
        "## Selector CV",
        "",
        dataframe_to_markdown(selectors),
        "",
        "## Readout",
        "",
        "- `oracle_hit` is the upper bound if we could always choose a hitting candidate from the set.",
        "- If `binary_hit_probability` beats the best single candidate in CV, this is a good aggressive public submission candidate.",
        "- If selector CV underperforms, the candidate set has signal but the selector features are not predictive enough yet.",
    ]
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nWrote selector submission: {selector_path}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()

