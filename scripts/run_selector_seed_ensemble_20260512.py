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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, split_mask, stack_samples  # noqa: E402
from run_direct_multiplier_selector_20260510 import (  # noqa: E402
    CA_A6_SPEC,
    CURRENT_BEST_NAME,
    SOURCE_SUBMISSION,
    aligned_predict_proba,
    build_oof_direct_local,
    candidate_index,
    label_weights,
    make_candidate_predictions,
    make_selector,
    make_selector_features,
    recover_test_local_from_source,
)
from run_hit_weighted_local_frame import make_features  # noqa: E402


PUBLIC_BEST_SUBMISSION = "direct_selector_rank2_selectorsoft.csv"
PUBLIC_BEST_SCORE = 0.68440
CV_OUTER_SEEDS = [42, 777, 2026]
ENSEMBLES = {
    "seedens3": [42, 777, 2026],
    "seedens5": [42, 777, 2026, 88010, 12345],
}
BLENDS = [0.20, 0.35]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_submission_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def soft_predict(proba: np.ndarray, candidate_preds: np.ndarray) -> np.ndarray:
    return np.einsum("nc,ncd->nd", proba, candidate_preds)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def predict_proba_ensemble(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    pred_x: np.ndarray,
    seeds: list[int],
) -> np.ndarray:
    probas = []
    for seed in seeds:
        model = make_selector(seed)
        model.fit(train_x, train_y, sample_weight=weights)
        probas.append(aligned_predict_proba(model, pred_x))
    return np.mean(probas, axis=0)


def evaluate_cv(
    selector_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    candidate_preds: np.ndarray,
    y: np.ndarray,
    current_idx: int,
) -> pd.DataFrame:
    rows = []
    for outer_seed in CV_OUTER_SEEDS:
        val_mask = split_mask(len(labels), 0.2, outer_seed)
        train_mask = ~val_mask
        y_val = y[val_mask]
        current_pred = candidate_preds[val_mask, current_idx]
        rows.append({"outer_seed": outer_seed, "strategy": "current_mult", **distance_summary(current_pred, y_val)})
        for name, seeds in ENSEMBLES.items():
            proba = predict_proba_ensemble(
                selector_features[train_mask],
                labels[train_mask],
                weights[train_mask],
                selector_features[val_mask],
                seeds,
            )
            pred = soft_predict(proba, candidate_preds[val_mask])
            rows.append({"outer_seed": outer_seed, "strategy": name, **distance_summary(pred, y_val)})

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("strategy", as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    summary["risk_adjusted_hit"] = summary["mean_r_hit"] - 0.25 * summary["std_r_hit"].fillna(0.0)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed ensemble probes for selector soft probabilities.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_selector_seed_ensemble_20260512.md")
    parser.add_argument("--run-cv", action="store_true", help="Run expensive outer-CV seed ensemble diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    print(f"Loading data from {data_dir}", flush=True)

    train_samples = read_trajectory_folder(data_dir / "train")
    test_samples = read_trajectory_folder(data_dir / "test")
    targets = read_targets(data_dir / "train_labels.csv")
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    ids, missing = aligned_ids(train_samples, targets[ID_COLUMN].tolist())
    if missing:
        raise ValueError(f"{len(missing)} train ids are missing trajectory files, examples: {missing[:5]}")

    train_coords = stack_samples(train_samples, ids)
    test_coords = stack_samples(test_samples, sample_submission[ID_COLUMN].tolist())
    y = targets.set_index(ID_COLUMN).loc[ids, COORD_COLUMNS].to_numpy(dtype=np.float64)
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)

    print("Building OOF direct-step local predictions", flush=True)
    oof_local = build_oof_direct_local(train_coords, y, train_features)
    train_candidate_preds = make_candidate_predictions(train_coords, oof_local)
    distances = np.linalg.norm(train_candidate_preds - y[:, None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    best_dist = np.min(distances, axis=1)
    current_idx = candidate_index(CURRENT_BEST_NAME)
    current_dist = distances[:, current_idx]
    weights = label_weights(best_dist, current_dist)
    selector_features = make_selector_features(train_features, train_coords, oof_local)

    cv = None
    if args.run_cv:
        print("Evaluating selector seed ensembles", flush=True)
        cv = evaluate_cv(selector_features, labels, weights, train_candidate_preds, y, current_idx)
        print(cv.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    source_pred = read_submission_coords(args.submission_dir / SOURCE_SUBMISSION)
    test_local = recover_test_local_from_source(test_coords, source_pred)
    test_candidate_preds = make_candidate_predictions(test_coords, test_local)
    test_selector_features = make_selector_features(test_features, test_coords, test_local)
    public_best = read_submission_coords(args.submission_dir / PUBLIC_BEST_SUBMISSION)

    rows = []
    written = []
    rank = 1
    for name, seeds in ENSEMBLES.items():
        print(f"Fitting full selector ensemble {name}: {seeds}", flush=True)
        proba = predict_proba_ensemble(selector_features, labels, weights, test_selector_features, seeds)
        pred = soft_predict(proba, test_candidate_preds)
        path = args.submission_dir / f"seedens_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "strategy": name,
                "seeds": str(seeds),
                "blend_with_seedens": 1.0,
                **delta_summary(pred, public_best, "vs_public_best"),
            }
        )
        rank += 1
        for blend in BLENDS:
            blended = (1.0 - blend) * public_best + blend * pred
            path = args.submission_dir / f"seedens_rank{rank}_{slug(name)}blend{int(blend * 100):02d}.csv"
            write_submission(sample_submission, blended, path)
            written.append(path)
            rows.append(
                {
                    "rank": rank,
                    "submission": path.name,
                    "strategy": name,
                    "seeds": str(seeds),
                    "blend_with_seedens": blend,
                    **delta_summary(blended, public_best, "vs_public_best"),
                }
            )
            rank += 1

    report = [
        "# 2026-05-12 Selector Seed Ensemble",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(rows)),
        "",
        "## Notes",
        "",
        "- Temperature/top-k hurt public, so this probe keeps the original candidate pool and stabilizes selector probabilities with seed ensembling.",
        "- Blend outputs are safer if the full seed ensemble moves too far from the current public best.",
    ]
    if cv is not None:
        report[8:8] = ["## CV", "", dataframe_to_markdown(cv), ""]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
