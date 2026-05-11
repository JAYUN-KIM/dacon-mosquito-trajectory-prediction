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
CONF45_SUBMISSION = "direct_selector_rank4_selectorconf045.csv"
CV_SEEDS = [42, 777, 2026]
TEMP_VALUES = [0.75, 0.85, 0.95, 1.10, 1.25, 1.50]
TOPK_CONFIGS = [(2, 1.00), (3, 1.00), (2, 0.85), (3, 0.85), (2, 1.15), (3, 1.15)]
ANCHOR_BLEND_CONFIGS = [
    ("soft_conf45blend010", 0.90, CONF45_SUBMISSION),
    ("soft_conf45blend020", 0.80, CONF45_SUBMISSION),
]


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


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def temperature_proba(proba: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(proba, 1e-12, 1.0)
    adjusted = clipped ** (1.0 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def topk_proba(proba: np.ndarray, top_k: int, temperature: float = 1.0) -> np.ndarray:
    adjusted = temperature_proba(proba, temperature)
    if top_k >= adjusted.shape[1]:
        return adjusted
    keep = np.argpartition(adjusted, -top_k, axis=1)[:, -top_k:]
    mask = np.zeros_like(adjusted, dtype=bool)
    row = np.arange(len(adjusted))[:, None]
    mask[row, keep] = True
    truncated = np.where(mask, adjusted, 0.0)
    return truncated / truncated.sum(axis=1, keepdims=True)


def soft_predict(proba: np.ndarray, candidate_preds: np.ndarray) -> np.ndarray:
    return np.einsum("nc,ncd->nd", proba, candidate_preds)


def evaluate_soft_cv(
    selector_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    candidate_preds: np.ndarray,
    y: np.ndarray,
    current_idx: int,
) -> pd.DataFrame:
    rows = []
    for seed in CV_SEEDS:
        val_mask = split_mask(len(labels), 0.2, seed)
        train_mask = ~val_mask
        model = make_selector(seed)
        model.fit(selector_features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        proba = aligned_predict_proba(model, selector_features[val_mask])
        y_val = y[val_mask]
        current_pred = candidate_preds[val_mask, current_idx]
        rows.append({"seed": seed, "strategy": "current_mult", **distance_summary(current_pred, y_val)})
        rows.append({"seed": seed, "strategy": "soft_t1.00", **distance_summary(soft_predict(proba, candidate_preds[val_mask]), y_val)})

        for temperature in TEMP_VALUES:
            name = f"soft_t{temperature:.2f}"
            pred = soft_predict(temperature_proba(proba, temperature), candidate_preds[val_mask])
            rows.append({"seed": seed, "strategy": name, **distance_summary(pred, y_val)})

        for top_k, temperature in TOPK_CONFIGS:
            name = f"top{top_k}_t{temperature:.2f}"
            pred = soft_predict(topk_proba(proba, top_k, temperature), candidate_preds[val_mask])
            rows.append({"seed": seed, "strategy": name, **distance_summary(pred, y_val)})

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


def build_context(data_dir: Path, submission_dir: Path) -> dict[str, object]:
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

    source_pred = read_submission_coords(submission_dir / SOURCE_SUBMISSION)
    test_local = recover_test_local_from_source(test_coords, source_pred)
    test_candidate_preds = make_candidate_predictions(test_coords, test_local)
    test_selector_features = make_selector_features(test_features, test_coords, test_local)

    print("Training full selector for test probabilities", flush=True)
    model = make_selector(88010)
    model.fit(selector_features, labels, sample_weight=weights)
    test_proba = aligned_predict_proba(model, test_selector_features)

    return {
        "sample_submission": sample_submission,
        "selector_features": selector_features,
        "labels": labels,
        "weights": weights,
        "train_candidate_preds": train_candidate_preds,
        "y": y,
        "current_idx": current_idx,
        "test_candidate_preds": test_candidate_preds,
        "test_proba": test_proba,
    }


def build_test_outputs(context: dict[str, object], submission_dir: Path, top_k: int) -> tuple[pd.DataFrame, list[Path]]:
    sample_submission = context["sample_submission"]
    test_candidate_preds = context["test_candidate_preds"]
    test_proba = context["test_proba"]
    public_best = read_submission_coords(submission_dir / PUBLIC_BEST_SUBMISSION)

    candidates: list[tuple[str, np.ndarray, str]] = []
    for temperature in TEMP_VALUES:
        pred = soft_predict(temperature_proba(test_proba, temperature), test_candidate_preds)
        candidates.append((f"soft_t{temperature:.2f}", pred, f"selector probability temperature {temperature:.2f}"))
    for top_k_value, temperature in TOPK_CONFIGS:
        pred = soft_predict(topk_proba(test_proba, top_k_value, temperature), test_candidate_preds)
        candidates.append((f"top{top_k_value}_t{temperature:.2f}", pred, f"top-{top_k_value} truncated soft selector, T={temperature:.2f}"))

    reference_by_name = {PUBLIC_BEST_SUBMISSION: public_best}
    reference_by_name[CONF45_SUBMISSION] = read_submission_coords(submission_dir / CONF45_SUBMISSION)
    soft_anchor = read_submission_coords(submission_dir / PUBLIC_BEST_SUBMISSION)
    for name, anchor_weight, reference_name in ANCHOR_BLEND_CONFIGS:
        ref = reference_by_name[reference_name]
        pred = anchor_weight * soft_anchor + (1.0 - anchor_weight) * ref
        candidates.append((name, pred, f"{anchor_weight:.2f} * selector_soft + {1.0 - anchor_weight:.2f} * {reference_name}"))

    rows = []
    written = []
    for rank, (name, pred, note) in enumerate(candidates[:top_k], start=1):
        path = submission_dir / f"softtemp_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "candidate": name,
                "note": note,
                **delta_summary(pred, public_best, "vs_public_best"),
            }
        )
    return pd.DataFrame(rows), written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selector soft probability temperature and top-k truncation probes.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_selector_soft_temperature_20260512.md")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    context = build_context(data_dir, args.submission_dir)

    print("Evaluating CV strategies", flush=True)
    cv = evaluate_soft_cv(
        context["selector_features"],
        context["labels"],
        context["weights"],
        context["train_candidate_preds"],
        context["y"],
        context["current_idx"],
    )
    print(cv.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    outputs, written = build_test_outputs(context, args.submission_dir, args.top_k)
    report = [
        "# 2026-05-12 Selector Soft Temperature",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## CV",
        "",
        dataframe_to_markdown(cv),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(outputs),
        "",
        "## Notes",
        "",
        "- The 2026-05-11 public best came from probability-weighted selector soft routing.",
        "- This probe changes only the probability shape via temperature or top-k truncation, keeping the same direct-step candidate pool.",
        "- If CV ranks lower-temperature soft variants near soft_t1.00, submit the smallest movement candidates first.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
