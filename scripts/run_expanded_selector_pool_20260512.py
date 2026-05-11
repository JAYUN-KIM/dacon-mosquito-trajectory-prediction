from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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
    SOURCE_SUBMISSION,
    aligned_predict_proba,
    build_oof_direct_local,
    label_weights,
    make_selector_features,
    recover_test_local_from_source,
)
from run_direct_step_refine_20260509 import direct_prediction  # noqa: E402
from run_hit_weighted_local_frame import make_features  # noqa: E402


PUBLIC_BEST_SUBMISSION = "direct_selector_rank2_selectorsoft.csv"
PUBLIC_BEST_SCORE = 0.68440
CURRENT_MULT = (1.02, 1.06, 0.94)
CV_SEEDS = [42, 777, 2026]
BLEND_WEIGHTS = [0.15, 0.25, 0.35]


@dataclass(frozen=True)
class Candidate:
    name: str
    mult: tuple[float, float, float]


def build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[float, float, float]] = set()

    def add(name: str, mult: tuple[float, float, float]) -> None:
        rounded = tuple(round(v, 4) for v in mult)
        if rounded in seen:
            return
        seen.add(rounded)
        candidates.append(Candidate(name, rounded))

    diagonal = [
        (1.02, 1.00, 1.00),
        (1.02, 1.02, 0.98),
        (1.02, 1.04, 0.96),
        (1.02, 1.06, 0.94),
        (1.02, 1.08, 0.92),
        (1.02, 1.10, 0.90),
        (1.02, 1.12, 0.88),
    ]
    for mult in diagonal:
        add(f"diag_f{mult[0]:.2f}_s{mult[1]:.2f}_u{mult[2]:.2f}", mult)

    for forward in [1.00, 1.01, 1.03, 1.04]:
        for side, up in [(1.04, 0.96), (1.06, 0.94), (1.08, 0.92), (1.10, 0.90)]:
            add(f"f{forward:.2f}_s{side:.2f}_u{up:.2f}", (forward, side, up))

    off_diagonal = [
        (1.02, 1.04, 0.94),
        (1.02, 1.06, 0.96),
        (1.02, 1.08, 0.94),
        (1.02, 1.10, 0.92),
        (1.02, 1.12, 0.90),
        (1.03, 1.06, 0.96),
        (1.03, 1.08, 0.94),
        (1.01, 1.06, 0.96),
        (1.01, 1.08, 0.94),
    ]
    for mult in off_diagonal:
        add(f"off_f{mult[0]:.2f}_s{mult[1]:.2f}_u{mult[2]:.2f}", mult)

    return candidates


CANDIDATES = build_candidates()


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def candidate_index(mult: tuple[float, float, float]) -> int:
    rounded = tuple(round(v, 4) for v in mult)
    for idx, candidate in enumerate(CANDIDATES):
        if candidate.mult == rounded:
            return idx
    raise ValueError(f"missing candidate multiplier: {mult}")


def make_candidate_predictions(coords: np.ndarray, pred_local_scaled: np.ndarray) -> np.ndarray:
    return np.stack([direct_prediction(coords, pred_local_scaled, candidate.mult) for candidate in CANDIDATES], axis=1)


def make_selector(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(CANDIDATES),
        n_estimators=520,
        learning_rate=0.022,
        num_leaves=63,
        min_child_samples=14,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.92,
        reg_alpha=0.05,
        reg_lambda=0.35,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def align_proba(model: LGBMClassifier, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    aligned = np.zeros((len(features), len(CANDIDATES)), dtype=np.float64)
    for col, class_id in enumerate(model.classes_):
        aligned[:, int(class_id)] = raw[:, col]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sum, out=np.full_like(aligned, 1.0 / len(CANDIDATES)), where=row_sum > 1e-12)


def soft_predict(proba: np.ndarray, preds: np.ndarray) -> np.ndarray:
    return np.einsum("nc,ncd->nd", proba, preds)


def pick_by_indices(preds: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return preds[np.arange(len(indices)), indices]


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def read_submission_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def evaluate_cv(
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
        proba = align_proba(model, selector_features[val_mask])
        hard_idx = np.argmax(proba, axis=1)
        current_pred = candidate_preds[val_mask, current_idx]
        soft_pred = soft_predict(proba, candidate_preds[val_mask])
        hard_pred = pick_by_indices(candidate_preds[val_mask], hard_idx)
        y_val = y[val_mask]
        rows.append({"seed": seed, "strategy": "current_mult", **distance_summary(current_pred, y_val)})
        rows.append({"seed": seed, "strategy": "expanded_soft", **distance_summary(soft_pred, y_val)})
        rows.append({"seed": seed, "strategy": "expanded_hard", **distance_summary(hard_pred, y_val)})
        for blend in BLEND_WEIGHTS:
            pred = (1.0 - blend) * current_pred + blend * soft_pred
            rows.append({"seed": seed, "strategy": f"current_expsoft_blend{blend:.2f}", **distance_summary(pred, y_val)})

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
    parser = argparse.ArgumentParser(description="Expanded candidate-pool selector soft probes.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_expanded_selector_pool_20260512.md")
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

    print(f"candidate_count={len(CANDIDATES)}", flush=True)
    print("Building OOF direct-step local predictions", flush=True)
    oof_local = build_oof_direct_local(train_coords, y, train_features)
    train_candidate_preds = make_candidate_predictions(train_coords, oof_local)
    distances = np.linalg.norm(train_candidate_preds - y[:, None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    best_dist = np.min(distances, axis=1)
    current_idx = candidate_index(CURRENT_MULT)
    current_dist = distances[:, current_idx]
    weights = label_weights(best_dist, current_dist)
    selector_features = make_selector_features(train_features, train_coords, oof_local)

    print("Evaluating expanded selector CV", flush=True)
    cv = evaluate_cv(selector_features, labels, weights, train_candidate_preds, y, current_idx)
    print(cv.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Preparing test predictions", flush=True)
    source_pred = read_submission_coords(args.submission_dir / SOURCE_SUBMISSION)
    test_local = recover_test_local_from_source(test_coords, source_pred)
    test_candidate_preds = make_candidate_predictions(test_coords, test_local)
    test_selector_features = make_selector_features(test_features, test_coords, test_local)
    model = make_selector(91212)
    model.fit(selector_features, labels, sample_weight=weights)
    test_proba = align_proba(model, test_selector_features)
    expanded_soft = soft_predict(test_proba, test_candidate_preds)
    expanded_hard = pick_by_indices(test_candidate_preds, np.argmax(test_proba, axis=1))
    public_best = read_submission_coords(args.submission_dir / PUBLIC_BEST_SUBMISSION)

    output_specs = [
        ("expanded_soft_blend015", (0.85 * public_best + 0.15 * expanded_soft), "0.85 * current public best + 0.15 * expanded candidate soft"),
        ("expanded_soft_blend025", (0.75 * public_best + 0.25 * expanded_soft), "0.75 * current public best + 0.25 * expanded candidate soft"),
        ("expanded_soft_blend035", (0.65 * public_best + 0.35 * expanded_soft), "0.65 * current public best + 0.35 * expanded candidate soft"),
        ("expanded_soft_full", expanded_soft, "full expanded candidate soft"),
        ("expanded_hard_blend015", (0.85 * public_best + 0.15 * expanded_hard), "0.85 * current public best + 0.15 * expanded hard route"),
    ]
    rows = []
    written = []
    for rank, (name, pred, note) in enumerate(output_specs, start=1):
        path = args.submission_dir / f"expanded_selector_rank{rank}_{slug(name)}.csv"
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

    candidate_rows = []
    for idx, candidate in enumerate(CANDIDATES):
        candidate_rows.append(
            {
                "idx": idx,
                "candidate": candidate.name,
                "forward": candidate.mult[0],
                "side": candidate.mult[1],
                "up": candidate.mult[2],
                "label_count": int(np.sum(labels == idx)),
                "oof_hit": float(np.mean(distances[:, idx] <= 0.01)),
                "oof_mean_distance": float(np.mean(distances[:, idx])),
            }
        )
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["label_count", "oof_hit"], ascending=[False, False])

    report = [
        "# 2026-05-12 Expanded Selector Pool",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- candidate_count: `{len(CANDIDATES)}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## CV",
        "",
        dataframe_to_markdown(cv),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(rows)),
        "",
        "## Candidate Label Distribution",
        "",
        dataframe_to_markdown(candidate_df),
        "",
        "## Notes",
        "",
        "- Temperature and top-k truncation hurt public, so this probe changes the multiplier candidate pool itself.",
        "- Output candidates blend the expanded-pool soft route back into the current public best to control public risk.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
