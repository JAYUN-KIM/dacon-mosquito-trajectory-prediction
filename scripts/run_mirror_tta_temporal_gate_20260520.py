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

from mosquito_trajectory.data import (  # noqa: E402
    COORD_COLUMNS,
    ID_COLUMN,
    aligned_ids,
    read_sample_submission,
    read_targets,
    read_trajectory_folder,
    resolve_raw_data_dir,
)
from run_aggressive_experiments import dataframe_to_markdown, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import (  # noqa: E402
    boundary_weights,
    curvature_correction,
    make_gate_classifier,
    make_gate_features,
    optimal_alpha,
    predict_gate_proba,
    read_submission_coords,
    write_submission,
)
from run_hit_weighted_local_frame import make_features  # noqa: E402
from run_temporal_backcast_augmentation_20260516 import (  # noqa: E402
    SPECS,
    direct_prediction,
    fit_predict_augmented,
)


SELECTOR_SOFT_ANCHOR = "direct_selector_rank2_selectorsoft.csv"
CURRENT_BEST_SUBMISSION = "curvgate_refine_rank2_gatet52a105.csv"
TEMPORAL_SPEC_NAME = "tb_c678_w020"
TEMPORAL_MULT = (1.02, 1.00, 1.00)
FULL_SEEDS = [42, 777, 2026]
GATE_THRESHOLD = 0.52
GATE_ALPHA = 0.105
TEMPORAL_WEIGHTS = [0.50, 0.55, 0.60]
BEST_BLEND_WEIGHTS = [0.35, 0.50]


def mirror_coords_y(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    out[..., 1] *= -1.0
    return out


def mirror_points_y(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, 1] *= -1.0
    return out


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def train_predict_mirror_tta_temporal(
    train_coords: np.ndarray,
    y: np.ndarray,
    test_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    specs = {spec.name: spec for spec in SPECS}
    if TEMPORAL_SPEC_NAME not in specs:
        raise ValueError(f"unknown temporal spec: {TEMPORAL_SPEC_NAME}")

    train_aug_coords = np.concatenate([train_coords, mirror_coords_y(train_coords)], axis=0)
    y_aug = np.concatenate([y, mirror_points_y(y)], axis=0)
    test_mirror_coords = mirror_coords_y(test_coords)

    print("Building mirror-augmented feature matrices", flush=True)
    train_aug_features, _ = make_features(train_aug_coords)
    test_features, _ = make_features(test_coords)
    test_mirror_features, _ = make_features(test_mirror_coords)
    pred_features = np.vstack([test_features, test_mirror_features]).astype(np.float32)

    raw_preds = []
    tta_preds = []
    seed_rows = []
    for seed in FULL_SEEDS:
        print(f"mirror temporal full seed={seed}", flush=True)
        pred_local = fit_predict_augmented(
            train_aug_coords,
            y_aug,
            train_aug_features,
            pred_features,
            specs[TEMPORAL_SPEC_NAME],
            520000 + seed,
        )
        n_test = len(test_coords)
        pred_raw = direct_prediction(test_coords, pred_local[:n_test], TEMPORAL_MULT)
        pred_mirror = direct_prediction(test_mirror_coords, pred_local[n_test:], TEMPORAL_MULT)
        pred_mirror_unflipped = mirror_points_y(pred_mirror)
        pred_tta = 0.5 * (pred_raw + pred_mirror_unflipped)
        raw_preds.append(pred_raw)
        tta_preds.append(pred_tta)
        seed_rows.append(
            {
                "seed": seed,
                "raw_vs_tta_mean_delta": float(np.mean(np.linalg.norm(pred_raw - pred_tta, axis=1))),
                "raw_vs_tta_p95_delta": float(np.quantile(np.linalg.norm(pred_raw - pred_tta, axis=1), 0.95)),
            }
        )
    return np.mean(raw_preds, axis=0), np.mean(tta_preds, axis=0), pd.DataFrame(seed_rows)


def train_gate_models(
    train_gate_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> list:
    models = []
    for seed in [1301, 2718, 3141]:
        print(f"full gate model for mirror-TTA seed={seed}", flush=True)
        model = make_gate_classifier(620000 + seed)
        model.fit(train_gate_features, labels, sample_weight=weights)
        models.append(model)
    return models


def predict_gate(models: list, test_gate_features: np.ndarray) -> np.ndarray:
    return np.mean([predict_gate_proba(model, test_gate_features) for model in models], axis=0)


def apply_gate(anchor: np.ndarray, correction: np.ndarray, gate_proba: np.ndarray) -> tuple[np.ndarray, float]:
    mask = gate_proba >= GATE_THRESHOLD
    pred = anchor.copy()
    pred[mask] = anchor[mask] + GATE_ALPHA * correction[mask]
    return pred, float(np.mean(mask))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror-symmetry TTA temporal-backcast with curvature gate.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_mirror_tta_temporal_gate_20260520.md")
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    if not args.cache_path.exists():
        raise FileNotFoundError(args.cache_path)

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

    print("Training mirror-symmetry temporal TTA model", flush=True)
    mirror_raw_temporal, mirror_tta_temporal, seed_diag = train_predict_mirror_tta_temporal(train_coords, y, test_coords)

    selector_soft = read_submission_coords(args.submission_dir / SELECTOR_SOFT_ANCHOR)
    current_best = read_submission_coords(args.submission_dir / CURRENT_BEST_SUBMISSION)

    print("Preparing curvature gate from cached OOF labels", flush=True)
    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)
    fixed_oof = anchor_oof + 0.090 * train_correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    gate_labels = (fixed_dist < anchor_dist).astype(np.int8)
    gate_weights = boundary_weights(anchor_dist, fixed_dist)
    alpha_target = optimal_alpha(anchor_oof, train_correction, y)
    _ = alpha_target  # kept for parity with previous gate workflow and future extension.

    train_gate_features = make_gate_features(train_coords, train_features, anchor_oof, train_correction)
    gate_models = train_gate_models(train_gate_features, gate_labels, gate_weights)

    output_rows = []
    written: list[Path] = []

    candidate_anchors: dict[str, np.ndarray] = {
        "mirrorraw_temporal_w55": (1.0 - 0.55) * selector_soft + 0.55 * mirror_raw_temporal,
        "mirrortta_temporal_w55": (1.0 - 0.55) * selector_soft + 0.55 * mirror_tta_temporal,
    }
    for weight in TEMPORAL_WEIGHTS:
        candidate_anchors[f"mirrortta_temporal_w{int(weight * 100):02d}"] = (1.0 - weight) * selector_soft + weight * mirror_tta_temporal

    gated_candidates: dict[str, tuple[np.ndarray, float]] = {}
    for name, anchor in candidate_anchors.items():
        test_gate_features = make_gate_features(test_coords, test_features, anchor, test_correction)
        gate_proba = predict_gate(gate_models, test_gate_features)
        gated, route_fraction = apply_gate(anchor, test_correction, gate_proba)
        gated_candidates[f"{name}_gate_t52_a105"] = (gated, route_fraction)

    ranked_outputs: list[tuple[str, np.ndarray, float, str]] = []
    for name in [
        "mirrortta_temporal_w55_gate_t52_a105",
        "mirrortta_temporal_w50_gate_t52_a105",
        "mirrortta_temporal_w60_gate_t52_a105",
        "mirrorraw_temporal_w55_gate_t52_a105",
    ]:
        if name not in gated_candidates:
            continue
        pred, route_fraction = gated_candidates[name]
        ranked_outputs.append((name, pred, route_fraction, "pure_mirror_axis"))
        for blend in BEST_BLEND_WEIGHTS:
            blended = (1.0 - blend) * current_best + blend * pred
            ranked_outputs.append((f"{name}_bestblend{int(blend * 100):02d}", blended, route_fraction, "blend_with_current_best"))

    # Hand-picked order: use moderate blends first, then pure mirror-TTA if the new axis is strong.
    preferred_order = [
        "mirrortta_temporal_w55_gate_t52_a105_bestblend35",
        "mirrortta_temporal_w55_gate_t52_a105_bestblend50",
        "mirrortta_temporal_w50_gate_t52_a105_bestblend35",
        "mirrortta_temporal_w60_gate_t52_a105_bestblend35",
        "mirrortta_temporal_w55_gate_t52_a105",
        "mirrorraw_temporal_w55_gate_t52_a105_bestblend35",
    ]
    by_name = {name: (pred, route_fraction, kind) for name, pred, route_fraction, kind in ranked_outputs}
    for rank, name in enumerate(preferred_order, start=1):
        pred, route_fraction, kind = by_name[name]
        path = args.submission_dir / f"mirror_tta_rank{rank}_{name}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "name": name,
                "kind": kind,
                "route_fraction": route_fraction,
                **delta_summary(pred, current_best, "vs_current_best"),
                **delta_summary(pred, mirror_tta_temporal, "vs_mirror_tta_temporal"),
            }
        )
        if len(written) >= args.top_k:
            break

    report = [
        "# 2026-05-20 Mirror-Symmetry Temporal TTA Gate",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- current_best: `{CURRENT_BEST_SUBMISSION} = 0.69120`",
        "- idea: exploit left-right symmetry by augmenting train with `y -> -y` and averaging original/mirrored test predictions after unflipping.",
        f"- temporal_spec: `{TEMPORAL_SPEC_NAME}`",
        f"- temporal_mult: `{TEMPORAL_MULT}`",
        f"- gate: `threshold={GATE_THRESHOLD}`, `alpha={GATE_ALPHA}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Seed Diagnostics",
        "",
        dataframe_to_markdown(seed_diag),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- This is intentionally a new axis, not a threshold micro-tune.",
        "- If a blended candidate improves public, mirror-symmetry augmentation is useful but should be regularized.",
        "- If only the pure candidate improves, the temporal model itself benefits strongly from symmetry TTA.",
        "- If all candidates drop, the sensor-local left/right axis is not symmetric enough for this dataset.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
