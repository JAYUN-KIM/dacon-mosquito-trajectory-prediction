from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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
from run_direct_step_refine_20260509 import (  # noqa: E402
    WeightSpec,
    direct_prediction,
    direct_target_local,
    sample_weights,
    write_submission,
)
from run_hit_weighted_local_frame import fit_predict_axes_weighted, make_features  # noqa: E402


PUBLIC_BEST_SUBMISSION = "direct_selector_rank2_selectorsoft.csv"
PUBLIC_BEST_SCORE = 0.68440
TRUE_WEIGHT_SPEC = WeightSpec("ca_a6_s0055_c0105", "ca", 6.0, 0.0055, 0.0105)
CV_SEEDS = [42, 777]
FULL_SEEDS = [42, 777, 2026]
MULTS = [
    (1.02, 1.06, 0.94),
    (1.02, 1.04, 0.94),
    (1.02, 1.04, 0.96),
    (1.01, 1.04, 0.96),
    (1.03, 1.06, 0.94),
    (1.03, 1.08, 0.92),
    (1.02, 1.00, 1.00),
]
ANCHOR_BLEND_WEIGHTS = [0.20, 0.35, 0.50]


@dataclass(frozen=True)
class AugmentSpec:
    name: str
    cutoffs: tuple[int, ...]
    pseudo_weight: float
    pseudo_boundary: bool
    n_estimators_hint: int = 0


SPECS = [
    AugmentSpec("tb_c8_w015", (8,), 0.15, True),
    AugmentSpec("tb_c8_w030", (8,), 0.30, True),
    AugmentSpec("tb_c78_w015", (7, 8), 0.15, True),
    AugmentSpec("tb_c78_w030", (7, 8), 0.30, True),
    AugmentSpec("tb_c678_w010", (6, 7, 8), 0.10, True),
    AugmentSpec("tb_c678_w020", (6, 7, 8), 0.20, True),
    AugmentSpec("tb_c8_flatw030", (8,), 0.30, False),
    AugmentSpec("tb_c78_flatw020", (7, 8), 0.20, False),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_submission_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def pseudo_from_cutoff(coords: np.ndarray, cutoff: int, horizon_steps: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if cutoff + horizon_steps >= coords.shape[1]:
        raise ValueError(f"cutoff={cutoff} with horizon={horizon_steps} exceeds trajectory length={coords.shape[1]}")
    prefix = coords[:, : cutoff + 1, :]
    missing = coords.shape[1] - prefix.shape[1]
    first_step = coords[:, 1, :] - coords[:, 0, :]
    backcast = [coords[:, 0, :] - step * first_step for step in range(missing, 0, -1)]
    pseudo_coords = np.concatenate([np.stack(backcast, axis=1), prefix], axis=1)
    pseudo_y = coords[:, cutoff + horizon_steps, :]
    return pseudo_coords.astype(np.float64), pseudo_y.astype(np.float64)


def build_pseudo_samples(coords: np.ndarray, spec: AugmentSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_blocks = []
    target_blocks = []
    weight_blocks = []
    for cutoff in spec.cutoffs:
        pseudo_coords, pseudo_y = pseudo_from_cutoff(coords, cutoff)
        coord_blocks.append(pseudo_coords)
        target_blocks.append(pseudo_y)
        if spec.pseudo_boundary:
            weights = sample_weights(pseudo_coords, pseudo_y, TRUE_WEIGHT_SPEC) * spec.pseudo_weight
        else:
            weights = np.full(len(pseudo_coords), spec.pseudo_weight, dtype=np.float64)
        weight_blocks.append(weights)
    return (
        np.concatenate(coord_blocks, axis=0),
        np.concatenate(target_blocks, axis=0),
        np.concatenate(weight_blocks, axis=0),
    )


def augmented_train_matrix(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    spec: AugmentSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_target = direct_target_local(coords, y)
    true_weights = sample_weights(coords, y, TRUE_WEIGHT_SPEC)

    pseudo_coords, pseudo_y, pseudo_weights = build_pseudo_samples(coords, spec)
    pseudo_features, _ = make_features(pseudo_coords)
    pseudo_target = direct_target_local(pseudo_coords, pseudo_y)

    train_x = np.vstack([features, pseudo_features]).astype(np.float32)
    train_y = np.vstack([true_target, pseudo_target]).astype(np.float64)
    weights = np.concatenate([true_weights, pseudo_weights]).astype(np.float64)
    weights = weights / np.mean(weights)
    return train_x, train_y, weights


def fit_predict_augmented(
    train_coords: np.ndarray,
    train_y: np.ndarray,
    train_features: np.ndarray,
    pred_features: np.ndarray,
    spec: AugmentSpec,
    seed: int,
) -> np.ndarray:
    aug_x, aug_y, aug_w = augmented_train_matrix(train_coords, train_y, train_features, spec)
    print(
        f"    augmented rows={len(aug_x)} true={len(train_coords)} pseudo={len(aug_x) - len(train_coords)} "
        f"weight_mean={aug_w.mean():.4f}",
        flush=True,
    )
    return fit_predict_axes_weighted(aug_x, aug_y, pred_features, seed, "l2", aug_w)


def evaluate_cv(coords: np.ndarray, y: np.ndarray, features: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed in CV_SEEDS:
        print(f"CV seed={seed}", flush=True)
        val_mask = split_mask(len(coords), 0.2, seed)
        train_mask = ~val_mask
        for spec in SPECS:
            print(f"  fitting {spec.name}", flush=True)
            pred_local = fit_predict_augmented(
                coords[train_mask],
                y[train_mask],
                features[train_mask],
                features[val_mask],
                spec,
                160000 + seed,
            )
            for mult in MULTS:
                pred = direct_prediction(coords[val_mask], pred_local, mult)
                rows.append(
                    {
                        "seed": seed,
                        "spec": spec.name,
                        "cutoffs": ",".join(str(cutoff) for cutoff in spec.cutoffs),
                        "pseudo_weight": spec.pseudo_weight,
                        "pseudo_boundary": spec.pseudo_boundary,
                        "forward_mult": mult[0],
                        "side_mult": mult[1],
                        "up_mult": mult[2],
                        **distance_summary(pred, y[val_mask]),
                    }
                )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["spec", "cutoffs", "pseudo_weight", "pseudo_boundary", "forward_mult", "side_mult", "up_mult"], as_index=False)
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


def full_predict(
    train_coords: np.ndarray,
    y: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
    spec: AugmentSpec,
) -> np.ndarray:
    preds = []
    for seed in FULL_SEEDS:
        print(f"  full seed={seed} {spec.name}", flush=True)
        preds.append(fit_predict_augmented(train_coords, y, train_features, test_features, spec, 260000 + seed))
    return np.mean(preds, axis=0)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal backcast self-supervised augmentation for direct-step model.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_temporal_backcast_augmentation_20260516.md")
    parser.add_argument("--top-k", type=int, default=5)
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

    print("Building feature matrices", flush=True)
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)

    print("Evaluating temporal backcast augmentation CV", flush=True)
    leaderboard = evaluate_cv(train_coords, y, train_features)
    print(leaderboard.head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    specs = {spec.name: spec for spec in SPECS}
    public_anchor = read_submission_coords(args.submission_dir / PUBLIC_BEST_SUBMISSION)
    pred_local_cache: dict[str, np.ndarray] = {}
    output_rows = []
    written = []
    used = 0
    for _, row in leaderboard.iterrows():
        spec_name = str(row["spec"])
        if spec_name not in pred_local_cache:
            print(f"Training full temporal-backcast model: {spec_name}", flush=True)
            pred_local_cache[spec_name] = full_predict(train_coords, y, train_features, test_features, specs[spec_name])
        mult = (float(row["forward_mult"]), float(row["side_mult"]), float(row["up_mult"]))
        pred = direct_prediction(test_coords, pred_local_cache[spec_name], mult)

        used += 1
        path = args.submission_dir / f"temporalbc_rank{used}_{slug(spec_name)}_f{mult[0]:.2f}_s{mult[1]:.2f}_u{mult[2]:.2f}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": used,
                "submission": path.name,
                "spec": spec_name,
                "forward_mult": mult[0],
                "side_mult": mult[1],
                "up_mult": mult[2],
                "cv_mean_r_hit": float(row["mean_r_hit"]),
                "cv_mean_distance": float(row["mean_distance"]),
                **delta_summary(pred, public_anchor, "vs_public_best"),
            }
        )

        for blend in ANCHOR_BLEND_WEIGHTS:
            blend_pred = (1.0 - blend) * public_anchor + blend * pred
            blend_path = args.submission_dir / (
                f"temporalbc_rank{used}_anchorblend{int(blend * 100):02d}_{slug(spec_name)}_"
                f"f{mult[0]:.2f}_s{mult[1]:.2f}_u{mult[2]:.2f}.csv"
            )
            write_submission(sample_submission, blend_pred, blend_path)
            written.append(blend_path)
            output_rows.append(
                {
                    "rank": f"{used}b{int(blend * 100):02d}",
                    "submission": blend_path.name,
                    "spec": spec_name,
                    "forward_mult": mult[0],
                    "side_mult": mult[1],
                    "up_mult": mult[2],
                    "cv_mean_r_hit": float(row["mean_r_hit"]),
                    "cv_mean_distance": float(row["mean_distance"]),
                    **delta_summary(blend_pred, public_anchor, "vs_public_best"),
                }
            )

        if used >= args.top_k:
            break

    report = [
        "# 2026-05-16 Temporal Backcast Augmentation",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Problem Redefinition",
        "",
        "- Reset axis: instead of routing among existing candidates, use the internal time structure of each trajectory to create pseudo +80ms tasks.",
        "- For cutoff c, the model sees points up to c and predicts c+2. Missing early history is linearly backcast so every pseudo sample still has 11 input points.",
        "- Pseudo rows are down-weighted heavily, because the true competition target is still the provided +80ms label.",
        "",
        "## CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- This is intentionally experimental. If public improves, the next step is stronger pseudo curricula and regime-specific pseudo weights.",
        "- If public drops, it suggests within-history dynamics are distribution-shifted from the +80ms target or the backcast padding injects noise.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
