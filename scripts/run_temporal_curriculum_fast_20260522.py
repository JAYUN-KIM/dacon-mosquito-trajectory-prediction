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
from run_aggressive_experiments import dataframe_to_markdown, stack_samples  # noqa: E402
from run_direct_step_refine_20260509 import (  # noqa: E402
    WeightSpec,
    direct_prediction,
    direct_target_local,
    sample_weights,
    write_submission,
)
from run_hit_weighted_local_frame import fit_predict_axes_weighted, make_features  # noqa: E402


CHAMPION = "curvgate_refine_rank2_gatet52a105.csv"
BACKUP_CHAMPION = "curvgate_rank4_gatet54a105.csv"
TEMPORAL55 = "temporalbc_refine_r1f102s100u100_w55.csv"
SELECTOR_SOFT = "direct_selector_rank2_selectorsoft.csv"
TRUE_WEIGHT_SPEC = WeightSpec("ca_a6_s0055_c0105", "ca", 6.0, 0.0055, 0.0105)
FULL_SEEDS = [42, 777, 2026]


@dataclass(frozen=True)
class CurriculumSpec:
    name: str
    true_cutoffs: tuple[int, ...]
    true_weight: float
    velocity_cutoffs: tuple[int, ...]
    velocity_weight: float
    flat_true_weight: float = 0.0


SPECS = [
    CurriculumSpec("tc_c5678w012_v6789w006", (5, 6, 7, 8), 0.12, (6, 7, 8, 9), 0.06),
    CurriculumSpec("tc_c45678w006_v789w008", (4, 5, 6, 7, 8), 0.06, (7, 8, 9), 0.08),
    CurriculumSpec("tc_c678w022_v89w004", (6, 7, 8), 0.22, (8, 9), 0.04),
]

MULTS = [
    (1.02, 1.00, 1.00),
    (1.02, 1.04, 0.96),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_submission_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def pseudo_prefix(coords: np.ndarray, cutoff: int) -> np.ndarray:
    prefix = coords[:, : cutoff + 1, :]
    missing = coords.shape[1] - prefix.shape[1]
    first_step = coords[:, 1, :] - coords[:, 0, :]
    backcast = [coords[:, 0, :] - step * first_step for step in range(missing, 0, -1)]
    return np.concatenate([np.stack(backcast, axis=1), prefix], axis=1).astype(np.float64)


def build_true_pseudo(coords: np.ndarray, cutoffs: tuple[int, ...], weight: float, boundary: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_blocks = []
    target_blocks = []
    weight_blocks = []
    for cutoff in cutoffs:
        if cutoff + 2 >= coords.shape[1]:
            continue
        pseudo_coords = pseudo_prefix(coords, cutoff)
        pseudo_y = coords[:, cutoff + 2, :]
        coord_blocks.append(pseudo_coords)
        target_blocks.append(pseudo_y)
        if boundary:
            weight_values = sample_weights(pseudo_coords, pseudo_y, TRUE_WEIGHT_SPEC) * weight
        else:
            weight_values = np.full(len(coords), weight, dtype=np.float64)
        weight_blocks.append(weight_values)
    if not coord_blocks:
        return np.empty((0, 11, 3)), np.empty((0, 3)), np.empty((0,))
    return np.concatenate(coord_blocks), np.concatenate(target_blocks), np.concatenate(weight_blocks)


def build_velocity_pseudo(coords: np.ndarray, cutoffs: tuple[int, ...], weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_blocks = []
    target_blocks = []
    weight_blocks = []
    for cutoff in cutoffs:
        if cutoff + 1 >= coords.shape[1]:
            continue
        pseudo_coords = pseudo_prefix(coords, cutoff)
        one_step = coords[:, cutoff + 1, :] - coords[:, cutoff, :]
        pseudo_y = coords[:, cutoff, :] + 2.0 * one_step
        coord_blocks.append(pseudo_coords)
        target_blocks.append(pseudo_y)
        # Velocity pseudo rows are intentionally weak: useful bias, dangerous label.
        weight_blocks.append(np.full(len(coords), weight, dtype=np.float64))
    if not coord_blocks:
        return np.empty((0, 11, 3)), np.empty((0, 3)), np.empty((0,))
    return np.concatenate(coord_blocks), np.concatenate(target_blocks), np.concatenate(weight_blocks)


def augmented_train_matrix(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    spec: CurriculumSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_target = direct_target_local(coords, y)
    true_weights = sample_weights(coords, y, TRUE_WEIGHT_SPEC)

    pseudo_coords_1, pseudo_y_1, pseudo_w_1 = build_true_pseudo(coords, spec.true_cutoffs, spec.true_weight, True)
    pseudo_coords_2, pseudo_y_2, pseudo_w_2 = build_velocity_pseudo(coords, spec.velocity_cutoffs, spec.velocity_weight)
    blocks_coords = [pseudo_coords_1, pseudo_coords_2]
    blocks_y = [pseudo_y_1, pseudo_y_2]
    blocks_w = [pseudo_w_1, pseudo_w_2]
    if spec.flat_true_weight > 0:
        pseudo_coords_3, pseudo_y_3, pseudo_w_3 = build_true_pseudo(coords, spec.true_cutoffs, spec.flat_true_weight, False)
        blocks_coords.append(pseudo_coords_3)
        blocks_y.append(pseudo_y_3)
        blocks_w.append(pseudo_w_3)

    pseudo_coords = np.concatenate([block for block in blocks_coords if len(block)], axis=0)
    pseudo_y = np.concatenate([block for block in blocks_y if len(block)], axis=0)
    pseudo_weights = np.concatenate([block for block in blocks_w if len(block)], axis=0)
    pseudo_features, _ = make_features(pseudo_coords)
    pseudo_target = direct_target_local(pseudo_coords, pseudo_y)

    train_x = np.vstack([features, pseudo_features]).astype(np.float32)
    train_y = np.vstack([true_target, pseudo_target]).astype(np.float64)
    weights = np.concatenate([true_weights, pseudo_weights]).astype(np.float64)
    weights = weights / np.mean(weights)
    return train_x, train_y, weights


def fit_predict_curriculum(
    train_coords: np.ndarray,
    y: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
    spec: CurriculumSpec,
) -> np.ndarray:
    preds = []
    for seed in FULL_SEEDS:
        print(f"  seed={seed} spec={spec.name}", flush=True)
        train_x, train_y, weights = augmented_train_matrix(train_coords, y, train_features, spec)
        print(
            f"    rows={len(train_x)} pseudo={len(train_x) - len(train_coords)} "
            f"w_mean={weights.mean():.4f}",
            flush=True,
        )
        preds.append(fit_predict_axes_weighted(train_x, train_y, test_features, 520000 + seed, "l2", weights))
    return np.mean(preds, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast temporal curriculum candidates for 2026-05-22.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_temporal_curriculum_fast_20260522.md")
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

    champion = read_submission_coords(args.submission_dir / CHAMPION)
    backup = read_submission_coords(args.submission_dir / BACKUP_CHAMPION)
    temporal55 = read_submission_coords(args.submission_dir / TEMPORAL55)
    selector = read_submission_coords(args.submission_dir / SELECTOR_SOFT)
    cochamp = 0.5 * champion + 0.5 * backup

    rows = []
    written = []
    candidate_arrays: dict[str, np.ndarray] = {}
    for spec in SPECS:
        print(f"Training curriculum spec: {spec.name}", flush=True)
        pred_local = fit_predict_curriculum(train_coords, y, train_features, test_features, spec)
        for mult in MULTS:
            direct_pred = direct_prediction(test_coords, pred_local, mult)
            f, s, u = mult
            candidates = [
                (f"{spec.name}_direct_f{f:.2f}s{s:.2f}u{u:.2f}", direct_pred),
                (f"{spec.name}_champblend15_f{f:.2f}s{s:.2f}u{u:.2f}", 0.85 * champion + 0.15 * direct_pred),
                (f"{spec.name}_champblend25_f{f:.2f}s{s:.2f}u{u:.2f}", 0.75 * champion + 0.25 * direct_pred),
                (f"{spec.name}_cochampblend20_f{f:.2f}s{s:.2f}u{u:.2f}", 0.80 * cochamp + 0.20 * direct_pred),
                (f"{spec.name}_temporalblend45_f{f:.2f}s{s:.2f}u{u:.2f}", 0.55 * selector + 0.45 * direct_pred),
                (f"{spec.name}_temporal55blend35_f{f:.2f}s{s:.2f}u{u:.2f}", 0.65 * temporal55 + 0.35 * direct_pred),
            ]
            for name, pred in candidates:
                candidate_arrays[name] = pred
                rows.append(
                    {
                        "name": name,
                        "spec": spec.name,
                        "forward_mult": f,
                        "side_mult": s,
                        "up_mult": u,
                        **delta_summary(pred, champion, "vs_champion"),
                        **delta_summary(pred, cochamp, "vs_cochamp"),
                    }
                )

    df = pd.DataFrame(rows)
    # Submission ordering is deliberately conservative first, then one larger swing.
    priority_names = [
        "tc_c5678w012_v6789w006_champblend15_f1.02s1.00u1.00",
        "tc_c5678w012_v6789w006_champblend25_f1.02s1.00u1.00",
        "tc_c45678w006_v789w008_champblend15_f1.02s1.00u1.00",
        "tc_c678w022_v89w004_champblend15_f1.02s1.00u1.00",
        "tc_c5678w012_v6789w006_cochampblend20_f1.02s1.00u1.00",
    ]
    selected_rows = []
    rank = 1
    for name in priority_names:
        if name not in candidate_arrays:
            continue
        pred = candidate_arrays[name]
        path = args.submission_dir / f"tempcurr_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        row = df.loc[df["name"] == name].iloc[0].to_dict()
        row.update({"rank": rank, "submission": path.name})
        selected_rows.append(row)
        rank += 1
        if rank > 5:
            break

    out_df = pd.DataFrame(selected_rows)
    report = [
        "# 2026-05-22 Fast Temporal Curriculum",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- champion: `{CHAMPION} = 0.69120`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Fast 0.7 attempt: stop post-processing and expand the temporal-backcast supervision itself.",
        "- Add wider true +80ms internal cutoffs and weak horizon1 velocity pseudo-labels.",
        "- Submit conservative champion blends first because prior broad post-process probes dropped hard.",
        "",
        "## Candidate Diagnostics",
        "",
        dataframe_to_markdown(df.sort_values(["vs_champion_mean_delta", "vs_champion_p95_delta"]).head(80)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(out_df),
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
