from __future__ import annotations

import argparse
import re
import sys
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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import (  # noqa: E402
    boundary_weights,
    curvature_correction,
    make_gate_features,
    read_submission_coords,
    write_submission,
)
from run_hit_weighted_local_frame import make_features, physics_poly_candidates  # noqa: E402
from run_local_target_manifold_projection_20260521 import build_gate_oof_proba  # noqa: E402


CHAMPION = "curvgate_refine_rank2_gatet52a105.csv"
BACKUP_CHAMPION = "curvgate_rank4_gatet54a105.csv"
TEMPORAL55 = "temporalbc_refine_r1f102s100u100_w55.csv"
SELECTOR_SOFT = "direct_selector_rank2_selectorsoft.csv"
CURRENT_SCORE = 0.69120
R_HIT = 0.01

ACTION_SWITCH_FRACTIONS = [0.18, 0.28, 0.40, 0.55]
SMOOTH_BLEND_WEIGHTS = [0.18, 0.28, 0.40, 0.55]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def weighted_poly_weights(window: int, degree: int, decay: float, pred_step: float = 2.0) -> np.ndarray:
    t = np.arange(-(window - 1), 1, dtype=np.float64)
    powers = np.arange(degree + 1)
    design = t[:, None] ** powers[None, :]
    # Recent observations get weight 1.0, older observations decay geometrically.
    obs_w = decay ** np.arange(window - 1, -1, -1, dtype=np.float64)
    weighted_design = design * np.sqrt(obs_w)[:, None]
    pred_row = pred_step ** powers
    return pred_row @ np.linalg.pinv(weighted_design) @ np.diag(np.sqrt(obs_w))


def weighted_poly_prediction(coords: np.ndarray, window: int, degree: int, decay: float) -> np.ndarray:
    weights = weighted_poly_weights(window, degree, decay)
    return np.einsum("w,nwc->nc", weights, coords[:, -window:, :])


def smooth_candidate_block(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    configs = [
        (7, 2, 0.55),
        (7, 2, 0.70),
        (9, 2, 0.55),
        (9, 2, 0.70),
        (11, 2, 0.55),
        (11, 3, 0.60),
    ]
    preds = []
    names = []
    for window, degree, decay in configs:
        names.append(f"ewpoly_w{window}_d{degree}_r{int(decay * 100):02d}")
        preds.append(weighted_poly_prediction(coords, window, degree, decay))
    return np.stack(preds, axis=1), names


def make_champion_oof(
    coords: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    anchor_oof: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fixed_oof = anchor_oof + 0.090 * correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    labels = (fixed_dist < anchor_dist).astype(np.int8)
    weights = boundary_weights(anchor_dist, fixed_dist)
    gate_features = make_gate_features(coords, features, anchor_oof, correction)
    gate_proba = build_gate_oof_proba(gate_features, labels, weights)
    gate_t52 = anchor_oof.copy()
    gate_t52[gate_proba >= 0.52] = anchor_oof[gate_proba >= 0.52] + 0.105 * correction[gate_proba >= 0.52]
    gate_t54 = anchor_oof.copy()
    gate_t54[gate_proba >= 0.54] = anchor_oof[gate_proba >= 0.54] + 0.105 * correction[gate_proba >= 0.54]
    gate_t50 = anchor_oof.copy()
    gate_t50[gate_proba >= 0.50] = anchor_oof[gate_proba >= 0.50] + 0.105 * correction[gate_proba >= 0.50]
    return gate_t52, gate_t54, gate_t50, gate_proba


def make_action_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=380,
        learning_rate=0.026,
        num_leaves=47,
        min_child_samples=26,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.10,
        reg_lambda=1.25,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def action_features(coords: np.ndarray, base_features: np.ndarray, champion: np.ndarray, gate_signal: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(np.diff(diffs, axis=1), axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    last = coords[:, -1, :]
    champion_delta = champion - last
    dot = np.sum(d_last * d_prev, axis=1, keepdims=True)
    cos_turn = dot / (np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-12)
    compact = np.hstack(
        [
            base_features,
            d_last,
            d_prev,
            d_last - d_prev,
            speed[:, -6:],
            accel[:, -5:],
            cos_turn,
            champion_delta,
            np.linalg.norm(champion_delta, axis=1, keepdims=True),
            gate_signal[:, None],
        ]
    )
    return compact.astype(np.float32)


def average_aligned_proba(models: list[LGBMClassifier], features: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(features), n_classes), dtype=np.float64)
    for model in models:
        raw = model.predict_proba(features)
        for col, cls in enumerate(model.classes_):
            out[:, int(cls)] += raw[:, col] / len(models)
    return out


def build_candidate_library(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    temporal_oof: np.ndarray,
    selector_oof: np.ndarray,
    anchor_oof: np.ndarray,
    champion_oof: np.ndarray,
    gate_t54_oof: np.ndarray,
    gate_t50_oof: np.ndarray,
    train_correction: np.ndarray,
    test_correction: np.ndarray,
    submission_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    champion = read_submission_coords(submission_dir / CHAMPION)
    backup = read_submission_coords(submission_dir / BACKUP_CHAMPION)
    temporal = read_submission_coords(submission_dir / TEMPORAL55)
    selector = read_submission_coords(submission_dir / SELECTOR_SOFT)
    cochamp_oof = 0.5 * champion_oof + 0.5 * gate_t54_oof
    cochamp = 0.5 * champion + 0.5 * backup

    names = ["champion", "gate_t54", "cochamp", "gate_t50", "temporal55", "selector_soft", "fixed_a060", "fixed_a090", "fixed_a120"]
    train_preds = [
        champion_oof,
        gate_t54_oof,
        cochamp_oof,
        gate_t50_oof,
        temporal_oof,
        selector_oof,
        anchor_oof + 0.060 * train_correction,
        anchor_oof + 0.090 * train_correction,
        anchor_oof + 0.120 * train_correction,
    ]
    test_preds = [
        champion,
        backup,
        cochamp,
        read_submission_coords(submission_dir / "curvgate_rank2_gatet50a105.csv"),
        temporal,
        selector,
        temporal + 0.060 * test_correction,
        temporal + 0.090 * test_correction,
        temporal + 0.120 * test_correction,
    ]

    physics_names = [
        "phys_a000",
        "phys_a150",
        "phys_a275",
        "phys_a400",
        "phys_v098_a275",
        "phys_v102_a275",
        "poly_w3_d1",
        "poly_w3_d2",
        "poly_w4_d1",
        "poly_w4_d2",
        "poly_w5_d1",
        "poly_w5_d2",
        "poly_w7_d1",
        "poly_w7_d2",
        "poly_w11_d1",
        "poly_w11_d2",
        "poly_w11_d3",
        "wdiff_w5_d050",
        "wdiff_w5_d075",
        "wdiff_w7_d060",
        "wdiff_w11_d070",
    ]
    train_phys = physics_poly_candidates(train_coords)
    test_phys = physics_poly_candidates(test_coords)
    for idx, name in enumerate(physics_names):
        names.append(name)
        train_preds.append(train_phys[:, idx, :])
        test_preds.append(test_phys[:, idx, :])

    train_smooth, smooth_names = smooth_candidate_block(train_coords)
    test_smooth, _ = smooth_candidate_block(test_coords)
    for idx, name in enumerate(smooth_names):
        names.append(name)
        train_preds.append(train_smooth[:, idx, :])
        test_preds.append(test_smooth[:, idx, :])

    return names, np.stack(train_preds, axis=1), np.stack(test_preds, axis=1)


def evaluate_smooth_axis(
    smooth_names: list[str],
    train_smooth: np.ndarray,
    test_smooth: np.ndarray,
    champion_oof: np.ndarray,
    champion_test: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    arrays = {}
    for idx, name in enumerate(smooth_names):
        raw_train = train_smooth[:, idx, :]
        raw_test = test_smooth[:, idx, :]
        for weight in SMOOTH_BLEND_WEIGHTS:
            pred_train = (1.0 - weight) * champion_oof + weight * raw_train
            pred_test = (1.0 - weight) * champion_test + weight * raw_test
            out_name = f"smooth_{name}_blend{int(weight * 100):02d}"
            arrays[out_name] = pred_test
            rows.append(
                {
                    "name": out_name,
                    "smooth_candidate": name,
                    "blend_weight": weight,
                    **distance_summary(pred_train, y),
                    **delta_summary(pred_test, champion_test, "test_vs_champion"),
                }
            )
    df = pd.DataFrame(rows).sort_values(["r_hit_1cm", "mean_distance"], ascending=[False, True])
    return df, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two aggressive new axes for 2026-05-23.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_aggressive_new_axes_20260523.md")
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

    cache = np.load(args.cache_path)
    temporal_oof = cache["temporal_oof"]
    selector_oof = cache["selector_oof"]
    anchor_oof = cache["anchor_oof"]
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)

    print("Building champion OOF proxy", flush=True)
    champion_oof, gate_t54_oof, gate_t50_oof, gate_oof_proba = make_champion_oof(
        train_coords,
        y,
        train_features,
        anchor_oof,
        train_correction,
    )
    champion_test = read_submission_coords(args.submission_dir / CHAMPION)

    print("Axis A: smoothed polynomial physics", flush=True)
    train_smooth, smooth_names = smooth_candidate_block(train_coords)
    test_smooth, _ = smooth_candidate_block(test_coords)
    smooth_lb, smooth_arrays = evaluate_smooth_axis(smooth_names, train_smooth, test_smooth, champion_oof, champion_test, y)
    smooth_choice = smooth_lb.iloc[0]
    smooth_name = str(smooth_choice["name"])
    smooth_path = args.submission_dir / f"newaxis_smooth_rank1_{slug(smooth_name)}.csv"
    write_submission(sample_submission, smooth_arrays[smooth_name], smooth_path)

    print("Axis B: action distillation selector", flush=True)
    candidate_names, train_candidates, test_candidates = build_candidate_library(
        train_coords,
        test_coords,
        temporal_oof,
        selector_oof,
        anchor_oof,
        champion_oof,
        gate_t54_oof,
        gate_t50_oof,
        train_correction,
        test_correction,
        args.submission_dir,
    )
    candidate_dist = np.linalg.norm(train_candidates - y[:, None, :], axis=2)
    labels = np.argmin(candidate_dist, axis=1).astype(np.int32)
    champion_dist = candidate_dist[:, 0]
    best_dist = np.min(candidate_dist, axis=1)
    champion_hit = champion_dist <= R_HIT
    best_hit = best_dist <= R_HIT
    rescue = (~champion_hit) & best_hit
    boundary = np.exp(-0.5 * ((np.minimum(champion_dist, best_dist) - R_HIT) / 0.003) ** 2)
    weights = 1.0 + 10.0 * rescue + 6.0 * boundary + 2.0 * (labels != 0)
    weights = np.clip(weights, 1.0, 24.0)
    weights = weights / np.mean(weights)

    train_action_features = action_features(train_coords, train_features, champion_oof, gate_oof_proba)
    test_gate_signal = (np.linalg.norm(champion_test - read_submission_coords(args.submission_dir / TEMPORAL55), axis=1) > 1e-12).astype(np.float64)
    test_action_features = action_features(test_coords, test_features, champion_test, test_gate_signal)

    models = []
    for seed in [2301, 2302, 2303]:
        print(f"  action model seed={seed}", flush=True)
        model = make_action_classifier(seed)
        model.fit(train_action_features, labels, sample_weight=weights)
        models.append(model)
    proba = average_aligned_proba(models, test_action_features, train_candidates.shape[1])
    chosen = np.argmax(proba, axis=1)
    non_champion_conf = 1.0 - proba[:, 0]
    action_rows = []
    action_arrays = {}
    for fraction in ACTION_SWITCH_FRACTIONS:
        n_switch = max(1, int(round(len(chosen) * fraction)))
        threshold = np.partition(non_champion_conf, len(non_champion_conf) - n_switch)[len(non_champion_conf) - n_switch]
        mask = (non_champion_conf >= threshold) & (chosen != 0)
        pred = champion_test.copy()
        pred[mask] = test_candidates[np.arange(len(pred))[mask], chosen[mask], :]
        name = f"actiondistill_top{int(fraction * 100):02d}"
        action_arrays[name] = pred
        action_rows.append(
            {
                "name": name,
                "fraction": fraction,
                "actual_switch_fraction": float(np.mean(mask)),
                "top_candidate_mode": candidate_names[int(pd.Series(chosen[mask]).mode().iloc[0])] if np.any(mask) else "none",
                "mean_non_champion_conf": float(np.mean(non_champion_conf)),
                "min_routed_conf": float(np.min(non_champion_conf[mask])) if np.any(mask) else 0.0,
                **delta_summary(pred, champion_test, "vs_champion"),
            }
        )
    action_df = pd.DataFrame(action_rows)
    action_choice = action_df.sort_values(["fraction"]).iloc[1]
    action_name = str(action_choice["name"])
    action_path = args.submission_dir / f"newaxis_action_rank1_{slug(action_name)}.csv"
    write_submission(sample_submission, action_arrays[action_name], action_path)

    oracle_rows = []
    for idx, name in enumerate(candidate_names):
        pred = train_candidates[:, idx, :]
        dist = candidate_dist[:, idx]
        oracle_rows.append(
            {
                "candidate": name,
                "hit_rate": float(np.mean(dist <= R_HIT)),
                "rescue_rate_vs_champion": float(np.mean((champion_dist > R_HIT) & (dist <= R_HIT))),
                "harm_rate_vs_champion": float(np.mean((champion_dist <= R_HIT) & (dist > R_HIT))),
                "mean_distance": float(np.mean(dist)),
            }
        )
    oracle_df = pd.DataFrame(oracle_rows).sort_values(["rescue_rate_vs_champion", "hit_rate"], ascending=[False, False])

    output_df = pd.DataFrame(
        [
            {
                "axis": "smooth_physics",
                "submission": smooth_path.name,
                "selected": smooth_name,
                "oof_r_hit": float(smooth_choice["r_hit_1cm"]),
                "test_vs_champion_mean_delta": float(smooth_choice["test_vs_champion_mean_delta"]),
                "test_vs_champion_p95_delta": float(smooth_choice["test_vs_champion_p95_delta"]),
            },
            {
                "axis": "action_distillation",
                "submission": action_path.name,
                "selected": action_name,
                "oof_r_hit": np.nan,
                "test_vs_champion_mean_delta": float(action_choice["vs_champion_mean_delta"]),
                "test_vs_champion_p95_delta": float(action_choice["vs_champion_p95_delta"]),
            },
        ]
    )

    report = [
        "# 2026-05-23 Aggressive New Axes",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- champion: `{CHAMPION} = {CURRENT_SCORE:.5f}`",
        f"- generated_outputs: `{[str(smooth_path), str(action_path)]}`",
        "",
        "## Idea",
        "",
        "- Previous temporal curriculum probes all scored 0.69060, so this run intentionally tests two different assumptions.",
        "- Axis A: exp-weighted polynomial smoothing/denoising as a fresh physics bias.",
        "- Axis B: train OOF oracle action distillation, switching some samples to the candidate family predicted to rescue hits.",
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(output_df),
        "",
        "## Smooth Axis OOF",
        "",
        dataframe_to_markdown(smooth_lb.head(30)),
        "",
        "## Action Switch Diagnostics",
        "",
        dataframe_to_markdown(action_df),
        "",
        "## Candidate Oracle Diagnostics",
        "",
        dataframe_to_markdown(oracle_df.head(40)),
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
