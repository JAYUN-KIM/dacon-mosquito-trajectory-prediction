from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


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
from run_aggressive_experiments import (  # noqa: E402
    dataframe_to_markdown,
    distance_summary,
    stack_samples,
)
from run_curvature_gate_20260519 import delta_summary, read_submission_coords, write_submission  # noqa: E402
from run_direct_step_refine_20260509 import direct_prediction, direct_target_local  # noqa: E402
from run_hit_weighted_local_frame import make_features  # noqa: E402


CHAMPION_SUBMISSION = "champmicro_rank3_gatet520a1025.csv"
FALLBACK_CHAMPION = "champalpha_rank1_t52a1015.csv"
PUBLIC_BEST_SCORE = 0.69140
R_HIT = 0.01
OOF_FOLDS = 3
OOF_SEED = 20260526
FULL_SEEDS = [42]

MULTS = [
    (1.00, 1.00, 1.00),
    (1.04, 1.00, 1.00),
    (1.00, 1.04, 0.96),
]
GLOBAL_BLEND_WEIGHTS = [0.04, 0.08, 0.12, 0.18]
GATE_FRACS = [0.030, 0.080]
GATE_BLEND_WEIGHTS = [0.25, 0.40]
SOFT_THRESHOLDS = [0.60]
SOFT_BLEND_WEIGHTS = [0.18]


@dataclass(frozen=True)
class OneStepSpec:
    name: str
    cutoffs: tuple[int, ...]
    boundary_center: float
    boundary_sigma: float
    boundary_amp: float
    recent_boost: float


@dataclass(frozen=True)
class Candidate:
    kind: str
    spec: str
    mult: tuple[float, float, float]
    weight: float
    frac: float = 0.0
    threshold: float = 0.0


SPECS = [
    OneStepSpec("os_c789_b006_recent", (7, 8, 9), 0.0060, 0.0030, 5.0, 0.35),
    OneStepSpec("os_c89_b005_late", (8, 9), 0.0055, 0.0025, 6.0, 0.50),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def mult_slug(mult: tuple[float, float, float]) -> str:
    return f"f{mult[0]:.2f}s{mult[1]:.2f}u{mult[2]:.2f}".replace(".", "")


def make_folds(n_rows: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    return [fold.astype(np.int64) for fold in np.array_split(indices, n_folds)]


def append_point(coords: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.concatenate([coords[:, 1:, :], point[:, None, :]], axis=1)


def one_step_cv(coords: np.ndarray, velocity_scale: float = 1.0, accel_scale: float = 0.20) -> np.ndarray:
    last = coords[:, -1, :]
    d_last = coords[:, -1, :] - coords[:, -2, :]
    d_prev = coords[:, -2, :] - coords[:, -3, :]
    return last + velocity_scale * d_last + accel_scale * (d_last - d_prev)


def pseudo_from_cutoff(coords: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    if cutoff + 1 >= coords.shape[1]:
        raise ValueError(f"cutoff={cutoff} has no observed one-step target")
    prefix = coords[:, : cutoff + 1, :]
    missing = coords.shape[1] - prefix.shape[1]
    first_step = coords[:, 1, :] - coords[:, 0, :]
    backcast = [coords[:, 0, :] - step * first_step for step in range(missing, 0, -1)]
    pseudo_coords = np.concatenate([np.stack(backcast, axis=1), prefix], axis=1)
    pseudo_y = coords[:, cutoff + 1, :]
    return pseudo_coords.astype(np.float64), pseudo_y.astype(np.float64)


def build_onestep_training(coords: np.ndarray, spec: OneStepSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_blocks = []
    target_blocks = []
    weight_blocks = []
    min_cutoff = min(spec.cutoffs)
    max_cutoff = max(spec.cutoffs)
    denom = max(1, max_cutoff - min_cutoff)
    for cutoff in spec.cutoffs:
        pseudo_coords, pseudo_y = pseudo_from_cutoff(coords, cutoff)
        features, _ = make_features(pseudo_coords)
        target = direct_target_local(pseudo_coords, pseudo_y)

        ref_dist = np.linalg.norm(one_step_cv(pseudo_coords) - pseudo_y, axis=1)
        weights = 1.0 + spec.boundary_amp * np.exp(-0.5 * ((ref_dist - spec.boundary_center) / spec.boundary_sigma) ** 2)
        recent_factor = 1.0 + spec.recent_boost * ((cutoff - min_cutoff) / denom)
        weights = np.clip(weights * recent_factor, 0.5, 10.0)
        feature_blocks.append(features)
        target_blocks.append(target)
        weight_blocks.append(weights)

    train_x = np.vstack(feature_blocks).astype(np.float32)
    train_y = np.vstack(target_blocks).astype(np.float64)
    weights = np.concatenate(weight_blocks).astype(np.float64)
    weights = weights / np.mean(weights)
    return train_x, train_y, weights


def make_fast_lgbm(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.040,
        num_leaves=31,
        min_child_samples=22,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.03,
        reg_lambda=0.35,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_axis_models(train_x: np.ndarray, train_y: np.ndarray, weights: np.ndarray, seed: int):
    models = []
    for axis in range(3):
        model = make_fast_lgbm(seed + axis)
        model.fit(train_x, train_y[:, axis], sample_weight=weights)
        models.append(model)
    return models


def predict_axis_models(models, features: np.ndarray) -> np.ndarray:
    return np.vstack([model.predict(features) for model in models]).T.astype(np.float64)


def recursive_predict_with_models(models, coords: np.ndarray, mult: tuple[float, float, float]) -> np.ndarray:
    features_1, _ = make_features(coords)
    step1_local = predict_axis_models(models, features_1)
    step1 = direct_prediction(coords, step1_local, mult)
    coords_1 = append_point(coords, step1)
    features_2, _ = make_features(coords_1)
    step2_local = predict_axis_models(models, features_2)
    return direct_prediction(coords_1, step2_local, mult)


def build_oof_recursive(coords: np.ndarray, spec: OneStepSpec, folds: list[np.ndarray]) -> dict[tuple[float, float, float], np.ndarray]:
    preds = {mult: np.zeros((len(coords), 3), dtype=np.float64) for mult in MULTS}
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(coords), dtype=bool)
        train_mask[val_idx] = False
        print(
            f"OOF {spec.name} fold {fold_idx}/{len(folds)}: "
            f"train={int(train_mask.sum())} val={len(val_idx)}",
            flush=True,
        )
        train_x, train_y, weights = build_onestep_training(coords[train_mask], spec)
        models = fit_axis_models(train_x, train_y, weights, 526000 + fold_idx * 101 + len(spec.cutoffs))
        for mult in MULTS:
            preds[mult][val_idx] = recursive_predict_with_models(models, coords[val_idx], mult)
    return preds


def full_recursive_predictions(coords: np.ndarray, test_coords: np.ndarray, spec: OneStepSpec) -> dict[tuple[float, float, float], np.ndarray]:
    seed_step_preds = {mult: [] for mult in MULTS}
    train_x, train_y, weights = build_onestep_training(coords, spec)
    for seed in FULL_SEEDS:
        print(f"Full recursive one-step fit {spec.name} seed={seed}", flush=True)
        models = fit_axis_models(train_x, train_y, weights, 626000 + seed + len(spec.cutoffs))
        for mult in MULTS:
            seed_step_preds[mult].append(recursive_predict_with_models(models, test_coords, mult))
    return {mult: np.mean(pred_list, axis=0) for mult, pred_list in seed_step_preds.items()}


def gain_features(coords: np.ndarray, champion: np.ndarray, recursive: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    delta = recursive - champion
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    champ_delta = champion - coords[:, -1, :]
    rec_delta = recursive - coords[:, -1, :]
    champ_norm = np.linalg.norm(champ_delta, axis=1, keepdims=True)
    rec_norm = np.linalg.norm(rec_delta, axis=1, keepdims=True)
    dot = np.sum(champ_delta * delta, axis=1, keepdims=True)
    cos = dot / (champ_norm * delta_norm + 1e-12)
    return np.hstack(
        [
            speed[:, -5:],
            accel[:, -5:],
            speed[:, -1:] - speed[:, -2:-1],
            delta,
            np.abs(delta),
            delta_norm,
            champ_delta,
            rec_delta,
            champ_norm,
            rec_norm,
            rec_norm - champ_norm,
            dot,
            cos,
        ]
    ).astype(np.float32)


def make_gain_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=140,
        learning_rate=0.040,
        num_leaves=31,
        min_child_samples=28,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.03,
        reg_lambda=0.35,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def build_gain_proba_oof(
    coords: np.ndarray,
    y: np.ndarray,
    champion_oof: np.ndarray,
    recursive_oof: np.ndarray,
    folds: list[np.ndarray],
) -> np.ndarray:
    features = gain_features(coords, champion_oof, recursive_oof)
    champ_dist = np.linalg.norm(champion_oof - y, axis=1)
    rec_dist = np.linalg.norm(recursive_oof - y, axis=1)
    labels = (rec_dist < champ_dist).astype(np.int8)
    near_boundary = np.exp(-0.5 * ((np.minimum(champ_dist, rec_dist) - R_HIT) / 0.004) ** 2)
    weights = 1.0 + 4.0 * near_boundary + 2.0 * ((champ_dist > R_HIT) & (rec_dist <= R_HIT))
    weights = weights / np.mean(weights)

    proba = np.zeros(len(coords), dtype=np.float64)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(coords), dtype=bool)
        train_mask[val_idx] = False
        model = make_gain_classifier(726000 + fold_idx)
        model.fit(features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        proba[val_idx] = model.predict_proba(features[val_idx])[:, 1]
    return proba


def fit_gain_proba_test(
    coords: np.ndarray,
    y: np.ndarray,
    champion_oof: np.ndarray,
    recursive_oof: np.ndarray,
    test_coords: np.ndarray,
    champion_test: np.ndarray,
    recursive_test: np.ndarray,
) -> np.ndarray:
    train_x = gain_features(coords, champion_oof, recursive_oof)
    test_x = gain_features(test_coords, champion_test, recursive_test)
    champ_dist = np.linalg.norm(champion_oof - y, axis=1)
    rec_dist = np.linalg.norm(recursive_oof - y, axis=1)
    labels = (rec_dist < champ_dist).astype(np.int8)
    near_boundary = np.exp(-0.5 * ((np.minimum(champ_dist, rec_dist) - R_HIT) / 0.004) ** 2)
    weights = 1.0 + 4.0 * near_boundary + 2.0 * ((champ_dist > R_HIT) & (rec_dist <= R_HIT))
    weights = weights / np.mean(weights)
    probas = []
    for seed in [42, 777, 2026]:
        model = make_gain_classifier(826000 + seed)
        model.fit(train_x, labels, sample_weight=weights)
        probas.append(model.predict_proba(test_x)[:, 1])
    return np.mean(probas, axis=0)


def apply_candidate(champion: np.ndarray, recursive: np.ndarray, candidate: Candidate, gain_proba: np.ndarray | None = None) -> np.ndarray:
    delta = recursive - champion
    if candidate.kind == "global":
        strength = np.full(len(champion), candidate.weight, dtype=np.float64)
    elif candidate.kind == "gate":
        if gain_proba is None:
            raise ValueError("gate candidate requires gain_proba")
        cutoff = np.quantile(gain_proba, 1.0 - candidate.frac)
        strength = candidate.weight * (gain_proba >= cutoff).astype(np.float64)
    elif candidate.kind == "soft":
        if gain_proba is None:
            raise ValueError("soft candidate requires gain_proba")
        strength = candidate.weight * np.clip((gain_proba - candidate.threshold) / (1.0 - candidate.threshold + 1e-12), 0.0, 1.0)
    else:
        raise ValueError(f"unknown candidate kind: {candidate.kind}")
    return champion + strength[:, None] * delta


def candidate_name(rank: int, candidate: Candidate) -> str:
    base = f"recstep_rank{rank}_{candidate.kind}_{slug(candidate.spec)}_{mult_slug(candidate.mult)}"
    if candidate.kind == "global":
        return f"{base}_w{int(round(candidate.weight * 100)):02d}.csv"
    if candidate.kind == "gate":
        return f"{base}_top{int(round(candidate.frac * 1000)):03d}_b{int(round(candidate.weight * 100)):02d}.csv"
    return f"{base}_thr{int(round(candidate.threshold * 100)):02d}_b{int(round(candidate.weight * 100)):02d}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive one-step dynamics axis for 2026-05-26.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_recursive_onestep_dynamics_20260526.md")
    parser.add_argument("--top-k", type=int, default=6)
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

    champion_path = args.submission_dir / CHAMPION_SUBMISSION
    if not champion_path.exists():
        champion_path = args.submission_dir / FALLBACK_CHAMPION
    champion_test = read_submission_coords(champion_path)
    champion_cache = np.load(args.champion_oof)
    champion_oof = champion_cache["champion_oof"]
    champion_hit = distance_summary(champion_oof, y)["r_hit_1cm"]
    print(f"OOF champion hit={champion_hit:.6f}, public_best={PUBLIC_BEST_SCORE:.5f}", flush=True)

    folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED)
    rec_oof: dict[tuple[str, tuple[float, float, float]], np.ndarray] = {}
    raw_rows = []
    for spec in SPECS:
        spec_preds = build_oof_recursive(train_coords, spec, folds)
        for mult, pred in spec_preds.items():
            key = (spec.name, mult)
            rec_oof[key] = pred
            row = {
                "spec": spec.name,
                "mult": mult_slug(mult),
                **distance_summary(pred, y),
                **delta_summary(pred, champion_oof, "vs_champion"),
            }
            row["delta_hit_vs_champion"] = row["r_hit_1cm"] - champion_hit
            raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["delta_hit_vs_champion", "r_hit_1cm", "vs_champion_mean_delta"],
        ascending=[False, False, True],
    )
    print("Raw recursive OOF leaderboard", flush=True)
    print(raw_df.head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    candidate_rows = []
    candidates: list[Candidate] = []
    gain_proba_oof: dict[tuple[str, tuple[float, float, float]], np.ndarray] = {}
    for (spec_name, mult), pred in rec_oof.items():
        for weight in GLOBAL_BLEND_WEIGHTS:
            candidate = Candidate("global", spec_name, mult, weight)
            blended = apply_candidate(champion_oof, pred, candidate)
            row = {
                "kind": candidate.kind,
                "spec": spec_name,
                "mult": mult_slug(mult),
                "weight": weight,
                "frac": 0.0,
                "threshold": 0.0,
                **distance_summary(blended, y),
                **delta_summary(blended, champion_oof, "vs_champion"),
            }
            row["delta_hit_vs_champion"] = row["r_hit_1cm"] - champion_hit
            row["selection_score"] = row["delta_hit_vs_champion"] - 0.12 * max(0.0, row["vs_champion_mean_delta"] - 0.00035)
            candidate_rows.append(row)
            candidates.append(candidate)

    global_df = pd.DataFrame(candidate_rows).sort_values(
        ["selection_score", "delta_hit_vs_champion", "vs_champion_mean_delta"],
        ascending=[False, False, True],
    )
    top_pairs = []
    for _, row in global_df.head(3).iterrows():
        pair = (str(row["spec"]), next(mult for mult in MULTS if mult_slug(mult) == str(row["mult"])))
        if pair not in top_pairs:
            top_pairs.append(pair)

    gated_rows = []
    gated_candidates: list[Candidate] = []
    for pair in top_pairs:
        spec_name, mult = pair
        pred = rec_oof[pair]
        print(f"Training OOF gain selector for {spec_name} {mult_slug(mult)}", flush=True)
        proba = build_gain_proba_oof(train_coords, y, champion_oof, pred, folds)
        gain_proba_oof[pair] = proba
        for frac in GATE_FRACS:
            for weight in GATE_BLEND_WEIGHTS:
                candidate = Candidate("gate", spec_name, mult, weight, frac=frac)
                blended = apply_candidate(champion_oof, pred, candidate, proba)
                row = {
                    "kind": candidate.kind,
                    "spec": spec_name,
                    "mult": mult_slug(mult),
                    "weight": weight,
                    "frac": frac,
                    "threshold": 0.0,
                    **distance_summary(blended, y),
                    **delta_summary(blended, champion_oof, "vs_champion"),
                }
                row["delta_hit_vs_champion"] = row["r_hit_1cm"] - champion_hit
                row["selection_score"] = row["delta_hit_vs_champion"] - 0.10 * max(0.0, row["vs_champion_mean_delta"] - 0.00030)
                gated_rows.append(row)
                gated_candidates.append(candidate)
        for threshold in SOFT_THRESHOLDS:
            for weight in SOFT_BLEND_WEIGHTS:
                candidate = Candidate("soft", spec_name, mult, weight, threshold=threshold)
                blended = apply_candidate(champion_oof, pred, candidate, proba)
                row = {
                    "kind": candidate.kind,
                    "spec": spec_name,
                    "mult": mult_slug(mult),
                    "weight": weight,
                    "frac": 0.0,
                    "threshold": threshold,
                    **distance_summary(blended, y),
                    **delta_summary(blended, champion_oof, "vs_champion"),
                }
                row["delta_hit_vs_champion"] = row["r_hit_1cm"] - champion_hit
                row["selection_score"] = row["delta_hit_vs_champion"] - 0.10 * max(0.0, row["vs_champion_mean_delta"] - 0.00030)
                gated_rows.append(row)
                gated_candidates.append(candidate)

    all_rows = candidate_rows + gated_rows
    all_candidates = candidates + gated_candidates
    leaderboard = pd.DataFrame(all_rows)
    leaderboard["_candidate_idx"] = np.arange(len(leaderboard))
    leaderboard = leaderboard.sort_values(
        ["selection_score", "delta_hit_vs_champion", "vs_champion_mean_delta"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    print("Candidate OOF leaderboard", flush=True)
    print(leaderboard.head(30).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    full_cache: dict[str, dict[tuple[float, float, float], np.ndarray]] = {}
    test_gain_cache: dict[tuple[str, tuple[float, float, float]], np.ndarray] = {}
    used_signatures = set()
    rank = 1
    for _, row in leaderboard.iterrows():
        if rank > args.top_k:
            break
        candidate = all_candidates[int(row["_candidate_idx"])]
        signature = (candidate.kind, candidate.spec, candidate.mult, candidate.weight, candidate.frac, candidate.threshold)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        spec = next(spec for spec in SPECS if spec.name == candidate.spec)
        if spec.name not in full_cache:
            full_cache[spec.name] = full_recursive_predictions(train_coords, test_coords, spec)
        recursive_test = full_cache[spec.name][candidate.mult]
        pair = (candidate.spec, candidate.mult)
        gain_test = None
        if candidate.kind in {"gate", "soft"}:
            if pair not in test_gain_cache:
                test_gain_cache[pair] = fit_gain_proba_test(
                    train_coords,
                    y,
                    champion_oof,
                    rec_oof[pair],
                    test_coords,
                    champion_test,
                    recursive_test,
                )
            gain_test = test_gain_cache[pair]
        pred = apply_candidate(champion_test, recursive_test, candidate, gain_test)
        path = args.submission_dir / candidate_name(rank, candidate)
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "kind": candidate.kind,
                "spec": candidate.spec,
                "mult": mult_slug(candidate.mult),
                "weight": candidate.weight,
                "frac": candidate.frac,
                "threshold": candidate.threshold,
                "oof_delta_hit": float(row["delta_hit_vs_champion"]),
                "oof_hit": float(row["r_hit_1cm"]),
                "selection_score": float(row["selection_score"]),
                **delta_summary(pred, champion_test, "test_vs_champion"),
            }
        )
        rank += 1

    outputs_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-26 Recursive One-Step Dynamics",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{CHAMPION_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- oof_champion_hit: `{champion_hit:.6f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Stop micro-tuning the saturated t52 alpha/gate plateau.",
        "- Learn a 40ms one-step dynamics model from internal observed transitions.",
        "- Run the model twice at test time: current -> +40ms pseudo point -> +80ms prediction.",
        "- Submit only small champion-anchored global/gain-gated moves, because public feedback has punished large unanchored moves.",
        "",
        "## Raw Recursive OOF",
        "",
        dataframe_to_markdown(raw_df.head(20)),
        "",
        "## Candidate OOF",
        "",
        dataframe_to_markdown(leaderboard.drop(columns=["_candidate_idx"]).head(30)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(outputs_df),
        "",
        "## Recommended Public Order",
        "",
    ]
    for _, out_row in outputs_df.head(5).iterrows():
        report.append(f"{int(out_row['rank'])}. `{out_row['submission']}`")
    report.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- If rank1 is below 0.6910, stop this recursive one-step axis immediately.",
            "- If rank1 ties 0.6914, try rank2 only if it has a different gate/global profile.",
            "- If any candidate beats 0.6914, continue by expanding one-step cutoffs and gain-selector regimes.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
