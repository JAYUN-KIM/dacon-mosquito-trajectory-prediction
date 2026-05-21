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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, physics_prediction, split_mask, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import (  # noqa: E402
    boundary_weights,
    curvature_correction,
    make_gate_classifier,
    make_gate_features,
    predict_gate_proba,
    read_submission_coords,
    write_submission,
)
from run_feature_rich_residual import candidate_block  # noqa: E402
from run_hit_weighted_local_frame import make_features, physics_poly_candidates  # noqa: E402


CURRENT_BEST = "curvgate_refine_rank2_gatet52a105.csv"
CURRENT_BEST_SCORE = 0.69120
TEMPORAL_ANCHOR = "temporalbc_refine_r1f102s100u100_w55.csv"
SELECTOR_ANCHOR = "direct_selector_rank2_selectorsoft.csv"
GATE_T50 = "curvgate_rank2_gatet50a105.csv"
GATE_T54 = "curvgate_rank4_gatet54a105.csv"
R_HIT = 0.01

CV_SEEDS = [42, 777, 2026]
FULL_SEEDS = [1301, 2718, 3141]
TOP_FRACTIONS = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.075]


@dataclass(frozen=True)
class Candidate:
    name: str
    train_pred: np.ndarray
    test_pred: np.ndarray


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


def build_gate_oof_proba(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    proba = np.zeros(len(labels), dtype=np.float64)
    for seed in CV_SEEDS:
        fold_proba = np.zeros(len(labels), dtype=np.float64)
        val_mask = split_mask(len(labels), 0.2, seed)
        train_mask = ~val_mask

        model = make_gate_classifier(910000 + seed)
        model.fit(features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        fold_proba[val_mask] = predict_gate_proba(model, features[val_mask])

        # Keep every row out-of-sample for this seed by fitting the complementary split.
        model2 = make_gate_classifier(911000 + seed)
        model2.fit(features[val_mask], labels[val_mask], sample_weight=weights[val_mask])
        fold_proba[train_mask] = predict_gate_proba(model2, features[train_mask])
        proba += fold_proba / len(CV_SEEDS)
    return proba


def build_champion_proxy(
    coords: np.ndarray,
    y: np.ndarray,
    base_features: np.ndarray,
    anchor_oof: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_oof = anchor_oof + 0.090 * correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    labels = (fixed_dist < anchor_dist).astype(np.int8)
    weights = boundary_weights(anchor_dist, fixed_dist)
    gate_features = make_gate_features(coords, base_features, anchor_oof, correction)
    print("Building OOF gate probabilities for champion proxy", flush=True)
    gate_proba = build_gate_oof_proba(gate_features, labels, weights)
    champion = anchor_oof.copy()
    mask = gate_proba >= 0.52
    champion[mask] = anchor_oof[mask] + 0.105 * correction[mask]
    return champion, gate_proba


def candidate_names_for_block() -> list[str]:
    _, names = candidate_block(np.zeros((1, 11, 3), dtype=np.float64))
    return names


def build_candidates(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    selector_oof: np.ndarray,
    temporal_oof: np.ndarray,
    anchor_oof: np.ndarray,
    train_correction: np.ndarray,
    test_correction: np.ndarray,
    gate_oof_proba: np.ndarray,
    submission_dir: Path,
) -> list[Candidate]:
    _, physics_names = candidate_block(train_coords[:1])
    train_phys = physics_poly_candidates(train_coords)
    test_phys = physics_poly_candidates(test_coords)

    candidates: list[Candidate] = []
    for idx, name in enumerate(physics_names):
        candidates.append(Candidate(name, train_phys[:, idx, :], test_phys[:, idx, :]))

    temporal_test = read_submission_coords(submission_dir / TEMPORAL_ANCHOR)
    selector_test = read_submission_coords(submission_dir / SELECTOR_ANCHOR)
    gate_t50_test = read_submission_coords(submission_dir / GATE_T50)
    gate_t54_test = read_submission_coords(submission_dir / GATE_T54)

    gate_t50_oof = anchor_oof.copy()
    gate_t50_oof[gate_oof_proba >= 0.50] = anchor_oof[gate_oof_proba >= 0.50] + 0.105 * train_correction[gate_oof_proba >= 0.50]
    gate_t54_oof = anchor_oof.copy()
    gate_t54_oof[gate_oof_proba >= 0.54] = anchor_oof[gate_oof_proba >= 0.54] + 0.105 * train_correction[gate_oof_proba >= 0.54]

    extra = [
        Candidate("selector_soft", selector_oof, selector_test),
        Candidate("temporal55", temporal_oof, temporal_test),
        Candidate("fixed_a060", anchor_oof + 0.060 * train_correction, temporal_test + 0.060 * test_correction),
        Candidate("fixed_a090", anchor_oof + 0.090 * train_correction, temporal_test + 0.090 * test_correction),
        Candidate("fixed_a120", anchor_oof + 0.120 * train_correction, temporal_test + 0.120 * test_correction),
        Candidate("gate_t50_a105", gate_t50_oof, gate_t50_test),
        Candidate("gate_t54_a105", gate_t54_oof, gate_t54_test),
    ]
    candidates.extend(extra)
    return candidates


def common_route_features(coords: np.ndarray, base_features: np.ndarray, champion: np.ndarray, gate_proba: np.ndarray | None) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(np.diff(diffs, axis=1), axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    last = coords[:, -1, :]
    champion_delta = champion - last
    champion_norm = np.linalg.norm(champion_delta, axis=1, keepdims=True)
    last_norm = np.linalg.norm(d_last, axis=1, keepdims=True)
    dot = np.sum(d_last * d_prev, axis=1, keepdims=True)
    cos_turn = dot / (np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-12)

    compact = np.hstack(
        [
            d_last,
            d_prev,
            d_last - d_prev,
            np.abs(d_last),
            speed[:, -5:],
            accel[:, -4:],
            speed[:, -1:] - speed[:, -2:-1],
            cos_turn,
            champion_delta,
            np.abs(champion_delta),
            champion_norm,
            champion_norm / (2.0 * last_norm + 1e-12),
            base_features,
        ]
    )
    if gate_proba is not None:
        compact = np.hstack([compact, gate_proba[:, None]])
    return compact.astype(np.float32)


def candidate_route_features(common: np.ndarray, coords: np.ndarray, champion: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    last = coords[:, -1, :]
    d_last = coords[:, -1, :] - coords[:, -2, :]
    champion_delta = champion - last
    candidate_delta = candidate - last
    route_delta = candidate - champion
    route_norm = np.linalg.norm(route_delta, axis=1, keepdims=True)
    cand_norm = np.linalg.norm(candidate_delta, axis=1, keepdims=True)
    champ_norm = np.linalg.norm(champion_delta, axis=1, keepdims=True)
    last_norm = np.linalg.norm(d_last, axis=1, keepdims=True)
    cos_cand_last = np.sum(candidate_delta * d_last, axis=1, keepdims=True) / (cand_norm * last_norm + 1e-12)
    cos_route_champ = np.sum(route_delta * champion_delta, axis=1, keepdims=True) / (route_norm * champ_norm + 1e-12)
    specific = np.hstack(
        [
            candidate_delta,
            route_delta,
            np.abs(route_delta),
            route_norm,
            cand_norm,
            cand_norm / (champ_norm + 1e-12),
            route_norm / (champ_norm + 1e-12),
            cos_cand_last,
            cos_route_champ,
        ]
    )
    return np.hstack([common, specific]).astype(np.float32)


def make_rescue_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=520,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=36,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.18,
        reg_lambda=1.40,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def rescue_labels_and_weights(champion: np.ndarray, candidate: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_dist = np.linalg.norm(champion - y, axis=1)
    cand_dist = np.linalg.norm(candidate - y, axis=1)
    base_hit = base_dist <= R_HIT
    cand_hit = cand_dist <= R_HIT
    rescue = cand_hit & (~base_hit)
    strong_gain = (cand_dist + 0.0015 < base_dist) & (base_dist <= 0.018)
    labels = (rescue | strong_gain).astype(np.int8)

    boundary = np.exp(-0.5 * ((np.minimum(base_dist, cand_dist) - R_HIT) / 0.003) ** 2)
    harmful = base_hit & (~cand_hit)
    weights = 1.0 + 8.0 * boundary + 14.0 * rescue + 8.0 * harmful
    weights = np.clip(weights, 1.0, 28.0)
    weights = weights / np.mean(weights)
    return labels, weights


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    n_route = max(1, int(round(len(scores) * fraction)))
    threshold = np.partition(scores, len(scores) - n_route)[len(scores) - n_route]
    return scores >= threshold


def evaluate_candidates(
    train_common: np.ndarray,
    train_coords: np.ndarray,
    champion: np.ndarray,
    candidates: list[Candidate],
    y: np.ndarray,
) -> pd.DataFrame:
    rows = []
    baseline_by_seed: dict[int, float] = {}
    for seed in CV_SEEDS:
        val_mask = split_mask(len(y), 0.2, seed)
        baseline_by_seed[seed] = distance_summary(champion[val_mask], y[val_mask])["r_hit_1cm"]

    for cand_idx, candidate in enumerate(candidates, start=1):
        labels, weights = rescue_labels_and_weights(champion, candidate.train_pred, y)
        positive_rate = float(labels.mean())
        oracle_rescue = float(np.mean((np.linalg.norm(champion - y, axis=1) > R_HIT) & (np.linalg.norm(candidate.train_pred - y, axis=1) <= R_HIT)))
        oracle_harm = float(np.mean((np.linalg.norm(champion - y, axis=1) <= R_HIT) & (np.linalg.norm(candidate.train_pred - y, axis=1) > R_HIT)))

        if labels.sum() < 20:
            continue

        print(
            f"candidate {cand_idx}/{len(candidates)} {candidate.name}: "
            f"positive={positive_rate:.4f} oracle_rescue={oracle_rescue:.4f} oracle_harm={oracle_harm:.4f}",
            flush=True,
        )
        route_features = candidate_route_features(train_common, train_coords, champion, candidate.train_pred)
        for seed in CV_SEEDS:
            val_mask = split_mask(len(y), 0.2, seed)
            train_mask = ~val_mask
            model = make_rescue_classifier(930000 + seed + cand_idx * 101)
            model.fit(route_features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
            proba = predict_gate_proba(model, route_features[val_mask])

            for fraction in TOP_FRACTIONS:
                mask = top_fraction_mask(proba, fraction)
                pred = champion[val_mask].copy()
                pred[mask] = candidate.train_pred[val_mask][mask]
                row = {
                    "candidate": candidate.name,
                    "fraction": fraction,
                    "seed": seed,
                    "positive_rate": positive_rate,
                    "oracle_rescue_rate": oracle_rescue,
                    "oracle_harm_rate": oracle_harm,
                    "route_fraction": float(np.mean(mask)),
                    **distance_summary(pred, y[val_mask]),
                    **delta_summary(pred, champion[val_mask], "vs_champion"),
                }
                row["baseline_r_hit"] = baseline_by_seed[seed]
                row["delta_hit_vs_baseline"] = row["r_hit_1cm"] - baseline_by_seed[seed]
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = (
        df.groupby(["candidate", "fraction"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_delta_hit_vs_baseline=("delta_hit_vs_baseline", "mean"),
            min_delta_hit_vs_baseline=("delta_hit_vs_baseline", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
            positive_rate=("positive_rate", "mean"),
            oracle_rescue_rate=("oracle_rescue_rate", "mean"),
            oracle_harm_rate=("oracle_harm_rate", "mean"),
            route_fraction=("route_fraction", "mean"),
            vs_champion_mean_delta=("vs_champion_mean_delta", "mean"),
            vs_champion_p95_delta=("vs_champion_p95_delta", "mean"),
            vs_champion_max_delta=("vs_champion_max_delta", "mean"),
        )
        .sort_values(
            ["mean_delta_hit_vs_baseline", "min_delta_hit_vs_baseline", "mean_r_hit", "vs_champion_mean_delta"],
            ascending=[False, False, False, True],
        )
    )
    summary["risk_adjusted_delta"] = summary["mean_delta_hit_vs_baseline"] - 0.25 * summary["std_r_hit"].fillna(0.0)
    return summary


def fit_full_rescue_scores(
    train_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    test_features: np.ndarray,
    candidate_idx: int,
) -> np.ndarray:
    preds = []
    for seed in FULL_SEEDS:
        model = make_rescue_classifier(940000 + seed + candidate_idx * 101)
        model.fit(train_features, labels, sample_weight=weights)
        preds.append(predict_gate_proba(model, test_features))
    return np.mean(preds, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hit-rescue specialist: hard swap only likely champion misses.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_hit_rescue_specialist_20260521.md")
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

    print("Building feature matrices and champion proxy", flush=True)
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    cache = np.load(args.cache_path)
    temporal_oof = cache["temporal_oof"]
    selector_oof = cache["selector_oof"]
    anchor_oof = cache["anchor_oof"]
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)
    champion_oof, gate_oof_proba = build_champion_proxy(train_coords, y, train_features, anchor_oof, train_correction)
    current_best = read_submission_coords(args.submission_dir / CURRENT_BEST)
    champion_diag = pd.DataFrame([{"name": "champion_oof_proxy", **distance_summary(champion_oof, y)}])
    print(champion_diag.to_string(index=False), flush=True)

    print("Building candidate library", flush=True)
    candidates = build_candidates(
        train_coords,
        test_coords,
        selector_oof,
        temporal_oof,
        anchor_oof,
        train_correction,
        test_correction,
        gate_oof_proba,
        args.submission_dir,
    )
    train_gate_signal = (gate_oof_proba >= 0.52).astype(np.float64)
    temporal_test_for_signal = read_submission_coords(args.submission_dir / TEMPORAL_ANCHOR)
    test_gate_signal = (np.linalg.norm(current_best - temporal_test_for_signal, axis=1) > 1e-12).astype(np.float64)
    train_common = common_route_features(train_coords, train_features, champion_oof, train_gate_signal)
    test_common = common_route_features(test_coords, test_features, current_best, test_gate_signal)

    print("Evaluating rescue specialists", flush=True)
    leaderboard = evaluate_candidates(train_common, train_coords, champion_oof, candidates, y)
    if leaderboard.empty:
        raise RuntimeError("No rescue candidates produced enough positive labels.")
    print(leaderboard.head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    candidate_lookup = {candidate.name: (idx, candidate) for idx, candidate in enumerate(candidates, start=1)}
    written = []
    output_rows = []
    used: set[tuple[str, float]] = set()
    for _, row in leaderboard.iterrows():
        key = (str(row["candidate"]), float(row["fraction"]))
        if key in used:
            continue
        used.add(key)
        candidate_idx, candidate = candidate_lookup[str(row["candidate"])]
        labels, weights = rescue_labels_and_weights(champion_oof, candidate.train_pred, y)
        train_route_features = candidate_route_features(train_common, train_coords, champion_oof, candidate.train_pred)
        test_route_features = candidate_route_features(test_common, test_coords, current_best, candidate.test_pred)
        scores = fit_full_rescue_scores(train_route_features, labels, weights, test_route_features, candidate_idx)
        mask = top_fraction_mask(scores, float(row["fraction"]))
        pred = current_best.copy()
        pred[mask] = candidate.test_pred[mask]

        rank = len(written) + 1
        path = args.submission_dir / f"hitrescue_rank{rank}_{slug(candidate.name)}_top{int(round(float(row['fraction']) * 1000)):03d}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "candidate": candidate.name,
                "fraction": float(row["fraction"]),
                "cv_mean_r_hit": float(row["mean_r_hit"]),
                "cv_delta_vs_baseline": float(row["mean_delta_hit_vs_baseline"]),
                "cv_min_delta_vs_baseline": float(row["min_delta_hit_vs_baseline"]),
                "test_route_fraction": float(np.mean(mask)),
                "test_score_p50": float(np.quantile(scores, 0.50)),
                "test_score_p95": float(np.quantile(scores, 0.95)),
                "test_score_min_routed": float(np.min(scores[mask])),
                **delta_summary(pred, current_best, "vs_current_best"),
            }
        )
        if len(written) >= args.top_k:
            break

    output_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-21 Hit-Rescue Specialist",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- current_best: `{CURRENT_BEST} = {CURRENT_BEST_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- The manifold projection probe dropped to public 0.68980, so broad nudging around the champion is unsafe.",
        "- This experiment uses a different axis: keep the champion unchanged almost everywhere and hard-swap only the top-scored likely misses.",
        "- Each specialist is trained on train OOF outcomes where one candidate hits or strongly improves while the champion proxy misses.",
        "",
        "## Champion OOF Proxy",
        "",
        dataframe_to_markdown(champion_diag),
        "",
        "## Rescue CV Leaderboard",
        "",
        dataframe_to_markdown(leaderboard.head(80)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(output_df),
        "",
        "## Notes",
        "",
        "- This remains a public-risky probe because the target event is rare and classifier precision matters more than average distance.",
        "- Prefer the lowest route fraction first unless CV delta is much stronger at a higher fraction.",
        "- If public drops again, the useful lesson is that current champion errors are not separable by train OOF features, and the next axis should be new pseudo-label construction rather than post-processing.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
