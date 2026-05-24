from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
from run_curvature_gate_20260519 import curvature_correction, read_submission_coords, write_submission  # noqa: E402
from run_hit_weighted_local_frame import make_features, physics_poly_candidates  # noqa: E402
from run_local_target_manifold_projection_20260521 import build_champion_oof  # noqa: E402


CHAMPION = "curvgate_refine_rank2_gatet52a105.csv"
BACKUP_CHAMPION = "curvgate_rank4_gatet54a105.csv"
GATE_T50 = "curvgate_rank2_gatet50a105.csv"
TEMPORAL55 = "temporalbc_refine_r1f102s100u100_w55.csv"
SELECTOR_SOFT = "direct_selector_rank2_selectorsoft.csv"
CURRENT_SCORE = 0.69120
R_HIT = 0.01
CV_SEEDS = [42, 777, 2026]


@dataclass(frozen=True)
class PolicyConfig:
    n_clusters: int
    min_rows: int
    min_net: float
    penalty: float
    max_route_fraction: float


CONFIGS = [
    PolicyConfig(16, 120, 0.006, 1.25, 0.18),
    PolicyConfig(24, 100, 0.006, 1.25, 0.22),
    PolicyConfig(32, 80, 0.005, 1.35, 0.25),
    PolicyConfig(48, 70, 0.004, 1.50, 0.30),
    PolicyConfig(64, 55, 0.003, 1.65, 0.35),
    PolicyConfig(40, 80, 0.002, 1.20, 0.28),
]


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


def regime_features(coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    ddd = np.diff(dd, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    jerk = np.linalg.norm(ddd, axis=2)

    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    d_mean3 = diffs[:, -3:, :].mean(axis=1)
    speed_last = speed[:, -1]
    speed_prev = speed[:, -2]
    speed_mean3 = speed[:, -3:].mean(axis=1)
    speed_std5 = speed[:, -5:].std(axis=1)
    accel_last = accel[:, -1]
    accel_mean3 = accel[:, -3:].mean(axis=1)
    jerk_last = jerk[:, -1]

    dot = np.sum(d_last * d_prev, axis=1)
    denom = np.linalg.norm(d_last, axis=1) * np.linalg.norm(d_prev, axis=1) + 1e-12
    cos_turn = dot / denom
    turn_angle = np.arccos(np.clip(cos_turn, -1.0, 1.0))
    cross_norm = np.linalg.norm(np.cross(d_prev, d_last), axis=1)

    abs_last = np.abs(d_last)
    horiz = np.linalg.norm(d_last[:, :2], axis=1) + 1e-12
    vertical_ratio = d_last[:, 2] / horiz
    side_ratio = d_last[:, 1] / (np.abs(d_last[:, 0]) + 1e-12)
    fwd_ratio = d_last[:, 0] / (np.linalg.norm(d_last, axis=1) + 1e-12)

    return np.column_stack(
        [
            speed_last,
            speed_prev,
            speed_last - speed_prev,
            speed_mean3,
            speed_std5,
            accel_last,
            accel_mean3,
            jerk_last,
            turn_angle,
            cross_norm,
            abs_last,
            d_last,
            d_mean3,
            vertical_ratio,
            side_ratio,
            fwd_ratio,
        ]
    ).astype(np.float32)


def physics_candidate_names() -> list[str]:
    return [
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


def build_candidate_library(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    temporal_oof: np.ndarray,
    selector_oof: np.ndarray,
    anchor_oof: np.ndarray,
    champion_oof: np.ndarray,
    gate_proba: np.ndarray,
    train_correction: np.ndarray,
    test_correction: np.ndarray,
    submission_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    champion_test = read_submission_coords(submission_dir / CHAMPION)
    backup_test = read_submission_coords(submission_dir / BACKUP_CHAMPION)
    temporal_test = read_submission_coords(submission_dir / TEMPORAL55)
    selector_test = read_submission_coords(submission_dir / SELECTOR_SOFT)
    gate_t50_test = read_submission_coords(submission_dir / GATE_T50)

    gate_t54_oof = anchor_oof.copy()
    gate_t54_oof[gate_proba >= 0.54] = anchor_oof[gate_proba >= 0.54] + 0.105 * train_correction[gate_proba >= 0.54]
    gate_t50_oof = anchor_oof.copy()
    gate_t50_oof[gate_proba >= 0.50] = anchor_oof[gate_proba >= 0.50] + 0.105 * train_correction[gate_proba >= 0.50]

    names = [
        "champion",
        "gate_t54",
        "cochamp_t52_t54",
        "gate_t50",
        "temporal55",
        "selector_soft",
        "fixed_a060",
        "fixed_a090",
        "fixed_a120",
    ]
    train_preds = [
        champion_oof,
        gate_t54_oof,
        0.5 * champion_oof + 0.5 * gate_t54_oof,
        gate_t50_oof,
        temporal_oof,
        selector_oof,
        anchor_oof + 0.060 * train_correction,
        anchor_oof + 0.090 * train_correction,
        anchor_oof + 0.120 * train_correction,
    ]
    test_preds = [
        champion_test,
        backup_test,
        0.5 * champion_test + 0.5 * backup_test,
        gate_t50_test,
        temporal_test,
        selector_test,
        temporal_test + 0.060 * test_correction,
        temporal_test + 0.090 * test_correction,
        temporal_test + 0.120 * test_correction,
    ]

    train_phys = physics_poly_candidates(train_coords)
    test_phys = physics_poly_candidates(test_coords)
    for idx, name in enumerate(physics_candidate_names()):
        names.append(name)
        train_preds.append(train_phys[:, idx, :])
        test_preds.append(test_phys[:, idx, :])
    return names, np.stack(train_preds, axis=1), np.stack(test_preds, axis=1)


def fit_clusterer(features: np.ndarray, n_clusters: int, seed: int) -> tuple[StandardScaler, KMeans]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    km.fit(scaled)
    return scaler, km


def assign_clusters(features: np.ndarray, scaler: StandardScaler, km: KMeans) -> np.ndarray:
    return km.predict(scaler.transform(features))


def learn_policy(
    clusters: np.ndarray,
    candidate_dist: np.ndarray,
    config: PolicyConfig,
) -> dict[int, int]:
    champion_dist = candidate_dist[:, 0]
    champion_hit = champion_dist <= R_HIT
    policy: dict[int, int] = {}
    for cluster_id in np.unique(clusters):
        idx = np.where(clusters == cluster_id)[0]
        if len(idx) < config.min_rows:
            continue
        best_candidate = 0
        best_score = 0.0
        for cand_idx in range(1, candidate_dist.shape[1]):
            cand_hit = candidate_dist[idx, cand_idx] <= R_HIT
            rescue = np.mean((~champion_hit[idx]) & cand_hit)
            harm = np.mean(champion_hit[idx] & (~cand_hit))
            net = rescue - config.penalty * harm
            hit_delta = np.mean(cand_hit.astype(np.int8) - champion_hit[idx].astype(np.int8))
            if net > best_score and hit_delta >= config.min_net:
                best_score = net
                best_candidate = cand_idx
        if best_candidate != 0:
            policy[int(cluster_id)] = int(best_candidate)
    return policy


def apply_policy(base_pred: np.ndarray, candidate_preds: np.ndarray, clusters: np.ndarray, policy: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pred = base_pred.copy()
    chosen = np.zeros(len(base_pred), dtype=np.int32)
    for cluster_id, cand_idx in policy.items():
        mask = clusters == cluster_id
        pred[mask] = candidate_preds[mask, cand_idx, :]
        chosen[mask] = cand_idx
    return pred, chosen


def route_fraction_from_policy(clusters: np.ndarray, policy: dict[int, int]) -> float:
    if not policy:
        return 0.0
    return float(np.mean(np.isin(clusters, list(policy.keys()))))


def evaluate_policy_cv(
    regime_x: np.ndarray,
    candidate_preds: np.ndarray,
    y: np.ndarray,
    candidate_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_dist = np.linalg.norm(candidate_preds - y[:, None, :], axis=2)
    rows = []
    detail_rows = []
    baseline = candidate_preds[:, 0, :]
    for config_id, config in enumerate(CONFIGS, start=1):
        for seed in CV_SEEDS:
            val_mask = split_mask(len(y), 0.2, seed)
            train_mask = ~val_mask
            scaler, km = fit_clusterer(regime_x[train_mask], config.n_clusters, seed=610000 + seed + config_id)
            train_clusters = assign_clusters(regime_x[train_mask], scaler, km)
            val_clusters = assign_clusters(regime_x[val_mask], scaler, km)
            policy = learn_policy(train_clusters, candidate_dist[train_mask], config)

            route_frac = route_fraction_from_policy(val_clusters, policy)
            if route_frac > config.max_route_fraction and policy:
                # Keep only the largest training clusters until route size is in the intended range.
                cluster_counts = pd.Series(train_clusters).value_counts()
                ordered = sorted(policy.keys(), key=lambda cid: int(cluster_counts.get(cid, 0)), reverse=True)
                kept: dict[int, int] = {}
                for cid in ordered:
                    kept[cid] = policy[cid]
                    if route_fraction_from_policy(val_clusters, kept) >= config.max_route_fraction:
                        break
                policy = kept
                route_frac = route_fraction_from_policy(val_clusters, policy)

            pred, chosen = apply_policy(baseline[val_mask], candidate_preds[val_mask], val_clusters, policy)
            summary = distance_summary(pred, y[val_mask])
            base_summary = distance_summary(baseline[val_mask], y[val_mask])
            used = chosen[chosen > 0]
            top_candidate = candidate_names[int(pd.Series(used).mode().iloc[0])] if len(used) else "none"
            rows.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "n_clusters": config.n_clusters,
                    "min_rows": config.min_rows,
                    "min_net": config.min_net,
                    "penalty": config.penalty,
                    "max_route_fraction": config.max_route_fraction,
                    "policy_clusters": len(policy),
                    "route_fraction": route_frac,
                    "top_candidate": top_candidate,
                    "r_hit_1cm": summary["r_hit_1cm"],
                    "baseline_r_hit": base_summary["r_hit_1cm"],
                    "delta_hit": summary["r_hit_1cm"] - base_summary["r_hit_1cm"],
                    "mean_distance": summary["mean_distance"],
                }
            )
            for cid, cand_idx in policy.items():
                detail_rows.append(
                    {
                        "config_id": config_id,
                        "seed": seed,
                        "cluster": cid,
                        "candidate": candidate_names[cand_idx],
                    }
                )
    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby(["config_id", "n_clusters", "min_rows", "min_net", "penalty", "max_route_fraction"], as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            mean_baseline_r_hit=("baseline_r_hit", "mean"),
            mean_delta_hit=("delta_hit", "mean"),
            min_delta_hit=("delta_hit", "min"),
            mean_distance=("mean_distance", "mean"),
            mean_route_fraction=("route_fraction", "mean"),
            mean_policy_clusters=("policy_clusters", "mean"),
            top_candidate=("top_candidate", lambda x: x.mode().iloc[0] if len(x.mode()) else "none"),
        )
        .sort_values(["mean_delta_hit", "min_delta_hit", "mean_r_hit"], ascending=[False, False, False])
    )
    return leaderboard, pd.DataFrame(detail_rows)


def train_final_policy(
    regime_x: np.ndarray,
    candidate_preds: np.ndarray,
    y: np.ndarray,
    config: PolicyConfig,
    test_regime_x: np.ndarray,
) -> tuple[StandardScaler, KMeans, dict[int, int], np.ndarray]:
    candidate_dist = np.linalg.norm(candidate_preds - y[:, None, :], axis=2)
    scaler, km = fit_clusterer(regime_x, config.n_clusters, seed=710000 + config.n_clusters)
    clusters = assign_clusters(regime_x, scaler, km)
    test_clusters = assign_clusters(test_regime_x, scaler, km)
    policy = learn_policy(clusters, candidate_dist, config)
    if route_fraction_from_policy(test_clusters, policy) > config.max_route_fraction and policy:
        cluster_counts = pd.Series(clusters).value_counts()
        ordered = sorted(policy.keys(), key=lambda cid: int(cluster_counts.get(cid, 0)), reverse=True)
        kept: dict[int, int] = {}
        for cid in ordered:
            kept[cid] = policy[cid]
            if route_fraction_from_policy(test_clusters, kept) >= config.max_route_fraction:
                break
        policy = kept
    return scaler, km, policy, test_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime-wise champion miss policy exploration.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_regime_miss_policy_20260524.md")
    parser.add_argument("--top-k", type=int, default=3)
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

    print("Building champion OOF and candidate library", flush=True)
    train_features, _ = make_features(train_coords)
    cache = np.load(args.cache_path)
    temporal_oof = cache["temporal_oof"]
    selector_oof = cache["selector_oof"]
    anchor_oof = cache["anchor_oof"]
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)
    champion_oof, gate_oof_proba = build_champion_oof(train_coords, y, train_features, anchor_oof, train_correction)
    candidate_names, train_candidates, test_candidates = build_candidate_library(
        train_coords,
        test_coords,
        temporal_oof,
        selector_oof,
        anchor_oof,
        champion_oof,
        gate_oof_proba,
        train_correction,
        test_correction,
        args.submission_dir,
    )

    regime_x = regime_features(train_coords)
    test_regime_x = regime_features(test_coords)

    champion_diag = pd.DataFrame([{"name": "champion_oof_proxy", **distance_summary(champion_oof, y)}])
    oracle_rows = []
    champion_dist = np.linalg.norm(train_candidates[:, 0, :] - y, axis=1)
    for idx, name in enumerate(candidate_names):
        dist = np.linalg.norm(train_candidates[:, idx, :] - y, axis=1)
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

    print("Evaluating regime policies", flush=True)
    leaderboard, policy_details = evaluate_policy_cv(regime_x, train_candidates, y, candidate_names)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    champion_test = read_submission_coords(args.submission_dir / CHAMPION)
    written = []
    output_rows = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        config = CONFIGS[int(row["config_id"]) - 1]
        _, _, policy, test_clusters = train_final_policy(regime_x, train_candidates, y, config, test_regime_x)
        pred, chosen = apply_policy(champion_test, test_candidates, test_clusters, policy)
        path = args.submission_dir / (
            f"regimemiss_rank{rank}_c{config.n_clusters}_min{config.min_rows}_"
            f"net{slug(config.min_net)}_p{slug(config.penalty)}.csv"
        )
        write_submission(sample_submission, pred, path)
        written.append(path)
        used = chosen[chosen > 0]
        top_candidate = candidate_names[int(pd.Series(used).mode().iloc[0])] if len(used) else "none"
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "config_id": int(row["config_id"]),
                "cv_mean_delta_hit": float(row["mean_delta_hit"]),
                "cv_min_delta_hit": float(row["min_delta_hit"]),
                "cv_mean_route_fraction": float(row["mean_route_fraction"]),
                "test_route_fraction": float(np.mean(chosen > 0)),
                "policy_clusters": len(policy),
                "test_top_candidate": top_candidate,
                **delta_summary(pred, champion_test, "vs_champion"),
            }
        )
    output_df = pd.DataFrame(output_rows)

    report = [
        "# 2026-05-24 Regime Miss Policy",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- champion: `{CHAMPION} = {CURRENT_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Start over from miss-regime analysis instead of creating another global coordinate perturbation.",
        "- Cluster train/test by motion regime, then switch only clusters where train OOF says an alternate candidate has net hit gain.",
        "- CV policy selection is used to reduce pure in-sample oracle overfit.",
        "",
        "## Champion OOF Proxy",
        "",
        dataframe_to_markdown(champion_diag),
        "",
        "## Regime Policy CV",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(output_df),
        "",
        "## Candidate Oracle Diagnostics",
        "",
        dataframe_to_markdown(oracle_df.head(40)),
        "",
        "## Policy Details",
        "",
        dataframe_to_markdown(policy_details.head(120)),
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
