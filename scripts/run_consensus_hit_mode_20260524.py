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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import curvature_correction, read_submission_coords, write_submission  # noqa: E402
from run_hit_weighted_local_frame import make_features  # noqa: E402
from run_local_target_manifold_projection_20260521 import build_champion_oof  # noqa: E402
from run_regime_miss_policy_20260524 import (  # noqa: E402
    BACKUP_CHAMPION,
    CHAMPION,
    CURRENT_SCORE,
    R_HIT,
    TEMPORAL55,
    build_candidate_library,
    delta_summary,
)


@dataclass(frozen=True)
class ConsensusConfig:
    sigma: float
    power: float
    min_margin: float
    max_route_fraction: float
    blend: float
    use_top_k: int


CONFIGS = [
    ConsensusConfig(0.0040, 3.0, 0.010, 0.08, 1.00, 5),
    ConsensusConfig(0.0045, 3.0, 0.008, 0.12, 1.00, 5),
    ConsensusConfig(0.0050, 2.5, 0.006, 0.16, 1.00, 7),
    ConsensusConfig(0.0060, 2.0, 0.004, 0.22, 0.80, 7),
    ConsensusConfig(0.0075, 2.0, 0.002, 0.30, 0.65, 9),
    ConsensusConfig(0.0100, 1.5, 0.000, 0.40, 0.50, 11),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def pairwise_distances(preds: np.ndarray) -> np.ndarray:
    diff = preds[:, :, None, :] - preds[:, None, :, :]
    return np.linalg.norm(diff, axis=3)


def make_candidate_weights(train_preds: np.ndarray, y: np.ndarray, names: list[str], power: float) -> np.ndarray:
    distances = np.linalg.norm(train_preds - y[:, None, :], axis=2)
    hit_rates = np.mean(distances <= R_HIT, axis=0)
    # Keep all candidates alive, but make proven near-best anchors dominate the consensus.
    weights = np.clip(hit_rates, 0.05, None) ** power
    weights = weights / np.sum(weights)
    return weights.astype(np.float64)


def consensus_select(preds: np.ndarray, weights: np.ndarray, config: ConsensusConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pdist = pairwise_distances(preds)
    kernel = np.exp(-0.5 * (pdist / config.sigma) ** 2)
    top_k = min(config.use_top_k, preds.shape[1])
    top_idx = np.argsort(weights)[::-1][:top_k]
    score = kernel[:, :, top_idx] @ weights[top_idx]
    score = score / np.sum(weights[top_idx])
    selected = np.argmax(score, axis=1)
    champion_score = score[:, 0]
    margin = score[np.arange(len(preds)), selected] - champion_score
    return selected, margin, score


def capped_route_mask(margin: np.ndarray, selected: np.ndarray, config: ConsensusConfig) -> np.ndarray:
    eligible = (selected != 0) & (margin >= config.min_margin)
    max_n = int(round(len(margin) * config.max_route_fraction))
    if max_n <= 0 or not np.any(eligible):
        return np.zeros(len(margin), dtype=bool)
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) <= max_n:
        return eligible
    order = eligible_idx[np.argsort(margin[eligible_idx])[::-1]]
    mask = np.zeros(len(margin), dtype=bool)
    mask[order[:max_n]] = True
    return mask


def apply_consensus(
    preds: np.ndarray,
    selected: np.ndarray,
    margin: np.ndarray,
    config: ConsensusConfig,
) -> tuple[np.ndarray, np.ndarray]:
    out = preds[:, 0, :].copy()
    mask = capped_route_mask(margin, selected, config)
    chosen = preds[np.arange(len(preds)), selected]
    out[mask] = (1.0 - config.blend) * preds[:, 0, :][mask] + config.blend * chosen[mask]
    return out, mask


def candidate_diagnostics(names: list[str], train_preds: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    champion_dist = np.linalg.norm(train_preds[:, 0, :] - y, axis=1)
    champion_hit = champion_dist <= R_HIT
    rows = []
    for idx, name in enumerate(names):
        dist = np.linalg.norm(train_preds[:, idx, :] - y, axis=1)
        hit = dist <= R_HIT
        rows.append(
            {
                "candidate": name,
                "hit_rate": float(np.mean(hit)),
                "rescue_rate_vs_champion": float(np.mean((~champion_hit) & hit)),
                "harm_rate_vs_champion": float(np.mean(champion_hit & (~hit))),
                "mean_distance": float(np.mean(dist)),
            }
        )
    return pd.DataFrame(rows).sort_values(["hit_rate", "rescue_rate_vs_champion"], ascending=[False, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate consensus mode for hit-oriented coordinate selection.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_consensus_hit_mode_20260524.md")
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

    print("Building OOF candidate library", flush=True)
    train_features, _ = make_features(train_coords)
    cache = np.load(args.cache_path)
    temporal_oof = cache["temporal_oof"]
    selector_oof = cache["selector_oof"]
    anchor_oof = cache["anchor_oof"]
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)
    champion_oof, gate_proba = build_champion_oof(train_coords, y, train_features, anchor_oof, train_correction)
    names, train_preds, test_preds = build_candidate_library(
        train_coords,
        test_coords,
        temporal_oof,
        selector_oof,
        anchor_oof,
        champion_oof,
        gate_proba,
        train_correction,
        test_correction,
        args.submission_dir,
    )

    champion_test = read_submission_coords(args.submission_dir / CHAMPION)
    backup_test = read_submission_coords(args.submission_dir / BACKUP_CHAMPION)
    temporal_test = read_submission_coords(args.submission_dir / TEMPORAL55)
    # The imported library builds fixed-alpha test variants from temporal55. Keep champion column exact.
    test_preds[:, 0, :] = champion_test
    test_preds[:, 1, :] = backup_test
    test_preds[:, 4, :] = temporal_test

    champion_diag = pd.DataFrame([{"name": "champion_oof_proxy", **distance_summary(champion_oof, y)}])
    cand_diag = candidate_diagnostics(names, train_preds, y)

    print("Evaluating consensus configs", flush=True)
    rows = []
    train_outputs: list[tuple[ConsensusConfig, np.ndarray, np.ndarray, np.ndarray]] = []
    for config_id, config in enumerate(CONFIGS, start=1):
        weights = make_candidate_weights(train_preds, y, names, config.power)
        selected, margin, _ = consensus_select(train_preds, weights, config)
        pred, mask = apply_consensus(train_preds, selected, margin, config)
        routed_names, routed_counts = np.unique(np.array(names, dtype=object)[selected[mask]], return_counts=True)
        top_route = (
            ",".join(f"{name}:{count}" for name, count in sorted(zip(routed_names, routed_counts), key=lambda item: -item[1])[:3])
            if len(routed_names)
            else "none"
        )
        row = {
            "config_id": config_id,
            "sigma": config.sigma,
            "power": config.power,
            "min_margin": config.min_margin,
            "max_route_fraction": config.max_route_fraction,
            "blend": config.blend,
            "use_top_k": config.use_top_k,
            "route_fraction": float(np.mean(mask)),
            "top_route": top_route,
            **distance_summary(pred, y),
            **delta_summary(pred, train_preds[:, 0, :], "vs_champion"),
        }
        row["delta_hit_vs_champion"] = row["r_hit_1cm"] - distance_summary(train_preds[:, 0, :], y)["r_hit_1cm"]
        rows.append(row)
        train_outputs.append((config, selected, margin, weights))

    leaderboard = pd.DataFrame(rows).sort_values(
        ["delta_hit_vs_champion", "r_hit_1cm", "vs_champion_mean_delta"],
        ascending=[False, False, True],
    )
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    used_configs: set[int] = set()
    for _, row in leaderboard.iterrows():
        config_id = int(row["config_id"])
        if config_id in used_configs:
            continue
        used_configs.add(config_id)
        config, _, _, weights = train_outputs[config_id - 1]
        selected_test, margin_test, _ = consensus_select(test_preds, weights, config)
        pred_test, mask_test = apply_consensus(test_preds, selected_test, margin_test, config)

        rank = len(written) + 1
        path = args.submission_dir / (
            f"consensusmode_rank{rank}_s{slug(config.sigma)}"
            f"p{slug(config.power)}m{slug(config.min_margin)}b{int(round(config.blend * 100)):03d}.csv"
        )
        write_submission(sample_submission, pred_test, path)
        written.append(path)
        routed_names, routed_counts = np.unique(np.array(names, dtype=object)[selected_test[mask_test]], return_counts=True)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "config_id": config_id,
                "cv_like_delta_hit": float(row["delta_hit_vs_champion"]),
                "train_route_fraction": float(row["route_fraction"]),
                "test_route_fraction": float(np.mean(mask_test)),
                "test_top_route": (
                    ",".join(
                        f"{name}:{count}"
                        for name, count in sorted(zip(routed_names, routed_counts), key=lambda item: -item[1])[:3]
                    )
                    if len(routed_names)
                    else "none"
                ),
                **delta_summary(pred_test, champion_test, "vs_champion"),
            }
        )
        if len(written) >= args.top_k:
            break

    output_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-24 Consensus Hit Mode",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- champion: `{CHAMPION} = {CURRENT_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Idea",
        "",
        "- Start over with an unsupervised hit-mode view: candidate predictions are treated as a small coordinate cloud.",
        "- Instead of predicting the target directly, choose a coordinate that sits in the densest weighted local consensus around strong candidates.",
        "- This is deliberately different from residual fitting and post-hoc curvature thresholds; it tests whether candidate agreement is a usable proxy for 1cm hit probability.",
        "",
        "## Champion OOF Proxy",
        "",
        dataframe_to_markdown(champion_diag),
        "",
        "## Consensus Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(output_df),
        "",
        "## Candidate Diagnostics",
        "",
        dataframe_to_markdown(cand_diag.head(40)),
        "",
        "## Notes",
        "",
        "- This is a high-variance public probe because config ranking still uses train OOF diagnostics.",
        "- If the top route is mostly near-champion variants, submit risk is moderate; if it routes to low-hit physics/poly candidates, treat it as exploratory only.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
