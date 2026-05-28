from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, stack_samples  # noqa: E402
from run_curvature_gate_20260519 import delta_summary, read_submission_coords, write_submission  # noqa: E402
from run_hit_weighted_local_frame import make_features, safe_scale  # noqa: E402
from run_local_frame_residual import local_basis, to_local  # noqa: E402
from run_recursive_onestep_dynamics_20260526 import (  # noqa: E402
    CHAMPION_SUBMISSION,
    FALLBACK_CHAMPION,
    OOF_FOLDS,
    OOF_SEED,
    Candidate,
    apply_candidate,
    build_gain_proba_oof,
    build_oof_recursive,
    fit_gain_proba_test,
    full_recursive_predictions,
    make_folds,
)


PUBLIC_WINNER = "recstepgate27_rank1_top090_b450_f100s100u100_top090_b450.csv"
PUBLIC_WINNER_SCORE = 0.69200
SPEC_NAME = "os_c89_b005_late"
WINNER_MULT = (1.00, 1.00, 1.00)
WINNER_FRAC = 0.090
WINNER_WEIGHT = 0.450
R_HIT = 0.01


@dataclass(frozen=True)
class DensityConfig:
    name: str
    k: int
    centers: int
    sigma: float
    top_frac: float
    shrink: float
    cap: float
    min_gain: float = 0.0


CONFIGS = [
    DensityConfig("k128_c48_s004_top03_b035_cap003", 128, 48, 0.004, 0.030, 0.35, 0.0030),
    DensityConfig("k160_c64_s005_top05_b030_cap003", 160, 64, 0.005, 0.050, 0.30, 0.0030),
    DensityConfig("k192_c80_s006_top08_b025_cap004", 192, 80, 0.006, 0.080, 0.25, 0.0040),
    DensityConfig("k096_c48_s004_top025_b045_cap004", 96, 48, 0.004, 0.025, 0.45, 0.0040),
    DensityConfig("k256_c96_s006_top06_b035_cap005", 256, 96, 0.006, 0.060, 0.35, 0.0050),
    DensityConfig("k224_c80_s005_top04_b040_cap004", 224, 80, 0.005, 0.040, 0.40, 0.0040),
    DensityConfig("k128_c64_s006_top10_b020_cap003", 128, 64, 0.006, 0.100, 0.20, 0.0030),
    DensityConfig("k096_c32_s0035_top02_b055_cap004", 96, 32, 0.0035, 0.020, 0.55, 0.0040),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def find_spec():
    from run_recursive_onestep_dynamics_20260526 import SPECS  # noqa: PLC0415

    return next(spec for spec in SPECS if spec.name == SPEC_NAME)


def output_name(rank: int, config: DensityConfig) -> str:
    return f"densmode28_rank{rank}_{slug(config.name)}.csv"


def norm_columns(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True)


def cap_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norms = norm_columns(vectors)
    return vectors * np.minimum(1.0, cap / (norms + 1e-12))


def make_density_features(
    coords: np.ndarray,
    winner_pred: np.ndarray,
    recursive_pred: np.ndarray,
    gain_proba: np.ndarray,
) -> np.ndarray:
    base_features, _ = make_features(coords)
    basis = local_basis(coords)
    scale = safe_scale(coords)
    last = coords[:, -1, :]
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)

    winner_local = to_local(winner_pred - last, basis) / scale
    rec_local = to_local(recursive_pred - last, basis) / scale
    rec_delta_local = to_local(recursive_pred - winner_pred, basis) / scale
    diffs_local = np.einsum("nwc,nkc->nwk", diffs, basis) / scale[:, None, :]
    dd_local = np.einsum("nwc,nkc->nwk", dd, basis) / scale[:, None, :]

    handcrafted = np.hstack(
        [
            winner_local,
            np.abs(winner_local),
            rec_local,
            rec_delta_local,
            np.abs(rec_delta_local),
            norm_columns(winner_pred - last) / scale,
            norm_columns(recursive_pred - winner_pred) / scale,
            diffs_local[:, -5:, :].reshape(len(coords), -1),
            dd_local[:, -5:, :].reshape(len(coords), -1),
            gain_proba[:, None],
        ]
    )
    handcrafted[~np.isfinite(handcrafted)] = 0.0
    return np.hstack([base_features, handcrafted]).astype(np.float32)


def weighted_density_mode(
    ref_features: np.ndarray,
    ref_residuals: np.ndarray,
    query_features: np.ndarray,
    config: DensityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    ref_x = scaler.fit_transform(ref_features)
    query_x = scaler.transform(query_features)
    nn = NearestNeighbors(n_neighbors=config.k, metric="euclidean", algorithm="auto")
    nn.fit(ref_x)
    feature_dist, indices = nn.kneighbors(query_x)

    corrections = np.zeros((len(query_features), 3), dtype=np.float64)
    gains = np.zeros(len(query_features), dtype=np.float64)
    best_scores = np.zeros(len(query_features), dtype=np.float64)

    for row_idx in range(len(query_features)):
        residuals = ref_residuals[indices[row_idx]]
        fd = feature_dist[row_idx]
        tau = np.median(fd) + 1e-8
        weights = np.exp(-fd / tau)
        weights = weights / (weights.sum() + 1e-12)

        centers = [np.zeros(3, dtype=np.float64)]
        centers.extend(residuals[: config.centers])
        for top_n in [8, 16, 32, 64]:
            if top_n <= len(residuals):
                centers.append(np.average(residuals[:top_n], axis=0, weights=weights[:top_n]))
        centers_arr = np.vstack(centers)

        dist = np.linalg.norm(residuals[None, :, :] - centers_arr[:, None, :], axis=2)
        kernel = np.exp(-0.5 * (dist / config.sigma) ** 2)
        scores = kernel @ weights
        zero_score = float(scores[0])
        best_idx = int(np.argmax(scores))
        corrections[row_idx] = centers_arr[best_idx]
        best_scores[row_idx] = float(scores[best_idx])
        gains[row_idx] = best_scores[row_idx] - zero_score

    return corrections, gains, best_scores


def oof_density_mode(
    features: np.ndarray,
    residuals: np.ndarray,
    folds: list[np.ndarray],
    config: DensityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrections = np.zeros_like(residuals)
    gains = np.zeros(len(features), dtype=np.float64)
    scores = np.zeros(len(features), dtype=np.float64)
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(features), dtype=bool)
        train_mask[val_idx] = False
        print(f"density mode OOF fold {fold_idx}/{len(folds)} config={config.name}", flush=True)
        corr, gain, score = weighted_density_mode(
            features[train_mask],
            residuals[train_mask],
            features[val_idx],
            config,
        )
        corrections[val_idx] = corr
        gains[val_idx] = gain
        scores[val_idx] = score
    return corrections, gains, scores


def apply_density_shift(
    winner_pred: np.ndarray,
    corrections: np.ndarray,
    gains: np.ndarray,
    config: DensityConfig,
) -> np.ndarray:
    cutoff = np.quantile(gains, 1.0 - config.top_frac)
    mask = (gains >= cutoff) & (gains > config.min_gain)
    soft = np.clip((gains - cutoff) / (gains.max() - cutoff + 1e-12), 0.0, 1.0)
    movement = config.shrink * soft[:, None] * cap_vectors(corrections, config.cap)
    pred = winner_pred.copy()
    pred[mask] = pred[mask] + movement[mask]
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual density mode optimizer around the 0.692 winner.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_residual_density_mode_20260528.md")
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

    base_champion_path = args.submission_dir / CHAMPION_SUBMISSION
    if not base_champion_path.exists():
        base_champion_path = args.submission_dir / FALLBACK_CHAMPION
    base_champion_test = read_submission_coords(base_champion_path)
    winner_test = read_submission_coords(args.submission_dir / PUBLIC_WINNER)
    base_champion_oof = np.load(args.champion_oof)["champion_oof"]

    spec = find_spec()
    folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED)
    print(f"Building recursive predictions for {SPEC_NAME}", flush=True)
    rec_oof = build_oof_recursive(train_coords, spec, folds)[WINNER_MULT]
    rec_test = full_recursive_predictions(train_coords, test_coords, spec)[WINNER_MULT]
    print("Building gain selectors for winner proxy", flush=True)
    gain_oof = build_gain_proba_oof(train_coords, y, base_champion_oof, rec_oof, folds)
    gain_test = fit_gain_proba_test(
        train_coords,
        y,
        base_champion_oof,
        rec_oof,
        test_coords,
        base_champion_test,
        rec_test,
    )
    winner_candidate = Candidate("gate", SPEC_NAME, WINNER_MULT, WINNER_WEIGHT, frac=WINNER_FRAC)
    winner_oof = apply_candidate(base_champion_oof, rec_oof, winner_candidate, gain_oof)
    winner_hit = distance_summary(winner_oof, y)["r_hit_1cm"]
    residuals = y - winner_oof

    print("Building density feature matrices", flush=True)
    train_features = make_density_features(train_coords, winner_oof, rec_oof, gain_oof)
    test_features = make_density_features(test_coords, winner_test, rec_test, gain_test)

    rows = []
    predictions = {}
    oof_folds = make_folds(len(train_coords), OOF_FOLDS, OOF_SEED + 281)
    for config in CONFIGS:
        corr_oof, gains_oof, scores_oof = oof_density_mode(train_features, residuals, oof_folds, config)
        pred_oof = apply_density_shift(winner_oof, corr_oof, gains_oof, config)

        print(f"density mode full config={config.name}", flush=True)
        corr_test, gains_test, scores_test = weighted_density_mode(train_features, residuals, test_features, config)
        pred_test = apply_density_shift(winner_test, corr_test, gains_test, config)

        metrics = distance_summary(pred_oof, y)
        row = {
            "config": config.name,
            "k": config.k,
            "centers": config.centers,
            "sigma": config.sigma,
            "top_frac": config.top_frac,
            "shrink": config.shrink,
            "cap": config.cap,
            "oof_hit": metrics["r_hit_1cm"],
            "oof_delta_vs_winner": metrics["r_hit_1cm"] - winner_hit,
            "oof_gain_p95": float(np.quantile(gains_oof, 0.95)),
            "test_gain_p95": float(np.quantile(gains_test, 0.95)),
            "test_score_p95": float(np.quantile(scores_test, 0.95)),
            **delta_summary(pred_oof, winner_oof, "oof_vs_winner"),
            **delta_summary(pred_test, winner_test, "test_vs_winner"),
        }
        row["selection_score"] = (
            row["oof_delta_vs_winner"]
            - 0.07 * max(0.0, row["test_vs_winner_mean_delta"] - 0.00011)
            - 0.025 * max(0.0, row["test_vs_winner_p95_delta"] - 0.0010)
        )
        rows.append(row)
        predictions[config.name] = pred_test

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "oof_delta_vs_winner", "test_vs_winner_mean_delta"],
        ascending=[False, False, True],
    )
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        config = next(config for config in CONFIGS if config.name == row["config"])
        path = args.submission_dir / output_name(rank, config)
        write_submission(sample_submission, predictions[config.name], path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "config": config.name,
                "oof_delta_vs_winner": float(row["oof_delta_vs_winner"]),
                "selection_score": float(row["selection_score"]),
                "test_vs_winner_mean_delta": float(row["test_vs_winner_mean_delta"]),
                "test_vs_winner_p95_delta": float(row["test_vs_winner_p95_delta"]),
                "test_vs_winner_max_delta": float(row["test_vs_winner_max_delta"]),
            }
        )

    outputs_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-28 Residual Density Mode",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{PUBLIC_WINNER} = {PUBLIC_WINNER_SCORE:.5f}`",
        f"- oof_winner_hit_proxy: `{winner_hit:.6f}`",
        "",
        "## Idea",
        "",
        "- Treat the 0.692 recursive winner as the origin.",
        "- Retrieve similar train trajectories in feature space.",
        "- Estimate the residual density mode that maximizes local probability mass inside the 1cm hit ball.",
        "- Move only the highest density-gain test samples by a capped fraction of that residual mode.",
        "",
        "## Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(outputs_df),
        "",
        "## Recommended Public Order",
        "",
    ]
    for _, out_row in outputs_df.head(4).iterrows():
        report.append(f"{int(out_row['rank'])}. `{out_row['submission']}`")
    report.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- If rank1 is below 0.6920, density-mode retrieval is still not public-stable.",
            "- If rank1 ties but rank2 improves movement diversity, try rank2 only if submissions remain.",
            "- If any candidate improves, refine sigma/K and residual-mode candidate centers instead of returning to recursive gate tuning.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
