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
from run_curvature_gate_20260519 import delta_summary, read_submission_coords, write_submission  # noqa: E402
from run_hit_weighted_local_frame import safe_scale  # noqa: E402
from run_local_frame_residual import local_basis, to_global, to_local  # noqa: E402
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
class BiasCandidate:
    name: str
    family: str
    shrink: float
    bias: tuple[float, float, float] | None = None


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def find_spec():
    from run_recursive_onestep_dynamics_20260526 import SPECS  # noqa: PLC0415

    return next(spec for spec in SPECS if spec.name == SPEC_NAME)


def norm_columns(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True)


def output_name(rank: int, candidate: BiasCandidate) -> str:
    return f"metricbias29_rank{rank}_{slug(candidate.name)}.csv"


def make_bias_grid(limit: float = 0.0020, step: float = 0.00025) -> np.ndarray:
    values = np.arange(-limit, limit + 0.5 * step, step, dtype=np.float64)
    return np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1).reshape(-1, 3)


def score_biases(residuals: np.ndarray, biases: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    scores = np.zeros(len(biases), dtype=np.float64)
    for start in range(0, len(biases), chunk_size):
        chunk = biases[start : start + chunk_size]
        distances = np.linalg.norm(residuals[None, :, :] - chunk[:, None, :], axis=2)
        scores[start : start + chunk_size] = np.mean(distances <= R_HIT, axis=1)
    return scores


def best_bias(residuals: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, float]:
    scores = score_biases(residuals, biases)
    idx = int(np.argmax(scores))
    return biases[idx].copy(), float(scores[idx])


def top_bias_table(residuals: np.ndarray, biases: np.ndarray, top_k: int = 12) -> pd.DataFrame:
    scores = score_biases(residuals, biases)
    order = np.argsort(scores)[::-1][:top_k]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "bias_x": biases[order, 0],
            "bias_y": biases[order, 1],
            "bias_z": biases[order, 2],
            "oof_hit": scores[order],
        }
    )


def motion_regime_ids(coords: np.ndarray, thresholds: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    diffs = np.diff(coords, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    speed_last = speed[:, -1]
    dot = np.sum(d_last * d_prev, axis=1)
    denom = np.linalg.norm(d_last, axis=1) * np.linalg.norm(d_prev, axis=1) + 1e-12
    turn = np.arccos(np.clip(dot / denom, -1.0, 1.0))
    vertical_ratio = np.abs(d_last[:, 2]) / (np.linalg.norm(d_last, axis=1) + 1e-12)

    if thresholds is None:
        thresholds = {
            "speed": np.quantile(speed_last, [0.33, 0.66]),
            "turn": np.quantile(turn, [0.55, 0.82]),
            "vertical": np.quantile(vertical_ratio, [0.70]),
        }

    speed_bin = np.digitize(speed_last, thresholds["speed"])
    turn_bin = np.digitize(turn, thresholds["turn"])
    vertical_bin = np.digitize(vertical_ratio, thresholds["vertical"])
    regime = speed_bin * 6 + turn_bin * 2 + vertical_bin
    return regime.astype(np.int64), thresholds


def fit_regime_biases(
    residuals_local: np.ndarray,
    regimes: np.ndarray,
    biases: np.ndarray,
    fallback_bias: np.ndarray,
    min_rows: int = 220,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for regime_id in sorted(np.unique(regimes)):
        mask = regimes == regime_id
        if int(mask.sum()) < min_rows:
            out[int(regime_id)] = fallback_bias
            continue
        out[int(regime_id)] = best_bias(residuals_local[mask], biases)[0]
    return out


def apply_regime_local_bias(
    coords: np.ndarray,
    winner_pred: np.ndarray,
    regimes: np.ndarray,
    regime_biases: dict[int, np.ndarray],
    shrink: float,
    fallback_bias: np.ndarray,
) -> np.ndarray:
    basis = local_basis(coords)
    local_bias = np.zeros((len(coords), 3), dtype=np.float64)
    for regime_id in sorted(np.unique(regimes)):
        local_bias[regimes == regime_id] = regime_biases.get(int(regime_id), fallback_bias)
    return winner_pred + to_global(shrink * local_bias, basis)


def cross_validated_constant_bias(
    residuals: np.ndarray,
    biases: np.ndarray,
    folds: list[np.ndarray],
    shrink: float = 1.0,
) -> tuple[float, list[np.ndarray]]:
    pred_residuals = residuals.copy()
    fold_biases = []
    for val_idx in folds:
        train_mask = np.ones(len(residuals), dtype=bool)
        train_mask[val_idx] = False
        bias = best_bias(residuals[train_mask], biases)[0]
        fold_biases.append(bias)
        pred_residuals[val_idx] = residuals[val_idx] - shrink * bias
    return float(np.mean(np.linalg.norm(pred_residuals, axis=1) <= R_HIT)), fold_biases


def cross_validated_regime_bias(
    residuals_local: np.ndarray,
    regimes: np.ndarray,
    biases: np.ndarray,
    folds: list[np.ndarray],
    shrink: float,
    fallback_bias: np.ndarray,
) -> float:
    pred_residuals = residuals_local.copy()
    for val_idx in folds:
        train_mask = np.ones(len(residuals_local), dtype=bool)
        train_mask[val_idx] = False
        regime_biases = fit_regime_biases(
            residuals_local[train_mask],
            regimes[train_mask],
            biases,
            fallback_bias=fallback_bias,
        )
        local_bias = np.zeros((len(val_idx), 3), dtype=np.float64)
        for regime_id in sorted(np.unique(regimes[val_idx])):
            local_bias[regimes[val_idx] == regime_id] = regime_biases.get(int(regime_id), fallback_bias)
        pred_residuals[val_idx] = residuals_local[val_idx] - shrink * local_bias
    return float(np.mean(np.linalg.norm(pred_residuals, axis=1) <= R_HIT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-variance metric-center bias sweep around the 0.692 winner.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--champion-oof", type=Path, default=ROOT / "outputs" / "cache" / "champion_oof_20260524.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_metric_center_bias_20260529.md")
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

    residual_global = y - winner_oof
    train_basis = local_basis(train_coords)
    test_basis = local_basis(test_coords)
    residual_local = to_local(residual_global, train_basis)
    biases = make_bias_grid()
    eval_folds = make_folds(len(train_coords), 5, OOF_SEED + 529)

    print("Sweeping global and local metric-center bias", flush=True)
    global_table = top_bias_table(residual_global, biases)
    local_table = top_bias_table(residual_local, biases)
    global_bias, global_full_hit = best_bias(residual_global, biases)
    local_bias, local_full_hit = best_bias(residual_local, biases)
    global_cv_hit, global_fold_biases = cross_validated_constant_bias(residual_global, biases, eval_folds)
    local_cv_hit, local_fold_biases = cross_validated_constant_bias(residual_local, biases, eval_folds)
    global_cv_by_shrink = {
        shrink: cross_validated_constant_bias(residual_global, biases, eval_folds, shrink=shrink)[0]
        for shrink in [0.50, 0.75, 1.00]
    }
    local_cv_by_shrink = {
        shrink: cross_validated_constant_bias(residual_local, biases, eval_folds, shrink=shrink)[0]
        for shrink in [0.50, 0.75, 1.00]
    }

    train_regimes, thresholds = motion_regime_ids(train_coords)
    test_regimes, _ = motion_regime_ids(test_coords, thresholds)
    regime_biases = fit_regime_biases(residual_local, train_regimes, biases, fallback_bias=local_bias)
    regime_cv_rows = []
    for shrink in [0.35, 0.50, 0.65, 0.80, 1.00]:
        regime_cv_rows.append(
            {
                "shrink": shrink,
                "cv_hit": cross_validated_regime_bias(
                    residual_local,
                    train_regimes,
                    biases,
                    eval_folds,
                    shrink=shrink,
                    fallback_bias=local_bias,
                ),
            }
        )
    regime_cv = pd.DataFrame(regime_cv_rows).sort_values(["cv_hit", "shrink"], ascending=[False, True])
    regime_cv_map = {float(row["shrink"]): float(row["cv_hit"]) for _, row in regime_cv.iterrows()}

    candidates = [
        BiasCandidate("global_full", "global", 1.00, tuple(global_bias)),
        BiasCandidate("global_shrink075", "global", 0.75, tuple(global_bias)),
        BiasCandidate("local_full", "local", 1.00, tuple(local_bias)),
        BiasCandidate("local_shrink075", "local", 0.75, tuple(local_bias)),
        BiasCandidate("local_shrink050", "local", 0.50, tuple(local_bias)),
    ]
    for shrink in regime_cv["shrink"].head(3):
        candidates.append(BiasCandidate(f"regimelocal_shrink{int(round(float(shrink) * 100)):03d}", "regime_local", float(shrink)))

    rows = []
    predictions = {}
    for candidate in candidates:
        if candidate.family == "global":
            bias = np.asarray(candidate.bias, dtype=np.float64)
            pred_oof = winner_oof + candidate.shrink * bias
            pred_test = winner_test + candidate.shrink * bias
            cv_hit_proxy = global_cv_by_shrink[candidate.shrink]
        elif candidate.family == "local":
            bias = np.asarray(candidate.bias, dtype=np.float64)
            pred_oof = winner_oof + to_global(np.repeat((candidate.shrink * bias)[None, :], len(train_coords), axis=0), train_basis)
            pred_test = winner_test + to_global(np.repeat((candidate.shrink * bias)[None, :], len(test_coords), axis=0), test_basis)
            cv_hit_proxy = local_cv_by_shrink[candidate.shrink]
        elif candidate.family == "regime_local":
            pred_oof = apply_regime_local_bias(train_coords, winner_oof, train_regimes, regime_biases, candidate.shrink, local_bias)
            pred_test = apply_regime_local_bias(test_coords, winner_test, test_regimes, regime_biases, candidate.shrink, local_bias)
            cv_hit_proxy = regime_cv_map[candidate.shrink]
        else:
            raise ValueError(candidate.family)
        metrics = distance_summary(pred_oof, y)
        row = {
            "candidate": candidate.name,
            "family": candidate.family,
            "shrink": candidate.shrink,
            "oof_hit": metrics["r_hit_1cm"],
            "oof_delta_vs_winner": metrics["r_hit_1cm"] - winner_hit,
            "cv_hit_proxy": cv_hit_proxy,
            "cv_delta_vs_winner": cv_hit_proxy - winner_hit,
            **delta_summary(pred_oof, winner_oof, "oof_vs_winner"),
            **delta_summary(pred_test, winner_test, "test_vs_winner"),
        }
        row["selection_score"] = (
            row["cv_delta_vs_winner"]
            + 0.25 * row["oof_delta_vs_winner"]
            - 0.05 * max(0.0, row["test_vs_winner_mean_delta"] - 0.0010)
            - 0.02 * max(0.0, row["test_vs_winner_p95_delta"] - 0.0020)
        )
        rows.append(row)
        predictions[candidate.name] = pred_test

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "oof_delta_vs_winner", "test_vs_winner_mean_delta"],
        ascending=[False, False, True],
    )
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    written = []
    output_rows = []
    for rank, (_, row) in enumerate(leaderboard.head(args.top_k).iterrows(), start=1):
        candidate = next(candidate for candidate in candidates if candidate.name == row["candidate"])
        path = args.submission_dir / output_name(rank, candidate)
        write_submission(sample_submission, predictions[candidate.name], path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "candidate": candidate.name,
                "family": candidate.family,
                "oof_delta_vs_winner": float(row["oof_delta_vs_winner"]),
                "cv_delta_vs_winner": float(row["cv_delta_vs_winner"]),
                "selection_score": float(row["selection_score"]),
                "test_vs_winner_mean_delta": float(row["test_vs_winner_mean_delta"]),
                "test_vs_winner_p95_delta": float(row["test_vs_winner_p95_delta"]),
                "test_vs_winner_max_delta": float(row["test_vs_winner_max_delta"]),
            }
        )

    fold_bias_df = pd.DataFrame(
        {
            "fold": np.arange(1, len(eval_folds) + 1),
            "global_bias": [tuple(map(float, bias)) for bias in global_fold_biases],
            "local_bias": [tuple(map(float, bias)) for bias in local_fold_biases],
        }
    )
    outputs_df = pd.DataFrame(output_rows)
    report = [
        "# 2026-05-29 Metric-Center Bias",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_anchor: `{PUBLIC_WINNER} = {PUBLIC_WINNER_SCORE:.5f}`",
        f"- oof_winner_hit_proxy: `{winner_hit:.6f}`",
        f"- global_full_hit: `{global_full_hit:.6f}`",
        f"- local_full_hit: `{local_full_hit:.6f}`",
        f"- global_cv_hit: `{global_cv_hit:.6f}`",
        f"- local_cv_hit: `{local_cv_hit:.6f}`",
        "",
        "## Idea",
        "",
        "- Stop complex selector post-processing after the 0.692 plateau.",
        "- Search a low-dimensional constant bias that directly maximizes OOF R-Hit@1cm.",
        "- Test both global x/y/z bias and final-velocity local forward/side/up bias.",
        "- Add a simple unsupervised regime-local bias as a slightly stronger variant.",
        "",
        "## Leaderboard",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## Global Bias Top",
        "",
        dataframe_to_markdown(global_table),
        "",
        "## Local Bias Top",
        "",
        dataframe_to_markdown(local_table),
        "",
        "## Fold Bias Stability",
        "",
        dataframe_to_markdown(fold_bias_df),
        "",
        "## Regime CV",
        "",
        dataframe_to_markdown(regime_cv),
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
            "- If rank1 improves or ties, continue with low-variance metric-center calibration.",
            "- If rank1 drops, avoid constant-bias tuning and move to a genuinely new target formulation.",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
