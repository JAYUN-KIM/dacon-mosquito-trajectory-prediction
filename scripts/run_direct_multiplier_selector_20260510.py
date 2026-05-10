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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, split_mask, stack_samples  # noqa: E402
from run_direct_step_refine_20260509 import (  # noqa: E402
    WeightSpec,
    direct_prediction,
    direct_target_local,
    full_direct_local_prediction,
    sample_weights,
    write_submission,
)
from run_hit_weighted_local_frame import fit_predict_axes_weighted, make_features, safe_scale  # noqa: E402
from run_local_frame_residual import local_basis, to_local  # noqa: E402


CA_A6_SPEC = WeightSpec("ca_a6_s0055_c0105", "ca", 6.0, 0.0055, 0.0105)
SOURCE_SUBMISSION = "direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv"
SOURCE_MULT = (1.02, 1.00, 1.00)
CURRENT_BEST_NAME = "best_f102_s106_u094"
CURRENT_BEST_PUBLIC = 0.68300
N_OOF_FOLDS = 5
OOF_SEED = 20260510
SELECTOR_SEEDS = [42, 777, 2026]


@dataclass(frozen=True)
class Candidate:
    name: str
    mult: tuple[float, float, float]


CANDIDATES = [
    Candidate("base_f102_s100_u100", (1.02, 1.00, 1.00)),
    Candidate("prev_f102_s104_u096", (1.02, 1.04, 0.96)),
    Candidate(CURRENT_BEST_NAME, (1.02, 1.06, 0.94)),
    Candidate("side108_up092", (1.02, 1.08, 0.92)),
    Candidate("side104_up094", (1.02, 1.04, 0.94)),
    Candidate("side106_up096", (1.02, 1.06, 0.96)),
    Candidate("forward103_side104_up096", (1.03, 1.04, 0.96)),
    Candidate("forward101_side104_up096", (1.01, 1.04, 0.96)),
    Candidate("forward103_side106_up094", (1.03, 1.06, 0.94)),
    Candidate("forward101_side106_up094", (1.01, 1.06, 0.94)),
    Candidate("side110_up090", (1.02, 1.10, 0.90)),
    Candidate("forward103_side108_up092", (1.03, 1.08, 0.92)),
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def candidate_index(name: str) -> int:
    for idx, candidate in enumerate(CANDIDATES):
        if candidate.name == name:
            return idx
    raise ValueError(f"unknown candidate: {name}")


def make_folds(n_rows: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    return [fold.astype(np.int64) for fold in np.array_split(indices, n_folds)]


def build_oof_direct_local(coords: np.ndarray, y: np.ndarray, features: np.ndarray) -> np.ndarray:
    target = direct_target_local(coords, y)
    oof = np.zeros((len(coords), 3), dtype=np.float64)
    folds = make_folds(len(coords), N_OOF_FOLDS, OOF_SEED)

    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(coords), dtype=bool)
        train_mask[val_idx] = False
        print(f"OOF fold {fold_idx}/{N_OOF_FOLDS}: train={int(train_mask.sum())} val={len(val_idx)}", flush=True)
        weights = sample_weights(coords[train_mask], y[train_mask], CA_A6_SPEC)
        oof[val_idx] = fit_predict_axes_weighted(
            features[train_mask],
            target[train_mask],
            features[val_idx],
            5000 + fold_idx * 31,
            "l2",
            weights,
        )

    return oof


def make_candidate_predictions(coords: np.ndarray, pred_local_scaled: np.ndarray) -> np.ndarray:
    preds = []
    for candidate in CANDIDATES:
        preds.append(direct_prediction(coords, pred_local_scaled, candidate.mult))
    return np.stack(preds, axis=1)


def selector_extra_features(coords: np.ndarray, pred_local_scaled: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    dot = np.sum(d_last * d_prev, axis=1, keepdims=True)
    denom = np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-12
    turn_cos = dot / denom
    scale = safe_scale(coords)
    pred_abs = np.abs(pred_local_scaled)
    pred_norm = np.linalg.norm(pred_local_scaled, axis=1, keepdims=True)
    pred_ratio = pred_abs / (pred_norm + 1e-8)
    last_speed = speed[:, -1:]
    speed_ratio = last_speed / (speed.mean(axis=1, keepdims=True) + 1e-8)
    accel_ratio = accel[:, -1:] / (accel.mean(axis=1, keepdims=True) + 1e-8)
    z_ratio = np.abs(d_last[:, 2:3]) / (np.linalg.norm(d_last, axis=1, keepdims=True) + 1e-8)
    return np.hstack(
        [
            pred_local_scaled,
            pred_abs,
            pred_norm,
            pred_ratio,
            scale,
            last_speed,
            speed_ratio,
            accel[:, -1:],
            accel_ratio,
            turn_cos,
            z_ratio,
        ]
    ).astype(np.float32)


def make_selector_features(base_features: np.ndarray, coords: np.ndarray, pred_local_scaled: np.ndarray) -> np.ndarray:
    extra = selector_extra_features(coords, pred_local_scaled)
    out = np.hstack([base_features, extra]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def make_selector(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(CANDIDATES),
        n_estimators=420,
        learning_rate=0.025,
        num_leaves=47,
        min_child_samples=18,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.92,
        reg_alpha=0.04,
        reg_lambda=0.30,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def aligned_predict_proba(model: LGBMClassifier, features: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(features)
    if isinstance(raw, list):
        raw = np.vstack([p[:, 1] for p in raw]).T
    raw = np.asarray(raw, dtype=np.float64)
    aligned = np.zeros((len(features), len(CANDIDATES)), dtype=np.float64)
    for col, class_id in enumerate(model.classes_):
        aligned[:, int(class_id)] = raw[:, col]
    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, row_sum, out=np.full_like(aligned, 1.0 / len(CANDIDATES)), where=row_sum > 1e-12)
    return aligned


def label_weights(best_dist: np.ndarray, current_dist: np.ndarray) -> np.ndarray:
    weights = 1.0 + 4.0 * np.exp(-0.5 * ((best_dist - 0.010) / 0.0045) ** 2)
    weights += 2.0 * np.clip((current_dist - best_dist) / 0.004, 0.0, 1.0)
    weights = np.clip(weights, 0.5, 8.0)
    return weights / np.mean(weights)


def pick_by_indices(candidate_preds: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return candidate_preds[np.arange(len(indices)), indices]


def evaluate_selector_cv(
    selector_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    candidate_preds: np.ndarray,
    y: np.ndarray,
    current_idx: int,
) -> pd.DataFrame:
    rows = []
    for seed in SELECTOR_SEEDS:
        val_mask = split_mask(len(labels), 0.2, seed)
        train_mask = ~val_mask
        model = make_selector(seed)
        model.fit(selector_features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        proba = aligned_predict_proba(model, selector_features[val_mask])
        hard_idx = np.argmax(proba, axis=1)
        max_prob = np.max(proba, axis=1)
        current_pred = candidate_preds[val_mask, current_idx]
        y_val = y[val_mask]

        strategies: dict[str, np.ndarray] = {
            "selector_hard": pick_by_indices(candidate_preds[val_mask], hard_idx),
            "selector_soft": np.einsum("nc,ncd->nd", proba, candidate_preds[val_mask]),
            "best_global": current_pred,
        }
        for threshold in [0.35, 0.45, 0.55]:
            routed = current_pred.copy()
            chosen = pick_by_indices(candidate_preds[val_mask], hard_idx)
            route_mask = max_prob >= threshold
            routed[route_mask] = chosen[route_mask]
            strategies[f"selector_conf{threshold:.2f}"] = routed

        soft = strategies["selector_soft"]
        for weight in [0.20, 0.35, 0.50]:
            strategies[f"best_soft_blend_w{weight:.2f}"] = (1.0 - weight) * current_pred + weight * soft

        for name, pred in strategies.items():
            rows.append({"seed": seed, "strategy": name, **distance_summary(pred, y_val)})

    df = pd.DataFrame(rows)
    leaderboard = (
        df.groupby("strategy", as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    leaderboard["risk_adjusted_hit"] = leaderboard["mean_r_hit"] - 0.25 * leaderboard["std_r_hit"].fillna(0.0)
    return leaderboard


def read_submission_coords(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def recover_test_local_from_source(test_coords: np.ndarray, source_pred: np.ndarray) -> np.ndarray:
    basis = local_basis(test_coords)
    delta_local = to_local(source_pred - test_coords[:, -1, :], basis)
    return delta_local / safe_scale(test_coords) / np.asarray(SOURCE_MULT, dtype=np.float64)[None, :]


def build_test_outputs(
    selector_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    test_selector_features: np.ndarray,
    test_candidate_preds: np.ndarray,
    current_idx: int,
    sample_submission: pd.DataFrame,
    output_dir: Path,
    strategies: list[str],
) -> list[Path]:
    model = make_selector(88010)
    model.fit(selector_features, labels, sample_weight=weights)
    proba = aligned_predict_proba(model, test_selector_features)
    hard_idx = np.argmax(proba, axis=1)
    max_prob = np.max(proba, axis=1)
    current_pred = test_candidate_preds[:, current_idx]
    hard_pred = pick_by_indices(test_candidate_preds, hard_idx)
    soft_pred = np.einsum("nc,ncd->nd", proba, test_candidate_preds)

    strategy_preds: dict[str, np.ndarray] = {
        "selector_hard": hard_pred,
        "selector_soft": soft_pred,
        "best_global": current_pred,
    }
    for threshold in [0.35, 0.45, 0.55]:
        routed = current_pred.copy()
        route_mask = max_prob >= threshold
        routed[route_mask] = hard_pred[route_mask]
        strategy_preds[f"selector_conf{threshold:.2f}"] = routed
    for weight in [0.20, 0.35, 0.50]:
        strategy_preds[f"best_soft_blend_w{weight:.2f}"] = (1.0 - weight) * current_pred + weight * soft_pred

    written = []
    for rank, name in enumerate(strategies, start=1):
        pred = strategy_preds[name]
        path = output_dir / f"direct_selector_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample-wise selector for direct-step multiplier candidates.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_direct_multiplier_selector_20260510.md")
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
    train_features, candidate_feature_names = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    print(f"feature_count={train_features.shape[1]}", flush=True)

    print("Building 5-fold OOF direct-step local predictions", flush=True)
    oof_local = build_oof_direct_local(train_coords, y, train_features)
    train_candidate_preds = make_candidate_predictions(train_coords, oof_local)
    distances = np.linalg.norm(train_candidate_preds - y[:, None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    best_dist = np.min(distances, axis=1)
    current_idx = candidate_index(CURRENT_BEST_NAME)
    current_dist = distances[:, current_idx]
    weights = label_weights(best_dist, current_dist)

    selector_features = make_selector_features(train_features, train_coords, oof_local)

    print("Evaluating selector CV", flush=True)
    leaderboard = evaluate_selector_cv(selector_features, labels, weights, train_candidate_preds, y, current_idx)
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Preparing test candidate predictions", flush=True)
    source_pred = read_submission_coords(args.output_dir / SOURCE_SUBMISSION)
    if source_pred is None:
        print("Source submission is missing; training full CA_A6 direct-step local prediction", flush=True)
        test_local = full_direct_local_prediction(train_coords, y, train_features, test_features, CA_A6_SPEC)
    else:
        test_local = recover_test_local_from_source(test_coords, source_pred)
    test_candidate_preds = make_candidate_predictions(test_coords, test_local)
    test_selector_features = make_selector_features(test_features, test_coords, test_local)

    top_strategies = leaderboard.head(args.top_k)["strategy"].tolist()
    written = build_test_outputs(
        selector_features,
        labels,
        weights,
        test_selector_features,
        test_candidate_preds,
        current_idx,
        sample_submission,
        args.output_dir,
        top_strategies,
    )

    candidate_rows = []
    for idx, candidate in enumerate(CANDIDATES):
        candidate_rows.append(
            {
                "idx": idx,
                "candidate": candidate.name,
                "forward": candidate.mult[0],
                "side": candidate.mult[1],
                "up": candidate.mult[2],
                "label_count": int(np.sum(labels == idx)),
                "oof_hit": float(np.mean(distances[:, idx] <= 0.01)),
                "oof_mean_distance": float(np.mean(distances[:, idx])),
            }
        )
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["oof_hit", "oof_mean_distance"], ascending=[False, True])

    report = [
        "# 2026-05-10 Direct Multiplier Selector",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 데이터 경로: `{data_dir}`",
        f"- 현재 public best: `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv = {CURRENT_BEST_PUBLIC:.5f}`",
        f"- OOF folds: `{N_OOF_FOLDS}`",
        f"- selector CV seeds: `{SELECTOR_SEEDS}`",
        f"- 후보 multiplier 수: `{len(CANDIDATES)}`",
        f"- feature 수: `{train_features.shape[1]}` + selector extra",
        f"- direct feature family: `{candidate_feature_names}`",
        f"- 생성한 제출 파일: `{[str(path) for path in written]}`",
        "",
        "## Selector CV",
        "",
        dataframe_to_markdown(leaderboard),
        "",
        "## 후보별 OOF 성능과 label 분포",
        "",
        dataframe_to_markdown(candidate_df),
        "",
        "## 해석",
        "",
        "- 고정 multiplier best 주변 후보를 샘플별로 고르는 selector/routing 실험입니다.",
        "- train OOF direct-step 예측에서 후보별 거리를 계산해 oracle best 후보를 라벨로 만들었습니다.",
        "- public에서 selector가 실패하면, 후보 선택이 과적합이라는 뜻이므로 다음은 smoothing/denoising 또는 per-sample physics coefficient 축으로 이동하는 것이 좋습니다.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
