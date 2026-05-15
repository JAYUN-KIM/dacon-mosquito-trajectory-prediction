from __future__ import annotations

import argparse
import re
import sys
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
from run_aggressive_experiments import dataframe_to_markdown, distance_summary, split_mask, stack_samples  # noqa: E402
from run_direct_multiplier_selector_20260510 import (  # noqa: E402
    CURRENT_BEST_NAME,
    SOURCE_SUBMISSION,
    aligned_predict_proba,
    build_oof_direct_local,
    candidate_index,
    label_weights,
    make_candidate_predictions,
    make_folds,
    make_selector,
    make_selector_features,
    recover_test_local_from_source,
)
from run_hit_weighted_local_frame import make_features, safe_scale  # noqa: E402


PUBLIC_BEST_SUBMISSION = "direct_selector_rank2_selectorsoft.csv"
PUBLIC_BEST_SCORE = 0.68440
N_FOLDS = 5
FOLD_SEED = 20260515
CV_SEEDS = [42, 777, 2026]
CONFIGS = [
    {"name": "k32_s010", "k": 32, "shrink": 0.10, "power": 1.0, "boundary_amp": 1.5},
    {"name": "k32_s020", "k": 32, "shrink": 0.20, "power": 1.0, "boundary_amp": 1.5},
    {"name": "k48_s015", "k": 48, "shrink": 0.15, "power": 1.0, "boundary_amp": 2.5},
    {"name": "k64_s010", "k": 64, "shrink": 0.10, "power": 1.0, "boundary_amp": 2.5},
    {"name": "k64_s020", "k": 64, "shrink": 0.20, "power": 1.0, "boundary_amp": 2.5},
    {"name": "k96_s015", "k": 96, "shrink": 0.15, "power": 0.75, "boundary_amp": 3.0},
    {"name": "k128_s010", "k": 128, "shrink": 0.10, "power": 0.75, "boundary_amp": 3.0},
    {"name": "k128_s020", "k": 128, "shrink": 0.20, "power": 0.75, "boundary_amp": 3.0},
]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def read_submission_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[COORD_COLUMNS].to_numpy(dtype=np.float64)


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample_submission[[ID_COLUMN]].copy()
    out[COORD_COLUMNS] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def soft_predict(proba: np.ndarray, candidate_preds: np.ndarray) -> np.ndarray:
    return np.einsum("nc,ncd->nd", proba, candidate_preds)


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def build_oof_selector_proba(selector_features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    folds = make_folds(len(labels), N_FOLDS, FOLD_SEED)
    probas: np.ndarray | None = None
    for fold_idx, val_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[val_idx] = False
        model = make_selector(71000 + fold_idx * 23)
        model.fit(selector_features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        fold_proba = aligned_predict_proba(model, selector_features[val_idx])
        if probas is None:
            probas = np.zeros((len(labels), fold_proba.shape[1]), dtype=np.float64)
        if fold_proba.shape[1] > probas.shape[1]:
            expanded = np.zeros((len(labels), fold_proba.shape[1]), dtype=np.float64)
            expanded[:, : probas.shape[1]] = probas
            probas = expanded
        probas[val_idx, : fold_proba.shape[1]] = fold_proba
        print(f"selector OOF fold {fold_idx}/{N_FOLDS}: val={len(val_idx)}", flush=True)
    if probas is None:
        raise RuntimeError("failed to build selector OOF probabilities")
    row_sum = probas.sum(axis=1, keepdims=True)
    return np.divide(probas, row_sum, out=np.full_like(probas, 1.0 / probas.shape[1]), where=row_sum > 1e-12)


def motion_descriptor(coords: np.ndarray, pred_local_scaled: np.ndarray, proba: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    scale = safe_scale(coords)
    diffs_norm = diffs / scale[:, None, :]
    dd_norm = dd / scale[:, None, :]
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    turn_cos = np.sum(d_last * d_prev, axis=1, keepdims=True) / (
        np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-12
    )
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    top = sorted_proba[:, :4]
    entropy = -np.sum(np.clip(proba, 1e-12, 1.0) * np.log(np.clip(proba, 1e-12, 1.0)), axis=1, keepdims=True)
    descriptor = np.hstack(
        [
            diffs_norm[:, -5:, :].reshape(len(coords), -1),
            dd_norm[:, -4:, :].reshape(len(coords), -1),
            speed[:, -5:],
            accel[:, -4:],
            speed[:, -1:] / (np.mean(speed[:, -5:], axis=1, keepdims=True) + 1e-8),
            accel[:, -1:] / (np.mean(accel[:, -4:], axis=1, keepdims=True) + 1e-8),
            turn_cos,
            pred_local_scaled,
            np.abs(pred_local_scaled),
            np.linalg.norm(pred_local_scaled, axis=1, keepdims=True),
            top,
            top[:, 0:1] - top[:, 1:2],
            entropy,
        ]
    ).astype(np.float32)
    descriptor[~np.isfinite(descriptor)] = 0.0
    return descriptor


def boundary_weights(dist: np.ndarray, amp: float) -> np.ndarray:
    weights = 1.0 + amp * np.exp(-0.5 * ((dist - 0.010) / 0.0045) ** 2)
    return np.clip(weights, 0.5, 8.0)


def analog_correction(
    train_desc: np.ndarray,
    train_residual: np.ndarray,
    train_weight: np.ndarray,
    pred_desc: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_desc)
    pred_x = scaler.transform(pred_desc)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto", n_jobs=-1)
    nn.fit(train_x)
    dist, idx = nn.kneighbors(pred_x, return_distance=True)
    weights = train_weight[idx] / np.maximum(dist, 1e-6) ** power
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return np.einsum("nk,nkc->nc", weights, train_residual[idx])


def evaluate_configs(desc: np.ndarray, residual: np.ndarray, soft_pred: np.ndarray, y: np.ndarray, soft_dist: np.ndarray) -> pd.DataFrame:
    rows = [{"seed": -1, "config": "soft_anchor", **distance_summary(soft_pred, y)}]
    for seed in CV_SEEDS:
        val_mask = split_mask(len(y), 0.2, seed)
        train_mask = ~val_mask
        for cfg in CONFIGS:
            train_w = boundary_weights(soft_dist[train_mask], cfg["boundary_amp"])
            corr = analog_correction(
                desc[train_mask],
                residual[train_mask],
                train_w,
                desc[val_mask],
                cfg["k"],
                cfg["power"],
            )
            pred = soft_pred[val_mask] + cfg["shrink"] * corr
            rows.append({"seed": seed, "config": cfg["name"], **distance_summary(pred, y[val_mask])})
    df = pd.DataFrame(rows)
    summary = (
        df[df["config"] != "soft_anchor"]
        .groupby("config", as_index=False)
        .agg(
            mean_r_hit=("r_hit_1cm", "mean"),
            std_r_hit=("r_hit_1cm", "std"),
            min_r_hit=("r_hit_1cm", "min"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
        )
        .sort_values(["mean_r_hit", "min_r_hit", "mean_distance"], ascending=[False, False, True])
    )
    anchor = rows[0]
    summary["anchor_r_hit"] = anchor["r_hit_1cm"]
    summary["delta_hit_vs_anchor"] = summary["mean_r_hit"] - anchor["r_hit_1cm"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analog residual correction on top of selector-soft anchor.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_analog_residual_correction_20260515.md")
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
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)

    print("Building OOF direct-step local predictions", flush=True)
    oof_local = build_oof_direct_local(train_coords, y, train_features)
    train_candidate_preds = make_candidate_predictions(train_coords, oof_local)
    distances = np.linalg.norm(train_candidate_preds - y[:, None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    best_dist = np.min(distances, axis=1)
    current_dist = distances[:, candidate_index(CURRENT_BEST_NAME)]
    selector_weights = label_weights(best_dist, current_dist)
    selector_features = make_selector_features(train_features, train_coords, oof_local)

    print("Building selector OOF soft predictions", flush=True)
    train_proba = build_oof_selector_proba(selector_features, labels, selector_weights)
    soft_train = soft_predict(train_proba, train_candidate_preds)
    residual = y - soft_train
    soft_dist = np.linalg.norm(soft_train - y, axis=1)
    train_desc = motion_descriptor(train_coords, oof_local, train_proba)

    print("Evaluating analog residual configs", flush=True)
    cv = evaluate_configs(train_desc, residual, soft_train, y, soft_dist)
    print(cv.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    print("Preparing test analog corrections", flush=True)
    source_pred = read_submission_coords(args.submission_dir / SOURCE_SUBMISSION)
    test_local = recover_test_local_from_source(test_coords, source_pred)
    test_selector_features = make_selector_features(test_features, test_coords, test_local)
    full_selector = make_selector(88010)
    full_selector.fit(selector_features, labels, sample_weight=selector_weights)
    test_proba = aligned_predict_proba(full_selector, test_selector_features)
    test_desc = motion_descriptor(test_coords, test_local, test_proba)
    soft_test = read_submission_coords(args.submission_dir / PUBLIC_BEST_SUBMISSION)

    rows = []
    written = []
    top_configs = cv.head(args.top_k)["config"].tolist()
    cfg_by_name = {cfg["name"]: cfg for cfg in CONFIGS}
    for rank, name in enumerate(top_configs, start=1):
        cfg = cfg_by_name[name]
        train_w = boundary_weights(soft_dist, cfg["boundary_amp"])
        corr = analog_correction(train_desc, residual, train_w, test_desc, cfg["k"], cfg["power"])
        pred = soft_test + cfg["shrink"] * corr
        path = args.submission_dir / f"analogres_rank{rank}_{slug(name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        rows.append(
            {
                "rank": rank,
                "submission": path.name,
                **cfg,
                **delta_summary(pred, soft_test, "vs_soft_anchor"),
            }
        )

    report = [
        "# 2026-05-15 Analog Residual Correction",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- public_best_anchor: `{PUBLIC_BEST_SUBMISSION} = {PUBLIC_BEST_SCORE:.5f}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## 외부 자료/논문 검토",
        "",
        "- DACON 규칙상 외부 데이터 사용은 허용되지만, 제공 test 데이터는 어떤 형태로도 학습에 활용하면 안 되며 원격 API 기반 모델도 사용할 수 없다.",
        "- 모기/곤충 추적 문헌은 Kalman 계열의 운동 연속성, 속도/가속도 기반 예측, 유사 궤적 기반 보정 아이디어를 반복적으로 사용한다.",
        "- 이번 실험은 외부 궤적 데이터를 직접 섞지 않고, 논문 아이디어만 train-only analog residual prior로 구현했다.",
        "- 외부 모기 궤적 데이터는 센서, 좌표계, 실험 환경이 달라 private generalization을 해칠 위험이 커서 보류한다.",
        "- 참고: https://dacon.io/competitions/official/236716/overview/rules",
        "- 참고: https://arxiv.org/abs/2505.13615",
        "- 참고: https://www.mdpi.com/2079-9292/14/7/1333",
        "- 참고: https://arxiv.org/abs/2007.14216",
        "",
        "## CV",
        "",
        dataframe_to_markdown(cv),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(rows)),
        "",
        "## 메모",
        "",
        "- 새 비모수 축이다. 정규화된 최근 motion pattern이 비슷한 train 샘플을 찾고, 해당 샘플들의 OOF selector-soft 잔차를 작게 빌려온다.",
        "- public-best anchor가 이미 강하고 평가 지표가 1cm hit threshold이므로 correction은 강하게 shrink했다.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
