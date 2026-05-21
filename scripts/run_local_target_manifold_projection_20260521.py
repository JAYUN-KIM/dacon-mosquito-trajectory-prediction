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
from run_curvature_gate_20260519 import (  # noqa: E402
    boundary_weights,
    curvature_correction,
    make_gate_classifier,
    make_gate_features,
    predict_gate_proba,
    read_submission_coords,
    write_submission,
)
from run_direct_step_refine_20260509 import direct_prediction, direct_target_local  # noqa: E402
from run_hit_weighted_local_frame import make_features, safe_scale  # noqa: E402


CURRENT_BEST = "curvgate_refine_rank2_gatet52a105.csv"
CURRENT_BEST_SCORE = 0.69120
BACKUP_BEST = "curvgate_rank4_gatet54a105.csv"
GATE_THRESHOLD = 0.52
GATE_ALPHA = 0.105
OOF_FOLDS = 5
OOF_SEED = 20260521

K_VALUES = [32, 64, 128, 256]
BLENDS = [0.025, 0.040, 0.060, 0.080, 0.100]
CAPS = [0.00035, 0.00055, 0.00075, 0.00100]


def slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def clip_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / (norm + 1e-12))
    return vectors * scale


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def motion_context(coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(coords, axis=1)
    dd = np.diff(diffs, axis=1)
    speed = np.linalg.norm(diffs, axis=2)
    accel = np.linalg.norm(dd, axis=2)
    scale = safe_scale(coords)
    d_last = diffs[:, -1, :]
    d_prev = diffs[:, -2, :]
    dot = np.sum(d_last * d_prev, axis=1, keepdims=True)
    denom = np.linalg.norm(d_last, axis=1, keepdims=True) * np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-12
    turn_cos = dot / denom
    blocks = [
        d_last / scale,
        d_prev / scale,
        np.abs(d_last) / scale,
        speed[:, -5:] / scale,
        accel[:, -5:] / scale,
        speed[:, -1:] / (speed[:, -5:].mean(axis=1, keepdims=True) + 1e-8),
        accel[:, -1:] / (accel[:, -5:].mean(axis=1, keepdims=True) + 1e-8),
        turn_cos,
        np.abs(d_last[:, 1:2]) / (np.linalg.norm(d_last, axis=1, keepdims=True) + 1e-8),
        np.abs(d_last[:, 2:3]) / (np.linalg.norm(d_last, axis=1, keepdims=True) + 1e-8),
    ]
    out = np.hstack(blocks).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def build_gate_oof_proba(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    proba = np.zeros(len(labels), dtype=np.float64)
    for seed in [42, 777, 2026]:
        fold_proba = np.zeros(len(labels), dtype=np.float64)
        val_mask = split_mask(len(labels), 0.2, seed)
        train_mask = ~val_mask
        model = make_gate_classifier(810000 + seed)
        model.fit(features[train_mask], labels[train_mask], sample_weight=weights[train_mask])
        fold_proba[val_mask] = predict_gate_proba(model, features[val_mask])
        # Fill the complementary fold using a second model so every row gets 3 independent-ish predictions.
        model2 = make_gate_classifier(811000 + seed)
        model2.fit(features[val_mask], labels[val_mask], sample_weight=weights[val_mask])
        fold_proba[train_mask] = predict_gate_proba(model2, features[train_mask])
        proba += fold_proba / 3.0
    return proba


def build_full_gate_proba(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    preds = []
    for seed in [1301, 2718, 3141]:
        print(f"full gate for manifold seed={seed}", flush=True)
        model = make_gate_classifier(820000 + seed)
        model.fit(features, labels, sample_weight=weights)
        preds.append(predict_gate_proba(model, test_features))
    return np.mean(preds, axis=0)


def build_champion_oof(
    train_coords: np.ndarray,
    y: np.ndarray,
    train_features: np.ndarray,
    anchor_oof: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_oof = anchor_oof + 0.090 * correction
    anchor_dist = np.linalg.norm(anchor_oof - y, axis=1)
    fixed_dist = np.linalg.norm(fixed_oof - y, axis=1)
    labels = (fixed_dist < anchor_dist).astype(np.int8)
    weights = boundary_weights(anchor_dist, fixed_dist)
    gate_features = make_gate_features(train_coords, train_features, anchor_oof, correction)
    print("Building OOF gate probabilities for champion proxy", flush=True)
    gate_proba = build_gate_oof_proba(gate_features, labels, weights)
    champion = anchor_oof.copy()
    mask = gate_proba >= GATE_THRESHOLD
    champion[mask] = anchor_oof[mask] + GATE_ALPHA * correction[mask]
    return champion, gate_proba


def neighbor_projection(
    train_space: np.ndarray,
    train_target_local: np.ndarray,
    query_space: np.ndarray,
    k: int,
    self_indices: np.ndarray | None = None,
) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_space)
    query_scaled = scaler.transform(query_space)
    extra = 10 if self_indices is not None else 0
    n_neighbors = min(len(train_space), k + extra)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(train_scaled)
    distances, indices = nn.kneighbors(query_scaled, return_distance=True)
    projected = np.zeros((len(query_space), 3), dtype=np.float64)
    for row in range(len(query_space)):
        row_idx = indices[row]
        row_dist = distances[row]
        if self_indices is not None:
            keep = row_idx != self_indices[row]
            row_idx = row_idx[keep]
            row_dist = row_dist[keep]
        row_idx = row_idx[:k]
        row_dist = row_dist[:k]
        weights = 1.0 / (row_dist + 0.05)
        weights = weights / np.sum(weights)
        projected[row] = np.sum(train_target_local[row_idx] * weights[:, None], axis=0)
    return projected


def apply_projection(
    coords: np.ndarray,
    current_pred: np.ndarray,
    projected_local: np.ndarray,
    blend: float,
    cap: float,
) -> np.ndarray:
    current_local = direct_target_local(coords, current_pred)
    moved_local = current_local + blend * (projected_local - current_local)
    raw_pred = direct_prediction(coords, moved_local, (1.0, 1.0, 1.0))
    move = clip_vectors(raw_pred - current_pred, cap)
    return current_pred + move


def evaluate_configs(
    coords: np.ndarray,
    champion: np.ndarray,
    projected_by_k: dict[int, np.ndarray],
    y: np.ndarray,
) -> pd.DataFrame:
    rows = [{"kind": "baseline", "k": 0, "blend": 0.0, "cap": 0.0, **distance_summary(champion, y), **delta_summary(champion, champion, "vs_champion")}]
    for k, projected in projected_by_k.items():
        for blend in BLENDS:
            for cap in CAPS:
                pred = apply_projection(coords, champion, projected, blend, cap)
                rows.append(
                    {
                        "kind": "manifold_projection",
                        "k": k,
                        "blend": blend,
                        "cap": cap,
                        **distance_summary(pred, y),
                        **delta_summary(pred, champion, "vs_champion"),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["r_hit_1cm", "mean_distance", "vs_champion_p95_delta"], ascending=[False, True, True])
    baseline_hit = float(df.loc[df["kind"] == "baseline", "r_hit_1cm"].iloc[0])
    df["delta_hit_vs_baseline"] = df["r_hit_1cm"] - baseline_hit
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local target manifold projection around the curvature-gate champion.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cache-path", type=Path, default=ROOT / "outputs" / "cache" / "curvature_gate_oof_20260519.npz")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_local_target_manifold_projection_20260521.md")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_raw_data_dir(args.data_dir)
    if not args.cache_path.exists():
        raise FileNotFoundError(args.cache_path)

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

    print("Building champion OOF proxy", flush=True)
    cache = np.load(args.cache_path)
    anchor_oof = cache["anchor_oof"]
    train_features, _ = make_features(train_coords)
    test_features, _ = make_features(test_coords)
    _, _, train_correction = curvature_correction(train_coords)
    _, _, test_correction = curvature_correction(test_coords)
    champion_oof, gate_oof_proba = build_champion_oof(train_coords, y, train_features, anchor_oof, train_correction)
    print("Champion OOF proxy:")
    print(pd.DataFrame([{"name": "champion_oof_proxy", **distance_summary(champion_oof, y)}]).to_string(index=False), flush=True)

    true_local = direct_target_local(train_coords, y)
    champion_local = direct_target_local(train_coords, champion_oof)
    train_context = motion_context(train_coords)
    train_space = np.hstack([true_local, train_context]).astype(np.float32)
    train_query_space = np.hstack([champion_local, train_context]).astype(np.float32)

    print("Building OOF manifold projections", flush=True)
    projected_train_by_k = {
        k: neighbor_projection(train_space, true_local, train_query_space, k, self_indices=np.arange(len(train_coords)))
        for k in K_VALUES
    }
    leaderboard = evaluate_configs(train_coords, champion_oof, projected_train_by_k, y)
    print(leaderboard.head(30).to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)

    current_best = read_submission_coords(args.submission_dir / CURRENT_BEST)
    backup_best = read_submission_coords(args.submission_dir / BACKUP_BEST)
    test_local = direct_target_local(test_coords, current_best)
    test_context = motion_context(test_coords)
    test_space = np.hstack([test_local, test_context]).astype(np.float32)

    print("Building test manifold projections", flush=True)
    projected_test_by_k = {
        k: neighbor_projection(train_space, true_local, test_space, k, self_indices=None)
        for k in K_VALUES
    }

    output_rows = []
    written: list[Path] = []
    for _, row in leaderboard[leaderboard["kind"] == "manifold_projection"].iterrows():
        if len(written) >= args.top_k:
            break
        k = int(row["k"])
        blend = float(row["blend"])
        cap = float(row["cap"])
        pred = apply_projection(test_coords, current_best, projected_test_by_k[k], blend, cap)
        rank = len(written) + 1
        path = args.submission_dir / f"manifoldproj_rank{rank}_k{k}_b{int(blend * 1000):03d}_cap{int(cap * 10000):04d}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "k": k,
                "blend": blend,
                "cap": cap,
                "oof_r_hit": float(row["r_hit_1cm"]),
                "oof_delta_vs_baseline": float(row["delta_hit_vs_baseline"]),
                **delta_summary(pred, current_best, "vs_current_best"),
                **delta_summary(pred, backup_best, "vs_backup_best"),
            }
        )

    report = [
        "# 2026-05-21 Local Target Manifold Projection",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- current_best: `{CURRENT_BEST} = {CURRENT_BEST_SCORE:.5f}`",
        f"- backup_best: `{BACKUP_BEST}`",
        "- idea: project the champion's local-frame displacement weakly toward nearby train target-local manifold points conditioned on motion context.",
        f"- k_values: `{K_VALUES}`",
        f"- blends: `{BLENDS}`",
        f"- caps: `{CAPS}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## OOF Leaderboard",
        "",
        dataframe_to_markdown(leaderboard.head(50)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- This is intentionally different from adding a new model prediction. It only nudges champion predictions toward dense train target-local regions.",
        "- The OOF proxy excludes each sample's own target from its neighbor set, but this remains an optimistic train-manifold diagnostic.",
        "- If public drops, manifold projection should be used only as a diagnostic feature, not a direct post-process.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
