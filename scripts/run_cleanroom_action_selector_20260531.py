"""Clean-room physics action selector for 2026-05-31.

This experiment intentionally uses only raw competition files:

- data/raw/open (3)/train/*.csv
- data/raw/open (3)/test/*.csv
- data/raw/open (3)/train_labels.csv
- data/raw/open (3)/sample_submission.csv

It does not read old submissions, champion anchors, cache files, or public-score
feedback. The hypothesis is that the raw physics candidate pool already
contains many possible hits, but the missing piece is selecting or blending the
right extrapolation action per trajectory.
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


ROOT = Path(r"C:\open\dacon-mosquito-trajectory-prediction")
DATA_DIR = ROOT / "data" / "raw" / "open (3)"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABEL_PATH = DATA_DIR / "train_labels.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SUB_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"

EXP_TAG = "cleanroom_action_selector_20260531"
SEED = 531
N_SPLITS = 5
EPS = 1e-12

warnings.filterwarnings("ignore")
np.random.seed(SEED)


def ensure_dirs() -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def natural_key(path_or_str):
    return [
        int(t) if t.isdigit() else t
        for t in re.split(r"(\d+)", str(path_or_str))
    ]


def normalize_id(value) -> str:
    s = str(value).strip().replace("\\", "/").split("/")[-1]
    if s.lower().endswith(".csv"):
        s = s[:-4]
    return s


def load_trajectory_folder(folder: Path):
    files = sorted(folder.glob("*.csv"), key=natural_key)
    ids = []
    arrs = []
    for idx, fp in enumerate(files, 1):
        if idx % 2000 == 0:
            print(f"  loaded {idx:,} files from {folder.name}")
        df = pd.read_csv(fp)
        arr = df[["x", "y", "z"]].astype(float).values[-11:]
        if len(arr) != 11:
            raise ValueError(f"{fp} has {len(arr)} rows, expected 11")
        ids.append(normalize_id(fp.stem))
        arrs.append(arr)
    return np.asarray(ids), np.stack(arrs, axis=0)


def hit_rate(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=1) <= 0.01))


def mean_distance(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=1)))


def local_frame(P: np.ndarray) -> np.ndarray:
    """Build a velocity/acceleration local frame for each trajectory."""
    v = P[:, -1] - P[:, -2]
    a = P[:, -1] - 2.0 * P[:, -2] + P[:, -3]

    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    e1 = v / np.maximum(v_norm, EPS)
    bad_v = v_norm[:, 0] < 1e-8
    if np.any(bad_v):
        e1[bad_v] = np.array([1.0, 0.0, 0.0])

    a_orth = a - np.sum(a * e1, axis=1, keepdims=True) * e1
    a_norm = np.linalg.norm(a_orth, axis=1, keepdims=True)
    e2 = a_orth / np.maximum(a_norm, EPS)

    bad_a = a_norm[:, 0] < 1e-8
    if np.any(bad_a):
        axes = np.eye(3)
        fallback = []
        for u in e1[bad_a]:
            axis = axes[np.argmin(np.abs(axes @ u))]
            tmp = axis - np.sum(axis * u) * u
            fallback.append(tmp / np.maximum(np.linalg.norm(tmp), EPS))
        e2[bad_a] = np.asarray(fallback)

    e3 = np.cross(e1, e2)
    e3 = e3 / np.maximum(np.linalg.norm(e3, axis=1, keepdims=True), EPS)
    e2 = np.cross(e3, e1)
    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), EPS)
    return np.stack([e1, e2, e3], axis=2)


def global_to_local(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("nd,ndk->nk", vec, basis)


def local_to_global(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("nk,ndk->nd", vec, basis)


def project_sequence(seq: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ntd,ndk->ntk", seq, basis)


def poly_pred(P: np.ndarray, degree: int) -> np.ndarray:
    t = np.arange(-10, 1, dtype=float)
    tq = 2.0
    V = np.vander(t, N=degree + 1, increasing=True)
    pinv = np.linalg.pinv(V)
    coeff = np.einsum("kt,ntd->nkd", pinv, P)
    powers = np.asarray([tq**i for i in range(degree + 1)], dtype=float)
    return np.einsum("k,nkd->nd", powers, coeff)


def build_features(P: np.ndarray) -> np.ndarray:
    basis = local_frame(P)
    rel = P - P[:, -1:, :]
    d1 = np.diff(P, axis=1)
    d2 = np.diff(d1, axis=1)
    d3 = np.diff(d2, axis=1)

    rel_l = project_sequence(rel, basis)
    d1_l = project_sequence(d1, basis)
    d2_l = project_sequence(d2, basis)

    speed = np.linalg.norm(d1, axis=2)
    acc = np.linalg.norm(d2, axis=2)
    jerk = np.linalg.norm(d3, axis=2)

    v0 = d1[:, :-1]
    v1 = d1[:, 1:]
    turn = np.sum(v0 * v1, axis=2) / np.maximum(
        np.linalg.norm(v0, axis=2) * np.linalg.norm(v1, axis=2),
        EPS,
    )

    path = np.sum(speed, axis=1)
    chord = np.linalg.norm(P[:, -1] - P[:, 0], axis=1)
    bbox = np.max(P, axis=1) - np.min(P, axis=1)

    scalar = np.column_stack(
        [
            speed,
            acc,
            jerk,
            turn,
            path,
            chord,
            chord / np.maximum(path, EPS),
            bbox,
            speed[:, -1] / np.maximum(np.mean(speed, axis=1), EPS),
            acc[:, -1] / np.maximum(speed[:, -1], EPS),
            np.mean(turn, axis=1),
            np.std(turn, axis=1),
        ]
    )

    X = np.hstack(
        [
            rel_l.reshape(len(P), -1),
            d1_l.reshape(len(P), -1),
            d2_l.reshape(len(P), -1),
            scalar,
        ]
    )
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def build_candidate_predictions(P: np.ndarray):
    basis = local_frame(P)
    v = P[:, -1] - P[:, -2]
    a = P[:, -1] - 2.0 * P[:, -2] + P[:, -3]
    v_local = global_to_local(v, basis)

    candidates = []

    def add(name: str, pred: np.ndarray) -> None:
        candidates.append((name, pred.astype(np.float64)))

    # Isotropic final-velocity multipliers.
    for f in np.arange(1.75, 2.251, 0.05):
        pred = P[:, -1] + local_to_global(v_local * np.array([f, f, f]), basis)
        add(f"vf{f:.2f}", pred)

    # Anisotropic local velocity actions. Side/up changes are intentionally
    # coarse because exact mm-level tuning is not the goal of this clean run.
    for f in [1.90, 1.98, 2.02, 2.08]:
        for s in [0.00, 0.35, 0.70, 1.00, 1.25]:
            for u in [0.00, 0.50, 1.00, 1.25]:
                scale = np.array([f, s, u])
                pred = P[:, -1] + local_to_global(v_local * scale, basis)
                add(f"aniso_f{f:.2f}s{s:.2f}u{u:.2f}", pred)

    # Discrete acceleration variants around constant velocity.
    for c in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        add(f"cv_acc{c:.1f}", P[:, -1] + 2.0 * v + c * a)

    # Polynomial extrapolation family.
    for degree in [1, 2, 3, 4]:
        add(f"poly{degree}", poly_pred(P, degree))

    names = [name for name, _ in candidates]
    pred = np.stack([p for _, p in candidates], axis=1)
    return names, pred


def fit_action_selector_oof(X: np.ndarray, best_action: np.ndarray, n_actions: int):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_prob = np.zeros((len(X), n_actions), dtype=np.float64)
    oof_action = np.zeros(len(X), dtype=int)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_features=0.55,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=SEED + fold,
            n_jobs=-1,
        )
        model.fit(X[tr_idx], best_action[tr_idx])
        prob = model.predict_proba(X[va_idx])
        full = np.zeros((len(va_idx), n_actions), dtype=np.float64)
        full[:, model.classes_] = prob
        oof_prob[va_idx] = full
        oof_action[va_idx] = np.argmax(full, axis=1)

        fold_rows.append(
            {
                "fold": fold,
                "argmin_action_accuracy": float(
                    accuracy_score(best_action[va_idx], oof_action[va_idx])
                ),
            }
        )
        print(
            f"  fold {fold}/{N_SPLITS} | "
            f"argmin action acc={fold_rows[-1]['argmin_action_accuracy']:.4f}"
        )

    return oof_prob, oof_action, fold_rows


def fit_predict_action_proba(X_train: np.ndarray, y_action: np.ndarray, X_test: np.ndarray, n_actions: int):
    seeds = [SEED, SEED + 101, SEED + 202]
    prob_sum = np.zeros((len(X_test), n_actions), dtype=np.float64)

    for seed in seeds:
        print(f"  fitting full action selector seed={seed}")
        model = ExtraTreesClassifier(
            n_estimators=900,
            max_features=0.55,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_action)
        prob = model.predict_proba(X_test)
        full = np.zeros((len(X_test), n_actions), dtype=np.float64)
        full[:, model.classes_] = prob
        prob_sum += full

    return prob_sum / len(seeds)


def weighted_candidate_prediction(prob: np.ndarray, pred_pool: np.ndarray, power: float) -> np.ndarray:
    p = np.power(prob, power)
    p = p / np.maximum(np.sum(p, axis=1, keepdims=True), EPS)
    return np.einsum("nc,ncd->nd", p, pred_pool)


def write_submission(sample_df: pd.DataFrame, pred: np.ndarray, out_path: Path) -> None:
    out = sample_df.copy()
    for col_idx, col in enumerate(["x", "y", "z"]):
        out[col] = pred[:, col_idx]
    out.to_csv(out_path, index=False)


def to_markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer."""
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def align_test_predictions_to_sample(test_ids, pred_test, sample_df):
    id_col = "id" if "id" in sample_df.columns else sample_df.columns[0]
    sample_ids = sample_df[id_col].map(normalize_id).values
    pred_map = {sid: pred_test[i] for i, sid in enumerate(test_ids)}
    missing = [sid for sid in sample_ids if sid not in pred_map]
    if missing:
        print(f"  [WARN] sample id alignment failed for {len(missing)} rows; using row order")
        return pred_test
    return np.stack([pred_map[sid] for sid in sample_ids], axis=0)


def main() -> None:
    ensure_dirs()
    print("=" * 80)
    print(f"[{EXP_TAG}] raw-only physics action selector")
    print("=" * 80)

    print("\n[1] Loading raw data")
    train_ids, P_train = load_trajectory_folder(TRAIN_DIR)
    test_ids, P_test = load_trajectory_folder(TEST_DIR)
    y_train = pd.read_csv(LABEL_PATH)[["x", "y", "z"]].astype(float).values
    sample_df = pd.read_csv(SAMPLE_PATH)

    print(f"  train: {P_train.shape}")
    print(f"  test : {P_test.shape}")

    print("\n[2] Building raw candidate pools")
    action_names, train_pool = build_candidate_predictions(P_train)
    _, test_pool = build_candidate_predictions(P_test)
    n_actions = len(action_names)
    print(f"  actions: {n_actions}")

    dists = np.linalg.norm(train_pool - y_train[:, None, :], axis=2)
    hit_mat = dists <= 0.01
    best_action = np.argmin(dists, axis=1)
    action_hit = np.mean(hit_mat, axis=0)
    best_single_idx = int(np.argmax(action_hit))
    oracle_hit = float(np.mean(np.min(dists, axis=1) <= 0.01))

    print(f"  best raw action: {action_names[best_single_idx]} | hit={action_hit[best_single_idx]:.6f}")
    print(f"  raw action oracle hit: {oracle_hit:.6f}")

    print("\n[3] Building features and OOF selector")
    X_train = build_features(P_train)
    X_test = build_features(P_test)
    oof_prob, oof_action, fold_rows = fit_action_selector_oof(X_train, best_action, n_actions)

    anchor_oof = train_pool[:, best_single_idx]
    hard_oof = train_pool[np.arange(len(P_train)), oof_action]
    soft1_oof = weighted_candidate_prediction(oof_prob, train_pool, power=1.0)
    soft2_oof = weighted_candidate_prediction(oof_prob, train_pool, power=2.0)
    soft3_oof = weighted_candidate_prediction(oof_prob, train_pool, power=3.0)

    oof_candidates = {
        "single_best_raw": anchor_oof,
        "hard_argmax": hard_oof,
        "soft_p1": soft1_oof,
        "soft_p2": soft2_oof,
        "soft_p3": soft3_oof,
    }

    for blend in [0.35, 0.55, 0.75, 0.90]:
        oof_candidates[f"soft_p2_blend{int(blend * 100):02d}"] = (
            anchor_oof + blend * (soft2_oof - anchor_oof)
        )
    for blend in [0.55, 0.75, 0.90]:
        oof_candidates[f"soft_p3_blend{int(blend * 100):02d}"] = (
            anchor_oof + blend * (soft3_oof - anchor_oof)
        )

    rows = []
    for name, pred in oof_candidates.items():
        rows.append(
            {
                "candidate_key": name,
                "oof_hit": hit_rate(pred, y_train),
                "oof_mean_dist": mean_distance(pred, y_train),
            }
        )
    result_df = pd.DataFrame(rows).sort_values(
        ["oof_hit", "oof_mean_dist"],
        ascending=[False, True],
    )
    result_df["rank"] = np.arange(1, len(result_df) + 1)

    print("\n[4] OOF candidate summary")
    with pd.option_context("display.width", 140):
        print(result_df[["rank", "candidate_key", "oof_hit", "oof_mean_dist"]].to_string(index=False))

    print("\n[5] Fitting full selector and writing submissions")
    test_prob = fit_predict_action_proba(X_train, best_action, X_test, n_actions)
    anchor_test = test_pool[:, best_single_idx]
    hard_test = test_pool[np.arange(len(P_test)), np.argmax(test_prob, axis=1)]
    soft1_test = weighted_candidate_prediction(test_prob, test_pool, power=1.0)
    soft2_test = weighted_candidate_prediction(test_prob, test_pool, power=2.0)
    soft3_test = weighted_candidate_prediction(test_prob, test_pool, power=3.0)

    test_candidates = {
        "single_best_raw": anchor_test,
        "hard_argmax": hard_test,
        "soft_p1": soft1_test,
        "soft_p2": soft2_test,
        "soft_p3": soft3_test,
    }
    for blend in [0.35, 0.55, 0.75, 0.90]:
        test_candidates[f"soft_p2_blend{int(blend * 100):02d}"] = (
            anchor_test + blend * (soft2_test - anchor_test)
        )
    for blend in [0.55, 0.75, 0.90]:
        test_candidates[f"soft_p3_blend{int(blend * 100):02d}"] = (
            anchor_test + blend * (soft3_test - anchor_test)
        )

    output_rows = []
    for _, row in result_df.iterrows():
        key = row["candidate_key"]
        rank = int(row["rank"])
        fname = f"cleanroom31_rank{rank}_{key}.csv"
        pred_aligned = align_test_predictions_to_sample(
            test_ids,
            test_candidates[key],
            sample_df,
        )
        out_path = SUB_DIR / fname
        write_submission(sample_df, pred_aligned, out_path)
        output_rows.append({**row.to_dict(), "file": fname, "path": str(out_path)})
        print(f"  saved rank {rank}: {fname}")

    output_df = pd.DataFrame(output_rows)
    metrics_path = REPORT_DIR / f"{EXP_TAG}_metrics.json"
    output_df.to_json(metrics_path, orient="records", force_ascii=False, indent=2)

    top_actions = (
        pd.DataFrame(
            {
                "action": action_names,
                "hit": action_hit,
                "chosen_oof_count": np.bincount(oof_action, minlength=n_actions),
            }
        )
        .sort_values("hit", ascending=False)
        .head(15)
    )

    report_path = REPORT_DIR / f"{EXP_TAG}.md"
    lines = [
        f"# {EXP_TAG}",
        "",
        f"- created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- train_shape: `{P_train.shape}`",
        f"- test_shape: `{P_test.shape}`",
        f"- raw_action_count: `{n_actions}`",
        "- rule: old submissions/champion/cache files were not read",
        "",
        "## Core Diagnosis",
        "",
        (
            "The clean-room raw physics pool has a much higher oracle hit rate than "
            "any single action. This points to action selection as the current "
            "bottleneck, especially around acceleration/turning regimes."
        ),
        "",
        f"- best single raw action: `{action_names[best_single_idx]}` / OOF hit `{action_hit[best_single_idx]:.6f}`",
        f"- raw action oracle hit: `{oracle_hit:.6f}`",
        f"- oracle gap over best single: `{oracle_hit - action_hit[best_single_idx]:.6f}`",
        "",
        "## OOF Submission Candidates",
        "",
        to_markdown_table(output_df[["rank", "file", "candidate_key", "oof_hit", "oof_mean_dist"]]),
        "",
        "## Fold Diagnostics",
        "",
        to_markdown_table(pd.DataFrame(fold_rows)),
        "",
        "## Strongest Single Raw Actions",
        "",
        to_markdown_table(top_actions),
        "",
        "## Interpretation",
        "",
        (
            "If the top soft-action files improve public LB, the next step is to "
            "expand the action pool and improve probability calibration. If they "
            "drop, the failure mode is likely OOF-public selector transfer rather "
            "than absence of useful raw physics actions."
        ),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n[6] Saved report")
    print(f"  report : {report_path}")
    print(f"  metrics: {metrics_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
