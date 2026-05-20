from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
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
from run_aggressive_experiments import dataframe_to_markdown, stack_samples  # noqa: E402
from run_direct_step_refine_20260509 import (  # noqa: E402
    direct_prediction,
    direct_target_local,
    sample_weights,
)
from run_hit_weighted_local_frame import make_features  # noqa: E402
from run_temporal_backcast_augmentation_20260516 import TRUE_WEIGHT_SPEC, pseudo_from_cutoff  # noqa: E402


CURRENT_BEST = "curvgate_refine_rank2_gatet52a105.csv"
CURRENT_BEST_SCORE = 0.69120
SEEDS = [43, 777, 2026]
PSEUDO_CUTOFFS = (6, 7, 8)
PSEUDO_WEIGHT = 0.16
MULTS = {
    "f102_s100_u100": (1.02, 1.00, 1.00),
    "f102_s104_u096": (1.02, 1.04, 0.96),
    "f102_s106_u094": (1.02, 1.06, 0.94),
}


@dataclass(frozen=True)
class BlendSpec:
    name: str
    mult_name: str
    blend: float
    cap: float | None


BLEND_SPECS = [
    BlendSpec("blend12_base", "f102_s100_u100", 0.12, None),
    BlendSpec("blend08_base", "f102_s100_u100", 0.08, None),
    BlendSpec("blend16_base", "f102_s100_u100", 0.16, None),
    BlendSpec("blend20cap0012_base", "f102_s100_u100", 0.20, 0.0012),
    BlendSpec("blend12_tilt104096", "f102_s104_u096", 0.12, None),
    BlendSpec("blend20cap0012_tilt104096", "f102_s104_u096", 0.20, 0.0012),
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


def delta_summary(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    dist = np.linalg.norm(pred - reference, axis=1)
    return {
        f"{prefix}_mean_delta": float(np.mean(dist)),
        f"{prefix}_median_delta": float(np.median(dist)),
        f"{prefix}_p95_delta": float(np.quantile(dist, 0.95)),
        f"{prefix}_max_delta": float(np.max(dist)),
    }


def clip_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / (norm + 1e-12))
    return vectors * scale


def build_augmented_training(coords: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_blocks = [coords]
    target_blocks = [y]
    weight_blocks = [sample_weights(coords, y, TRUE_WEIGHT_SPEC)]

    for cutoff in PSEUDO_CUTOFFS:
        pseudo_coords, pseudo_y = pseudo_from_cutoff(coords, cutoff)
        coord_blocks.append(pseudo_coords)
        target_blocks.append(pseudo_y)
        pseudo_weights = sample_weights(pseudo_coords, pseudo_y, TRUE_WEIGHT_SPEC) * PSEUDO_WEIGHT
        weight_blocks.append(pseudo_weights)

    all_coords = np.concatenate(coord_blocks, axis=0)
    all_targets = np.concatenate(target_blocks, axis=0)
    weights = np.concatenate(weight_blocks).astype(np.float64)
    weights = weights / np.mean(weights)

    features, _ = make_features(all_coords)
    target_local = direct_target_local(all_coords, all_targets)
    return features.astype(np.float32), target_local.astype(np.float64), weights


def make_mlp(seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(192, 96),
        activation="relu",
        solver="adam",
        alpha=0.0035,
        batch_size=512,
        learning_rate="constant",
        learning_rate_init=0.00075,
        max_iter=260,
        shuffle=True,
        random_state=seed,
        tol=0.00004,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=18,
        verbose=False,
    )


def fit_predict_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(train_y)
    model = make_pipeline(StandardScaler(), make_mlp(seed))
    model.fit(train_x, y_scaled, mlpregressor__sample_weight=weights)
    mlp: MLPRegressor = model.named_steps["mlpregressor"]
    pred_scaled = model.predict(test_x)
    pred = target_scaler.inverse_transform(pred_scaled)
    return pred, {
        "seed": seed,
        "n_iter": float(mlp.n_iter_),
        "best_validation_score": float(getattr(mlp, "best_validation_score_", np.nan)),
        "loss": float(mlp.loss_),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP sequence pseudo-supervision blend experiment.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "latest_mlp_sequence_pseudo_blend_20260520.md")
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

    print("Building temporal-pseudo MLP training matrix", flush=True)
    train_x, train_y, weights = build_augmented_training(train_coords, y)
    test_x, _ = make_features(test_coords)
    print(f"train_rows={len(train_x)} feature_count={train_x.shape[1]} weight_mean={weights.mean():.4f}", flush=True)

    pred_locals = []
    seed_rows = []
    for seed in SEEDS:
        print(f"Training MLP seed={seed}", flush=True)
        pred_local, diagnostics = fit_predict_mlp(train_x, train_y, weights, test_x.astype(np.float32), seed)
        pred_locals.append(pred_local)
        seed_rows.append(diagnostics)
        print(
            f"  seed={seed} n_iter={diagnostics['n_iter']:.0f} "
            f"val_score={diagnostics['best_validation_score']:.6f} loss={diagnostics['loss']:.6f}",
            flush=True,
        )

    pred_local = np.mean(pred_locals, axis=0)
    current_best = read_submission_coords(args.submission_dir / CURRENT_BEST)
    mlp_preds = {name: direct_prediction(test_coords, pred_local, mult) for name, mult in MULTS.items()}

    output_rows = []
    written: list[Path] = []
    for rank, spec in enumerate(BLEND_SPECS, start=1):
        mlp_pred = mlp_preds[spec.mult_name]
        move = spec.blend * (mlp_pred - current_best)
        if spec.cap is not None:
            move = clip_vectors(move, spec.cap)
        pred = current_best + move
        path = args.submission_dir / f"mlpseq_rank{rank}_{slug(spec.name)}.csv"
        write_submission(sample_submission, pred, path)
        written.append(path)
        output_rows.append(
            {
                "rank": rank,
                "submission": path.name,
                "spec": spec.name,
                "mult": spec.mult_name,
                "blend": spec.blend,
                "cap": spec.cap if spec.cap is not None else "none",
                **delta_summary(mlp_pred, current_best, "mlp_vs_current_best"),
                **delta_summary(pred, current_best, "submission_vs_current_best"),
            }
        )

    report = [
        "# 2026-05-20 MLP Sequence Pseudo Blend",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- data_dir: `{data_dir}`",
        f"- current_best: `{CURRENT_BEST} = {CURRENT_BEST_SCORE:.5f}`",
        "- idea: leave tree/physics correction family and train a small neural MLP on normalized sequence features with temporal pseudo-supervision.",
        f"- pseudo_cutoffs: `{PSEUDO_CUTOFFS}`",
        f"- pseudo_weight: `{PSEUDO_WEIGHT}`",
        f"- seeds: `{SEEDS}`",
        f"- generated_outputs: `{[str(path) for path in written]}`",
        "",
        "## Seed Diagnostics",
        "",
        dataframe_to_markdown(pd.DataFrame(seed_rows)),
        "",
        "## Outputs",
        "",
        dataframe_to_markdown(pd.DataFrame(output_rows)),
        "",
        "## Notes",
        "",
        "- This is intentionally a new model-family axis after mirror and multi-curvature failed on public.",
        "- The outputs are small blends or capped moves from the current best to control public risk.",
        "- If even the 8-12% blends drop, this neural bias is not complementary enough and should be deprioritized.",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}", flush=True)
    for path in written:
        print(f"Wrote submission: {path}", flush=True)


if __name__ == "__main__":
    main()
