from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mosquito_trajectory.data import TrajectorySample


TARGET_DELTA_MS = 80.0
PredictionFn = Callable[[TrajectorySample, float], np.ndarray]


def _relative_time(sample: TrajectorySample, window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    timesteps = sample.timesteps_ms
    coords = sample.coords

    if window is not None:
        timesteps = timesteps[-window:]
        coords = coords[-window:]

    return timesteps - timesteps[-1], coords


def constant_position(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    return sample.coords[-1].copy()


def constant_velocity_last(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    dt_ms = sample.timesteps_ms[-1] - sample.timesteps_ms[-2]
    velocity = (sample.coords[-1] - sample.coords[-2]) / dt_ms
    return sample.coords[-1] + velocity * delta_ms


def constant_acceleration_last(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    if sample.coords.shape[0] < 3:
        return constant_velocity_last(sample, delta_ms)

    dt_prev = sample.timesteps_ms[-2] - sample.timesteps_ms[-3]
    dt_last = sample.timesteps_ms[-1] - sample.timesteps_ms[-2]
    velocity_prev = (sample.coords[-2] - sample.coords[-3]) / dt_prev
    velocity_last = (sample.coords[-1] - sample.coords[-2]) / dt_last
    acceleration = (velocity_last - velocity_prev) / ((dt_prev + dt_last) * 0.5)
    return sample.coords[-1] + velocity_last * delta_ms + 0.5 * acceleration * delta_ms**2


def polynomial_fit(
    sample: TrajectorySample,
    delta_ms: float = TARGET_DELTA_MS,
    *,
    degree: int,
    window: int | None = None,
    weighted_recent: bool = False,
) -> np.ndarray:
    t, coords = _relative_time(sample, window)
    degree = min(degree, len(t) - 1)
    if degree <= 0:
        return constant_position(sample, delta_ms)

    weights = None
    if weighted_recent:
        weights = np.linspace(0.4, 1.0, len(t))

    pred = np.empty(3, dtype=float)
    for axis in range(3):
        coeffs = np.polyfit(t, coords[:, axis], deg=degree, w=weights)
        pred[axis] = np.polyval(coeffs, delta_ms)
    return pred


def velocity_fit_all(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    return polynomial_fit(sample, delta_ms, degree=1)


def velocity_fit_recent5(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    return polynomial_fit(sample, delta_ms, degree=1, window=5)


def velocity_fit_weighted(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    return polynomial_fit(sample, delta_ms, degree=1, weighted_recent=True)


def poly2_all(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    return polynomial_fit(sample, delta_ms, degree=2)


def poly2_recent5(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    return polynomial_fit(sample, delta_ms, degree=2, window=5)


def poly2_weighted(sample: TrajectorySample, delta_ms: float = TARGET_DELTA_MS) -> np.ndarray:
    return polynomial_fit(sample, delta_ms, degree=2, weighted_recent=True)


BASELINE_METHODS: dict[str, PredictionFn] = {
    "position": constant_position,
    "velocity_last": constant_velocity_last,
    "acceleration_last": constant_acceleration_last,
    "velocity_fit_all": velocity_fit_all,
    "velocity_fit_recent5": velocity_fit_recent5,
    "velocity_fit_weighted": velocity_fit_weighted,
    "poly2_all": poly2_all,
    "poly2_recent5": poly2_recent5,
    "poly2_weighted": poly2_weighted,
}


def predict_many(
    samples: dict[str, TrajectorySample],
    ids: list[str],
    method_name: str,
    delta_ms: float = TARGET_DELTA_MS,
) -> np.ndarray:
    if method_name not in BASELINE_METHODS:
        raise KeyError(f"unknown method '{method_name}', choose from {sorted(BASELINE_METHODS)}")
    fn = BASELINE_METHODS[method_name]
    return np.vstack([fn(samples[sample_id], delta_ms) for sample_id in ids])

