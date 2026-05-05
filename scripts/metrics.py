import numpy as np


def r_hit(pred, true, radius=0.01):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    distance = np.linalg.norm(pred - true, axis=-1)
    return float(np.mean(distance <= radius))


def mean_distance(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    distance = np.linalg.norm(pred - true, axis=-1)
    return float(np.mean(distance))

