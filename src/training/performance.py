from __future__ import annotations

import time

import numpy as np

# Global constants matching harness.py EVAL_CONFIG
N_CLASSES = 8
ABSTAIN = -1
WRONG_PENALTY = 0.33
EPSILON = 0.01
ALPHA = 0.5
P_EXPONENT = 2


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES
) -> np.ndarray:
    """Computes a confusion matrix including a column for abstentions.

    Rows: True class (0 to n_classes-1).
    Columns: Predicted class (0 to n_classes-1) + the last column representing ABSTAIN (-1).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = np.zeros((n_classes, n_classes + 1), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        col = n_classes if p == ABSTAIN else p
        if 0 <= t < n_classes:
            cm[t, col] += 1
    return cm


def accuracy_all(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes global accuracy (treating ABSTAIN as incorrect)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def accuracy_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES
) -> list[float | None]:
    """Computes accuracy per class (treating ABSTAIN as incorrect)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = []
    for k in range(n_classes):
        mask = y_true == k
        if mask.any():
            out.append(float(np.mean(y_pred[mask] == k)))
        else:
            out.append(None)
    return out


def geomean_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = N_CLASSES,
    wrong_penalty: float = WRONG_PENALTY,
    epsilon: float = EPSILON,
    alpha: float = ALPHA,
    p_exponent: int = P_EXPONENT,
) -> float:
    """Computes the official competition geometric mean score.

    Strictly follows the mathematical formulation:
        Score = (Product_{k=0}^{7} T_k) ** (1/8)
        T_k = max(epsilon, n_k**alpha * (c_k - lambda * w_k) / n_k * (c_k / n_k)**p)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    T_terms = []
    for k in range(n_classes):
        mask_k = y_true == k
        n_k = int(np.sum(mask_k))
        if n_k == 0:
            T_terms.append(epsilon)
            continue

        preds_k = y_pred[mask_k]
        c_k = int(np.sum(preds_k == k))
        # Wrong answers are those that are neither correct (k) nor ABSTAIN (-1)
        w_k = int(np.sum((preds_k != k) & (preds_k != ABSTAIN)))

        accuracy_term = (c_k - wrong_penalty * w_k) / n_k
        precision_term = (c_k / n_k) ** p_exponent
        speed_term = n_k**alpha

        term = speed_term * accuracy_term * precision_term
        T_k = max(epsilon, term)
        T_terms.append(T_k)

    return float(np.prod(T_terms) ** (1.0 / n_classes))


def measure_latency(model, samples: np.ndarray) -> tuple[list[int], list[float]]:
    """Measures single-sample inference latency sequentially, mimicking harness.py."""
    preds = []
    latencies = []
    for x in samples:
        t0 = time.perf_counter()
        pred = model.predict(x)
        latencies.append(time.perf_counter() - t0)
        preds.append(pred)
    return preds, latencies


def latency_stats(latencies: list[float]) -> dict[str, float]:
    """Generates standard descriptive statistics from measured latencies in seconds."""
    lat = np.asarray(latencies)
    return {
        "mean_ms": float(lat.mean() * 1000),
        "p95_ms": float(np.percentile(lat, 95) * 1000),
        "p99_ms": float(np.percentile(lat, 99) * 1000),
        "max_ms": float(lat.max() * 1000),
    }
