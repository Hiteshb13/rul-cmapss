"""Evaluation metrics and alert logic for RUL prediction.

Implements the two headline metrics used throughout the project:

* **RMSE** - the standard symmetric regression error.
* **NASA asymmetric scoring function** (Saxena et al., 2008) - penalises
  *late* predictions (over-estimating RUL, i.e. predicting failure later than
  it actually occurs) more heavily than *early* ones, because a late
  maintenance action can mean an in-flight failure, whereas an early action
  only costs unnecessary maintenance.

Also provides a simple maintenance-alert layer: raise an alert whenever the
predicted RUL drops below a configurable action threshold.
"""

from __future__ import annotations

import numpy as np

from .config import ALERT_THRESHOLD, SCORE_A1, SCORE_A2


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error between true and predicted RUL."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score_samples(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample NASA asymmetric score.

    .. math::
        d = \\hat{y} - y, \\quad
        s = \\begin{cases}
            \\exp(-d / 13) - 1 & d < 0 \\text{ (early, safe)} \\\\
            \\exp( d / 10) - 1 & d \\geq 0 \\text{ (late, risky)}
        \\end{cases}

    Returns
    -------
    np.ndarray
        One score per sample (>= 0; higher is worse).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / SCORE_A1) - 1.0, np.exp(d / SCORE_A2) - 1.0)
    return score


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean NASA asymmetric score (the figure reported in C-MAPSS literature)."""
    return float(np.mean(nasa_score_samples(y_true, y_pred)))


def score_rul(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dict of evaluation metrics for a single RUL prediction set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n_samples": int(y_true.shape[0]),
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "bias": float(np.mean(y_pred - y_true)),
    }


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Summarise where predictions over/under-shoot the true RUL.

    Returns
    -------
    dict
        ``over_estimate`` / ``under_estimate`` counts, and RMSE computed within
        early-life / mid-life / end-of-life RUL buckets.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    buckets = [(100, 125), (50, 100), (30, 50), (0, 30)]
    bucket_rmse = {}
    for lo, hi in buckets:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            bucket_rmse[f"rmse_rul_{lo}_{hi}"] = rmse(y_true[mask], y_pred[mask])

    return {
        "over_estimate_pct": float(100 * np.mean(error > 0)),
        "under_estimate_pct": float(100 * np.mean(error <= 0)),
        "mean_error": float(np.mean(error)),
        **bucket_rmse,
    }


def alert_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: int = ALERT_THRESHOLD,
) -> dict[str, float]:
    """Evaluate the maintenance-alert layer.

    An alert fires when ``y_pred < threshold``. True positives are engines
    whose alert fires before they actually reach ``threshold`` cycles of
    remaining life (i.e. predicted early enough to act).

    Returns
    -------
    dict
        Precision / recall / F1 of alerts, plus the mean prediction error at
        the first alert point (a proxy for lead-time accuracy).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_alert = y_true < threshold
    pred_alert = y_pred < threshold

    tp = int(np.sum(true_alert & pred_alert))
    fp = int(np.sum(~true_alert & pred_alert))
    fn = int(np.sum(true_alert & ~pred_alert))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
