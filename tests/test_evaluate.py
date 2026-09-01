"""Tests for ``src.evaluate`` (NASA scoring + alert layer)."""

from __future__ import annotations

import numpy as np

from src.evaluate import (
    alert_metrics,
    error_analysis,
    nasa_score,
    nasa_score_samples,
    rmse,
)


def test_rmse_basic():
    assert np.isclose(rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])), np.sqrt(12.5))


def test_nasa_score_penalises_late_more():
    true = np.array([50.0])
    early_pred = np.array([40.0])  # d = -10 (early, safe)
    late_pred = np.array([60.0])  # d = +10 (late, risky)

    s_early = nasa_score(true, early_pred)
    s_late = nasa_score(true, late_pred)

    # same magnitude of error, but late must be penalised more
    assert s_late > s_early


def test_nasa_score_perfect_is_zero():
    true = np.array([10.0, 20.0, 30.0])
    assert nasa_score(true, true) == 0.0
    assert np.allclose(nasa_score_samples(true, true), 0.0)


def test_nasa_score_known_values():
    # d = -13 (early) -> exp(13/13) - 1 = e - 1 ; d = +10 (late) -> e - 1
    true = np.array([0.0, 0.0])
    pred = np.array([-13.0, 10.0])
    e = np.e
    expected = np.array([e - 1, e - 1])
    assert np.allclose(nasa_score_samples(true, pred), expected)


def test_alert_metrics_full_recall():
    true = np.array([5.0, 10.0, 40.0])
    pred = np.array([4.0, 12.0, 45.0])
    metrics = alert_metrics(true, pred, threshold=30)
    assert metrics["tp"] == 2  # both low-RUL engines correctly flagged
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_error_analysis_shape():
    out = error_analysis(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert "over_estimate_pct" in out and "mean_error" in out
