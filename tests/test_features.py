"""Tests for ``src.features``."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import (
    ConditionNormalizer,
    _rolling_slope,
    add_degradation_features,
    add_rolling_features,
    build_rul_target,
    split_by_unit,
)


# ---------------------------------------------------------------------------
# RUL target
# ---------------------------------------------------------------------------
def test_build_rul_target_piecewise(sample_frame):
    rul = build_rul_target(sample_frame, rul_max=125)
    # unit has max cycle 6 -> RUL(cycle=1) = 5, RUL(cycle=6) = 0
    unit1 = sample_frame[sample_frame["unit"] == 1]
    assert rul.loc[unit1.index[0]] == 5
    assert rul.loc[unit1.index[-1]] == 0


def test_build_rul_target_caps(sample_frame):
    rul = build_rul_target(sample_frame, rul_max=2)
    assert rul.max() == 2


# ---------------------------------------------------------------------------
# Split (no leakage)
# ---------------------------------------------------------------------------
def test_split_by_unit_no_overlap_and_sizes(sample_frame):
    units = np.arange(1, 101)
    train, val, test = split_by_unit(units, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(units)


def test_split_by_unit_deterministic(sample_frame):
    units = np.arange(1, 101)
    a = split_by_unit(units, seed=7)
    b = split_by_unit(units, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------
def test_add_rolling_features_columns_and_no_nan(sample_frame):
    out = add_rolling_features(sample_frame, windows=(3,))
    assert out.shape[0] == sample_frame.shape[0]
    assert "s2_rollmean_3" in out.columns
    assert "s2_rollstd_3" in out.columns
    assert "s2_slope_3" in out.columns
    assert not out.isna().any().any()


def test_rolling_mean_correct(sample_frame):
    out = add_rolling_features(sample_frame, windows=(3,))
    unit1 = out[out["unit"] == 1].reset_index(drop=True)
    expected = unit1["s2"].rolling(3, min_periods=1).mean()
    assert np.allclose(unit1["s2_rollmean_3"], expected)


def test_rolling_slope_linear():
    values = np.arange(10, dtype=float)
    slope = _rolling_slope(values, window=5)
    assert np.isnan(slope[:4]).all()
    assert np.allclose(slope[4:], 1.0)


def test_rolling_slope_respects_unit_boundary(sample_frame):
    # slope computed per-unit must equal the slope of each unit's series
    # computed independently (i.e. no values leak across engine boundaries)
    out = add_rolling_features(sample_frame, windows=(3,))

    expected = np.concatenate(
        [_rolling_slope(grp["s2"].to_numpy(), 3) for _, grp in sample_frame.groupby("unit")]
    )
    computed = out["s2_slope_3"].to_numpy()
    expected = np.nan_to_num(expected)  # add_rolling_features zero-fills NaN
    assert np.allclose(computed, expected)


# ---------------------------------------------------------------------------
# Degradation features
# ---------------------------------------------------------------------------
def test_degradation_features_delta_and_life_fraction(sample_frame):
    out = add_degradation_features(sample_frame)
    unit1 = out[out["unit"] == 1].reset_index(drop=True)
    assert np.allclose(unit1["s2_delta_init"].iloc[0], 0.0)
    assert np.isclose(unit1["life_fraction"].iloc[-1], 1.0)
    assert "s2_ratio_init" in out.columns


# ---------------------------------------------------------------------------
# Condition normalizer
# ---------------------------------------------------------------------------
def test_condition_normalizer_fit_transform(sample_frame):
    norm = ConditionNormalizer(n_clusters=1).fit(sample_frame)
    out = norm.transform(sample_frame)
    assert out.shape == sample_frame.shape
    # a sensor with real variance should be ~unit std after z-scoring
    assert np.isclose(out["s2"].std(), 1.0, atol=0.2)


def test_condition_normalizer_requires_fit(sample_frame):
    norm = ConditionNormalizer(n_clusters=1)
    with pytest.raises(RuntimeError):
        norm.transform(sample_frame)
