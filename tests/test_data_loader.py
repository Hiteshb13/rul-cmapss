"""Tests for ``src.data_loader``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import COLUMN_NAMES, DATA_RAW
from src.data_loader import engine_units, get_engine_trajectory, load_raw, load_rul


def test_load_raw_columns_and_dtypes(sample_frame):
    assert list(sample_frame.columns) == COLUMN_NAMES
    assert sample_frame["unit"].dtype == int
    assert sample_frame["cycle"].dtype == int


def test_engine_units_sorted_unique(sample_frame):
    units = engine_units(sample_frame)
    assert units.tolist() == [1, 2, 3]
    assert np.all(np.diff(units) > 0)


def test_get_engine_trajectory_returns_sorted_single_unit(sample_frame):
    traj = get_engine_trajectory(sample_frame, unit=2)
    assert traj["unit"].nunique() == 1
    assert traj["unit"].iloc[0] == 2
    assert traj["cycle"].is_monotonic_increasing
    assert len(traj) == 6


def test_get_engine_trajectory_raises_on_missing_unit(sample_frame):
    with pytest.raises(ValueError):
        get_engine_trajectory(sample_frame, unit=999)


@pytest.mark.skipif(not (DATA_RAW / "train_FD001.txt").exists(), reason="full data not downloaded")
def test_load_raw_fd001_real():
    df = load_raw("train", fd=1)
    assert df.shape[1] == 26
    assert df["unit"].nunique() == 100
    assert df["cycle"].min() == 1


@pytest.mark.skipif(not (DATA_RAW / "RUL_FD001.txt").exists(), reason="full data not downloaded")
def test_load_rul_fd001_real():
    rul = load_rul(fd=1)
    assert isinstance(rul, np.ndarray)
    assert rul.shape == (100,)
    assert (rul > 0).all()
