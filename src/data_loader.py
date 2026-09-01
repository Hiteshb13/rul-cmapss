"""Loading utilities for the NASA C-MAPSS turbofan data set.

The raw files are space-delimited with no header row. Each row is::

    unit  cycle  setting_1  setting_2  setting_3  s1  s2 ... s21

``unit`` is the engine identifier, ``cycle`` the time index within an engine's
run, the three settings are operational parameters, and ``s1..s21`` are the 21
sensor channels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COLUMN_NAMES, DATA_RAW


def _raw_path(split: str, fd: int) -> Path:
    """Return the filesystem path for a raw C-MAPSS file.

    Parameters
    ----------
    split : {"train", "test"}
        Which split the file belongs to.
    fd : int
        Subset id (1-4).
    """
    valid = {"train", "test"}
    if split not in valid:
        raise ValueError(f"split must be one of {valid}, got {split!r}")
    if fd not in {1, 2, 3, 4}:
        raise ValueError(f"fd must be in {{1, 2, 3, 4}}, got {fd!r}")
    return DATA_RAW / f"{split}_FD00{fd}.txt"


def load_raw(split: str, fd: int = 1) -> pd.DataFrame:
    """Load a raw train/test file as a DataFrame with named columns.

    Parameters
    ----------
    split : {"train", "test"}
        Which split to load.
    fd : int
        Subset id (1-4). Defaults to FD001.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``unit, cycle, setting_1..3, s1..s21``.
    """
    path = _raw_path(split, fd)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="c",
    )
    # CMAPSS unit ids are integers starting at 1.
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_rul(fd: int = 1) -> np.ndarray:
    """Load the ground-truth RUL for the *test* set of a subset.

    Parameters
    ----------
    fd : int
        Subset id (1-4).

    Returns
    -------
    np.ndarray
        1-D array of length ``n_test_units``, the RUL at the final cycle of
        each test engine (in ascending unit order).
    """
    path = DATA_RAW / f"RUL_FD00{fd}.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="c").values.ravel()
    return rul.astype(float)


def engine_units(df: pd.DataFrame) -> np.ndarray:
    """Return the sorted unique engine unit ids present in ``df``."""
    return np.sort(df["unit"].unique())


def get_engine_trajectory(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    """Return the run-to-failure trajectory of a single engine.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`load_raw`.
    unit : int
        Engine unit id.

    Returns
    -------
    pd.DataFrame
        Rows for ``unit``, sorted by ``cycle``.
    """
    if unit not in df["unit"].values:
        raise ValueError(f"unit {unit} not present in data")
    return df[df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
