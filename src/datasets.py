"""Sequence windowing utilities for the sequence (GRU/LSTM) models.

Converts per-cycle sensor DataFrames into sliding windows of fixed length.
Windows are built *within* each engine unit so that no window straddles two
engines' histories.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def build_windows(
    df: pd.DataFrame,
    rul: pd.Series,
    window_size: int,
    feature_cols: Sequence[str],
    units: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Build every sliding window of ``window_size`` cycles.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with ``unit``, ``cycle`` and ``feature_cols``.
    rul : pd.Series
        RUL label aligned with ``df`` index.
    window_size : int
        Number of cycles per input window.
    feature_cols : sequence of str
        Columns used as model inputs.
    units : sequence of int, optional
        Restrict to these engine units.

    Returns
    -------
    (X, y, meta)
        ``X`` shape ``(n_windows, window_size, n_features)``, ``y`` the RUL at
        the final cycle of each window, and ``meta`` a list of
        ``(unit, final_cycle)`` identifying each window.
    """
    X: list[np.ndarray] = []
    y: list[float] = []
    meta: list[tuple[int, int]] = []

    for unit, grp in df.groupby("unit"):
        if units is not None and unit not in set(units):
            continue
        grp = grp.sort_values("cycle")
        values = grp[feature_cols].to_numpy(dtype=float)
        labels = rul.loc[grp.index].to_numpy(dtype=float)
        cycles = grp["cycle"].to_numpy()
        if values.shape[0] < window_size:
            continue
        for end in range(window_size - 1, values.shape[0]):
            X.append(values[end - window_size + 1 : end + 1])
            y.append(labels[end])
            meta.append((int(unit), int(cycles[end])))

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), meta


def build_last_windows(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: Sequence[str],
    units: Sequence[int],
    rul: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    """Build the single final window of each engine (for val/test prediction).

    Parameters
    ----------
    df : pd.DataFrame
        Frame with ``unit``, ``cycle`` and ``feature_cols``.
    window_size : int
        Number of cycles per input window.
    feature_cols : sequence of str
        Columns used as model inputs.
    units : sequence of int
        Engine units to build final windows for.
    rul : pd.Series, optional
        RUL labels aligned with ``df`` index. If omitted, ``y`` is ``None``.

    Returns
    -------
    (X, y, unit_ids)
        ``X`` shape ``(n_units, window_size, n_features)``, ``y`` the RUL at
        each engine's last cycle (or ``None``), and ``unit_ids`` the engine ids.
    """
    X: list[np.ndarray] = []
    y: list[float] = []
    unit_ids: list[int] = []

    for unit in units:
        grp = df[df["unit"] == unit].sort_values("cycle")
        values = grp[feature_cols].to_numpy(dtype=float)
        if values.shape[0] < window_size:
            raise ValueError(
                f"engine {unit} has {values.shape[0]} cycles, fewer than "
                f"window_size={window_size}"
            )
        X.append(values[-window_size:])
        if rul is not None:
            y.append(float(rul.loc[grp.index].to_numpy(dtype=float)[-1]))
        unit_ids.append(int(unit))

    y_arr: np.ndarray | None = np.asarray(y, dtype=float) if rul is not None else None
    return np.asarray(X, dtype=float), y_arr, unit_ids
