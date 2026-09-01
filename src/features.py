"""Feature engineering for the C-MAPSS RUL prediction task.

All functions operate on a DataFrame produced by :func:`data_loader.load_raw`
(columns ``unit, cycle, setting_1..3, s1..s21``). The engine ``unit`` column is
always respected so that no statistic computed on one engine's history ever
leaks into another engine (rolling windows never cross unit boundaries).

Functions
---------
- :func:`build_rul_target`: piecewise-linear RUL label (training only).
- :func:`add_rolling_features`: per-sensor rolling statistics + trend slope.
- :func:`add_degradation_features`: change-from-initial and life-fraction.
- :class:`ConditionNormalizer`: per-operating-condition sensor z-scoring
  (needed for the multi-condition subsets FD002-FD004).
- :func:`split_by_unit`: leakage-free train/val/test split on engine units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import (
    KEPT_SENSORS,
    N_OPERATING_CONDITIONS,
    RUL_MAX,
    SENSOR_COLS,
    SETTING_COLS,
)
from .seeds import SEED


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def build_rul_target(df: pd.DataFrame, rul_max: int = RUL_MAX) -> pd.Series:
    """Build the piecewise-linear RUL label for run-to-failure *training* data.

    ``RUL(cycle) = max_cycle(unit) - cycle``, capped at ``rul_max`` so that the
    flat, healthy early segment of each engine's life maps to a constant target
    (the degradation signal only appears near end-of-life).

    Parameters
    ----------
    df : pd.DataFrame
        Training data (run-to-failure).
    rul_max : int
        Upper cap on the RUL label.

    Returns
    -------
    pd.Series
        RUL label, same index as ``df``.
    """
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - df["cycle"]
    return rul.clip(upper=rul_max).astype(float)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------
def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of ``values`` over a trailing window of ``window`` points.

    For equally-spaced time the OLS slope is a fixed weighted sum of the window
    values, computed here as a single convolution per series. The first
    ``window - 1`` positions are NaN (not enough history).
    """
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n < 2 or window < 2 or window > n:
        return out

    weights = np.arange(window, dtype=float) - (window - 1) / 2.0
    norm = float(np.dot(weights, weights))
    if norm == 0:
        return out

    conv = np.convolve(values, weights[::-1], mode="valid")
    out[window - 1:] = conv / norm
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (10, 25, 50),
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling mean/std/min/max and trend slope for each sensor.

    Parameters
    ----------
    df : pd.DataFrame
        Raw C-MAPSS frame (must contain ``unit`` and ``cycle``).
    windows : tuple of int
        Rolling window sizes.
    sensors : list of str, optional
        Sensors to transform. Defaults to :data:`KEPT_SENSORS`.

    Returns
    -------
    pd.DataFrame
        Input frame with additional rolling-statistic columns appended.
    """
    sensors = sensors if sensors is not None else KEPT_SENSORS
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    features: dict[str, pd.Series] = {}
    for sensor in sensors:
        grouped = df.groupby("unit")[sensor]
        for w in windows:
            roll = grouped.rolling(w, min_periods=1)
            features[f"{sensor}_rollmean_{w}"] = (
                roll.mean().reset_index(level=0, drop=True)
            )
            features[f"{sensor}_rollstd_{w}"] = (
                roll.std().reset_index(level=0, drop=True)
            )
            features[f"{sensor}_rollmin_{w}"] = (
                roll.min().reset_index(level=0, drop=True)
            )
            features[f"{sensor}_rollmax_{w}"] = (
                roll.max().reset_index(level=0, drop=True)
            )
            features[f"{sensor}_slope_{w}"] = grouped.transform(
                lambda s: _rolling_slope(s.to_numpy(), w)
            )

    out = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
    # std of a single point and slope of a too-short window are NaN; they carry
    # no degradation signal in early life, so zero-fill is appropriate.
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# Degradation trend features
# ---------------------------------------------------------------------------
def add_degradation_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Add change-from-initial and life-fraction degradation features.

    ``*_delta_init`` is the absolute drift from the sensor's first reading;
    ``*_ratio_init`` is the relative drift. ``life_fraction`` is the fraction
    of the engine's life already elapsed (a strong, cheap proxy for RUL).

    Parameters
    ----------
    df : pd.DataFrame
        Raw C-MAPSS frame.
    sensors : list of str, optional
        Sensors to transform. Defaults to :data:`KEPT_SENSORS`.

    Returns
    -------
    pd.DataFrame
        Input frame with degradation features appended.
    """
    sensors = sensors if sensors is not None else KEPT_SENSORS
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    first = df.groupby("unit")[sensors].transform("first")

    features: dict[str, pd.Series] = {}
    for sensor in sensors:
        features[f"{sensor}_delta_init"] = df[sensor] - first[sensor]
        denominator = first[sensor].replace(0.0, np.nan)
        features[f"{sensor}_ratio_init"] = df[sensor] / denominator

    features["life_fraction"] = (
        df["cycle"] / df.groupby("unit")["cycle"].transform("max")
    )

    out = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# Per-operating-condition normalization
# ---------------------------------------------------------------------------
class ConditionNormalizer:
    """Normalise sensors within each operating-condition regime.

    FD002/FD004 contain six distinct operating regimes encoded by the three
    settings. If sensors are z-scored globally, condition changes masquerade as
    degradation. This normalizer first clusters the settings (KMeans) into
    regimes, then z-scores each sensor using the regime-specific mean/std
    learned on the training set only.

    Usage
    -----
        normalizer = ConditionNormalizer(n_clusters=6).fit(train_df)
        train_norm = normalizer.transform(train_df)
        test_norm = normalizer.transform(test_df)
    """

    def __init__(
        self,
        n_clusters: int = N_OPERATING_CONDITIONS,
        sensors: list[str] | None = None,
        random_state: int = SEED,
    ) -> None:
        self.n_clusters = n_clusters
        self.sensors = sensors if sensors is not None else SENSOR_COLS
        self.random_state = random_state
        self._kmeans: KMeans | None = None
        self._stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df: pd.DataFrame) -> "ConditionNormalizer":
        """Learn regime centroids and per-regime sensor mean/std."""
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(df[SETTING_COLS].to_numpy())

        labels = self._kmeans.labels_
        self._stats = {}
        for cluster in range(self.n_clusters):
            mask = labels == cluster
            values = df.loc[mask, self.sensors].to_numpy(dtype=float)
            if values.shape[0] == 0:
                mean = np.zeros(len(self.sensors))
                std = np.ones(len(self.sensors))
            else:
                mean = values.mean(axis=0)
                std = values.std(axis=0)
                std[std == 0] = 1.0
            self._stats[cluster] = (mean, std)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned normalization to a frame (no re-fitting)."""
        if self._kmeans is None:
            raise RuntimeError("ConditionNormalizer must be fit before transform")

        labels = self._kmeans.predict(df[SETTING_COLS].to_numpy())
        values = df[self.sensors].to_numpy(dtype=float)

        normalized = np.empty_like(values)
        for cluster, (mean, std) in self._stats.items():
            mask = labels == cluster
            normalized[mask] = (values[mask] - mean) / std

        out = df.copy()
        out[self.sensors] = normalized
        return out


# ---------------------------------------------------------------------------
# Leakage-free split
# ---------------------------------------------------------------------------
def split_by_unit(
    units: np.ndarray | list[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split *engine units* into train/val/test with no unit overlap.

    Splitting by unit (not by row/cycle) guarantees that an engine's entire
    life history lives in exactly one partition, which is the correct
    time-respecting protocol for prognostics.

    Parameters
    ----------
    units : array-like
        Unique engine unit ids.
    val_ratio, test_ratio : float
        Fraction of units allocated to validation / test.
    seed : int
        Random seed for the shuffle.

    Returns
    -------
    (train_units, val_units, test_units)
        Sorted arrays of unit ids.
    """
    units = np.asarray(sorted(set(units)))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)

    n = units.shape[0]
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    test_units = shuffled[:n_test]
    val_units = shuffled[n_test : n_test + n_val]
    train_units = shuffled[n_test + n_val :]

    return (
        np.sort(train_units),
        np.sort(val_units),
        np.sort(test_units),
    )
