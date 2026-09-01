"""Inference helpers: load trained artifacts and predict RUL per cycle.

Used by the Streamlit app and the results report to produce the live RUL
prediction curve for a chosen engine without retraining.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch

from .config import MODELS_DIR
from .data_loader import get_engine_trajectory
from .features import add_degradation_features, add_rolling_features
from .models import RULSequenceModel, predict_sequences

BASELINE_NAMES = ("linear_regression", "random_forest", "xgboost")


def _baseline_predictor(fd: int, name: str):
    """Return a per-cycle baseline predictor bound to a saved model."""
    preprocess = joblib.load(MODELS_DIR / f"preprocessing_fd00{fd}.joblib")
    model = joblib.load(MODELS_DIR / f"{name}_fd00{fd}.joblib")
    normalizer = preprocess["normalizer"]
    feature_cols = preprocess["feature_cols"]

    def predict(df_engine: pd.DataFrame) -> np.ndarray:
        norm = normalizer.transform(df_engine)
        feat = add_rolling_features(norm)
        feat = add_degradation_features(feat)
        X = feat[feature_cols].to_numpy(dtype=float)
        return model.predict(X)

    return predict


def _sequence_predictor(fd: int, rnn_type: str):
    """Return a per-cycle sequence predictor bound to a saved model."""
    preprocess = joblib.load(MODELS_DIR / f"preprocessing_fd00{fd}.joblib")
    normalizer = preprocess["normalizer"]
    window = int(preprocess["window"])
    sensors = preprocess["kept_sensors"]

    model = RULSequenceModel(
        n_features=len(sensors),
        hidden_size=preprocess["hidden_size"],
        num_layers=preprocess["num_layers"],
        rnn_type=preprocess["rnn_type"],
    )
    model.load_state_dict(torch.load(MODELS_DIR / f"sequence_{rnn_type}_fd00{fd}.pt", map_location="cpu"))
    model.eval()

    def predict(df_engine: pd.DataFrame) -> np.ndarray:
        norm = normalizer.transform(df_engine)
        values = norm.sort_values("cycle")[sensors].to_numpy(dtype=float)
        n = values.shape[0]
        preds = np.full(n, np.nan)
        if n < window:
            return preds
        windows = np.stack([values[i - window + 1 : i + 1] for i in range(window - 1, n)])
        preds[window - 1 :] = predict_sequences(model, windows, device="cpu")
        return preds

    return predict


def predict_rul_curve(
    df_engine: pd.DataFrame,
    model_name: str,
    fd: int = 1,
    clip: int | None = 125,
) -> pd.DataFrame:
    """Predict RUL at every cycle of a single engine.

    Parameters
    ----------
    df_engine : pd.DataFrame
        Raw trajectory of one engine (from ``data_loader.load_raw``).
    model_name : str
        One of ``linear_regression``, ``random_forest``, ``xgboost``, ``gru``,
        or ``lstm``.
    fd : int
        Subset id.
    clip : int, optional
        Clip predictions to ``[0, clip]``. Pass ``None`` to disable.

    Returns
    -------
    pd.DataFrame
        Columns ``cycle`` and ``predicted_rul``, sorted by cycle.
    """
    if model_name in BASELINE_NAMES:
        fn = _baseline_predictor(fd, model_name)
    elif model_name in ("gru", "lstm"):
        fn = _sequence_predictor(fd, model_name)
    else:
        raise ValueError(f"unknown model_name {model_name!r}")

    df = get_engine_trajectory(df_engine, int(df_engine["unit"].iloc[0])).copy()
    raw = fn(df)
    if clip is not None:
        raw = np.clip(raw, 0.0, float(clip))
    return pd.DataFrame({"cycle": df["cycle"].values, "predicted_rul": raw})
