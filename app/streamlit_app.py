"""Streamlit demo for the C-MAPSS RUL prediction project.

Pick an engine, inspect its sensor trends, and see the live RUL prediction
curve (with the maintenance-alert threshold) from any trained model.

Run from the project root::

    streamlit run app/streamlit_app.py

Requires trained artifacts in ``models/`` (see README: run
``python -m src.train --fd 1`` first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALERT_THRESHOLD, KEPT_SENSORS, MODELS_DIR, RUL_MAX  # noqa: E402
from src.data_loader import engine_units, load_raw, load_rul  # noqa: E402
from src.features import build_rul_target  # noqa: E402
from src.inference import predict_rul_curve  # noqa: E402

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "linear_regression": "Linear Regression",
    "gru": "GRU",
    "lstm": "LSTM",
}

st.set_page_config(page_title="Aircraft Engine RUL", layout="wide")
st.title("Aircraft Engine Predictive Maintenance - RUL Prediction")
st.caption("NASA C-MAPSS turbofan degradation dataset")


@st.cache_resource(show_spinner=False)
def load_data(fd: int):
    return load_raw("train", fd), load_raw("test", fd)


def main() -> None:
    fd = st.sidebar.selectbox("Dataset subset", [1, 2, 3, 4], index=0, format_func=lambda x: f"FD00{x}")
    train, test = load_data(fd)

    model_key = st.sidebar.selectbox(
        "Model", list(MODEL_LABELS.keys()), format_func=MODEL_LABELS.get
    )
    threshold = st.sidebar.slider(
        "Alert threshold (cycles)", 0, 50, ALERT_THRESHOLD
    )

    # ---- engine selector ----------------------------------------------------
    all_units = np.sort(np.unique(np.concatenate([train["unit"].unique(), test["unit"].unique()])))
    split_label = {u: ("train" if u in set(train["unit"].unique()) else "test") for u in all_units}
    unit = st.sidebar.selectbox(
        "Engine unit",
        all_units.tolist(),
        format_func=lambda u: f"unit {u} ({split_label[u]})",
    )

    is_train = unit in set(train["unit"].unique())
    df_engine = train[train["unit"] == unit] if is_train else test[test["unit"] == unit]
    df_engine = df_engine.sort_values("cycle").reset_index(drop=True)

    # ---- model availability -------------------------------------------------
    has_models = (MODELS_DIR / f"preprocessing_fd00{fd}.joblib").exists()

    # ---- sensor trends ------------------------------------------------------
    st.subheader(f"Engine {unit} - sensor trends ({split_label[unit]})")
    normalize = st.checkbox("Min-max normalise sensors for display", value=True)
    sensors = df_engine[KEPT_SENSORS].copy()
    if normalize:
        sensors = (sensors - sensors.min()) / (sensors.max() - sensors.min() + 1e-12)

    grid = st.columns(3)
    sensor_chart = st.empty()
    sel = st.selectbox("Sensor", KEPT_SENSORS, index=KEPT_SENSORS.index("s2"))
    chart_df = pd.DataFrame({"cycle": df_engine["cycle"], "value": sensors[sel]})
    st.line_chart(chart_df.set_index("cycle"))

    with st.expander("Show all sensors as a grid"):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 3, figsize=(14, 12), sharex=True)
        for ax, s in zip(axes.ravel(), KEPT_SENSORS):
            ax.plot(df_engine["cycle"], sensors[s], lw=1.5)
            ax.set_title(s)
            ax.set_xlabel("cycle")
        for ax in axes.ravel()[len(KEPT_SENSORS):]:
            ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)

    # ---- RUL prediction curve ----------------------------------------------
    st.subheader("Remaining Useful Life prediction")
    if not has_models:
        st.warning(
            "No trained models found in `models/`. Run `python -m src.train --fd "
            f"{fd}` first, then refresh."
        )
        return

    curve = predict_rul_curve(df_engine, model_key, fd=fd)
    curve = curve.dropna(subset=["predicted_rul"]).reset_index(drop=True)

    plot_df = pd.DataFrame(
        {"cycle": curve["cycle"], "predicted RUL": curve["predicted_rul"]}
    )
    if is_train:
        true_rul = build_rul_target(train).loc[df_engine.index].clip(upper=RUL_MAX)
        plot_df["true RUL"] = true_rul.values
    else:
        rul_file = load_rul(fd)
        final_true = rul_file[int(unit) - 1]
        st.info(f"True RUL at final observed cycle: **{final_true:.0f} cycles**")

    plot_df["alert threshold"] = threshold
    st.line_chart(plot_df.set_index("cycle"))

    alerts = curve[curve["predicted_rul"] < threshold]
    if alerts.empty:
        st.success("No maintenance alert raised within the observed window.")
    else:
        first_alert = int(alerts["cycle"].iloc[0])
        st.error(
            f"Maintenance alert: predicted RUL drops below {threshold} cycles "
            f"at cycle {first_alert}."
        )

    st.caption(
        "Predicted RUL is clipped to [0, 125] (piecewise-linear target). "
        "Sequence models only produce predictions once a full input window is available."
    )


if __name__ == "__main__":
    main()
