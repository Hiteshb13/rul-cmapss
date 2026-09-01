"""Generate the results comparison table and key figures for the README.

Loads the saved per-model test predictions (produced by ``python -m src.train``)
and produces:

* a printed Markdown comparison table (RMSE + NASA score),
* ``reports/results_comparison.png`` - RMSE / NASA score bar chart,
* ``reports/results_error_analysis.png`` - prediction error vs true RUL,
* ``reports/results_pred_curve.png`` - live RUL curve on a held-out engine.

Usage
-----
    python scripts/report.py --fd 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALERT_THRESHOLD, REPORTS_DIR
from src.data_loader import load_raw
from src.evaluate import error_analysis, nasa_score, rmse
from src.features import build_rul_target, split_by_unit
from src.inference import predict_rul_curve
from src.seeds import SEED, set_seed

MODEL_ORDER = ["linear_regression", "random_forest", "xgboost", "gru", "lstm"]
MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gru": "GRU",
    "lstm": "LSTM",
}


def _load_pred(fd: int, name: str) -> np.ndarray | None:
    path = REPORTS_DIR / f"pred_{name}_fd00{fd}.npy"
    return np.load(path) if path.exists() else None


def main(fd: int = 1) -> int:
    set_seed()
    truth = np.load(REPORTS_DIR / f"truth_fd00{fd}.npy")

    rows = []
    for name in MODEL_ORDER:
        pred = _load_pred(fd, name)
        if pred is None:
            continue
        rows.append(
            {
                "model": MODEL_LABELS[name],
                "rmse": round(rmse(truth, pred), 2),
                "nasa_score": round(nasa_score(truth, pred), 2),
                "mae": round(float(np.mean(np.abs(pred - truth))), 2),
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_markdown(index=False))
    table.to_csv(REPORTS_DIR / f"results_fd00{fd}.csv", index=False)

    # ------------------------------------------------------------------ figure 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(table["model"], table["rmse"], color="steelblue")
    axes[0].set_title("RMSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(table["model"], table["nasa_score"], color="crimson")
    axes[1].set_title("NASA asymmetric score (lower is better)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle(f"Model comparison - C-MAPSS FD00{fd}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "results_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ figure 2
    xgb = _load_pred(fd, "xgboost")
    gru = _load_pred(fd, "gru")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, pred, label in [(axes[0], xgb, "XGBoost"), (axes[1], gru, "GRU")]:
        err = pred - truth
        ax.scatter(truth, err, s=22, alpha=0.6)
        ax.axhline(0, color="black", lw=1)
        ax.axvline(ALERT_THRESHOLD, color="orange", ls="--", lw=1, label="alert threshold")
        ax.set_xlabel("true RUL (cycles)")
        ax.set_ylabel("prediction error (pred - true)")
        ax.set_title(f"{label} - error vs true RUL")
        ax.legend()
    fig.suptitle("Error analysis: over/under prediction by life stage")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "results_error_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ figure 3
    train = load_raw("train", fd)
    rul = build_rul_target(train)
    units = np.sort(train["unit"].unique())
    _, val_units, _ = split_by_unit(units, val_ratio=0.15, test_ratio=0.0, seed=SEED)
    demo_unit = int(val_units[len(val_units) // 2])

    fig, ax = plt.subplots(figsize=(9, 4))
    engine = train[train["unit"] == demo_unit]
    ax.plot(engine["cycle"], rul[engine.index], color="black", lw=2, label="true RUL")
    for name, color in [("xgboost", "tab:green"), ("gru", "tab:purple")]:
        curve = predict_rul_curve(engine, name, fd=fd)
        ax.plot(curve["cycle"], curve["predicted_rul"], color=color, lw=1.5, label=MODEL_LABELS[name])
    ax.axhline(ALERT_THRESHOLD, color="orange", ls="--", lw=1, label=f"alert threshold ({ALERT_THRESHOLD})")
    ax.set_xlabel("cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"RUL prediction curve - held-out engine {demo_unit} (FD00{fd})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "results_pred_curve.png", bbox_inches="tight")
    plt.close(fig)

    # error analysis text for README
    print("\nError analysis (XGBoost):")
    print(json.dumps(error_analysis(truth, xgb), indent=2))
    print("\nError analysis (GRU):")
    print(json.dumps(error_analysis(truth, gru), indent=2))

    print(f"\nFigures written to {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(main(args.fd))
