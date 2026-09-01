"""Training pipeline: baseline models + GRU/LSTM sequence model.

Reproducible, leakage-free workflow:

1. Load run-to-failure training data and build the piecewise-linear RUL target.
2. Split *engine units* into a tuning split (train/val) - the official test set
   is used only for final scoring.
3. Fit a per-condition normalizer on the tuning-train units only.
4. Train baselines on engineered features and a sequence model on raw windows.
5. Score both against the official test set with RMSE + NASA score, and save
   predictions / model artifacts for the app and README.

Run with::

    python -m src.train --fd 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from .config import (
    DATA_RAW,
    KEPT_SENSORS,
    MODELS_DIR,
    N_OPERATING_CONDITIONS,
    REPORTS_DIR,
    RUL_MAX,
    SETTING_COLS,
)
from .data_loader import engine_units, load_raw, load_rul
from .datasets import build_last_windows, build_windows
from .evaluate import error_analysis, score_rul
from .features import (
    ConditionNormalizer,
    add_degradation_features,
    add_rolling_features,
    build_rul_target,
    split_by_unit,
)
from .models import RULSequenceModel, make_baseline_models, predict_sequences
from .seeds import SEED, set_seed


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def build_baseline_features(
    df: pd.DataFrame,
    normalizer: ConditionNormalizer,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalise sensors, then add rolling + degradation features.

    Returns the feature frame (still containing ``unit``/``cycle``) and the
    list of model input columns.
    """
    norm = normalizer.transform(df)
    feat = add_rolling_features(norm, sensors=KEPT_SENSORS)
    feat = add_degradation_features(feat, sensors=KEPT_SENSORS)
    # ``life_fraction`` is deliberately excluded: it normalises cycle by the
    # engine's TOTAL life, which is unknowable for a live engine at inference
    # time (and always equals 1.0 on a truncated test engine's last cycle).
    drop = ["unit", "cycle", "life_fraction"] + SETTING_COLS
    feature_cols = [c for c in feat.columns if c not in drop]
    return feat, feature_cols


def _clip_rul(values: np.ndarray, upper: int = RUL_MAX) -> np.ndarray:
    """Clip RUL predictions / targets to ``[0, upper]`` (piecewise-linear cap)."""
    return np.clip(np.asarray(values, dtype=float), 0.0, float(upper))


# ---------------------------------------------------------------------------
# Sequence training
# ---------------------------------------------------------------------------
def train_sequence_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    patience: int = 5,
) -> dict:
    """Train the sequence model with early stopping on validation RMSE."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)

    n = X_train_t.shape[0]
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            pred = model(xb).unsqueeze(1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n

        val_pred = predict_sequences(model, X_val, device=device)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        scheduler.step(val_rmse)
        history.append(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  early stopping at epoch {epoch} (best val RMSE {best_val:.2f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_rmse": best_val, "val_history": history, "epochs_run": epoch}


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------
def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, tuple[object, np.ndarray]]:
    """Fit each baseline and return ``(model, test_predictions)`` per name."""
    results: dict[str, tuple[object, np.ndarray]] = {}
    for name, model in make_baseline_models(SEED).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = (model, pred)
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    fd: int = 1,
    window: int = 30,
    epochs: int = 30,
    batch_size: int = 256,
    hidden: int = 64,
    num_layers: int = 2,
    lr: float = 1e-3,
    rnn_type: str = "gru",
    val_ratio: float = 0.15,
    quick: bool = False,
    save: bool = True,
) -> dict:
    """Run the full train + evaluate pipeline for one subset. Returns metrics."""
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print(f"=== FD00{fd} === device={device} window={window}")

    train_df = load_raw("train", fd)
    test_df = load_raw("test", fd)
    rul_train = build_rul_target(train_df, rul_max=RUL_MAX)
    rul_test_true = load_rul(fd)

    all_units = engine_units(train_df)
    train_units, val_units, _ = split_by_unit(
        all_units, val_ratio=val_ratio, test_ratio=0.0, seed=SEED
    )

    # Per-condition normalizer (single regime for FD001/FD003, six for FD002/FD004).
    n_regimes = (
        1
        if train_df[SETTING_COLS].drop_duplicates().shape[0] == 1
        else N_OPERATING_CONDITIONS
    )
    normalizer = ConditionNormalizer(n_clusters=n_regimes).fit(
        train_df[train_df["unit"].isin(train_units)]
    )

    test_units = engine_units(test_df)

    summary: dict[str, float] = {}
    model_rows: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------- baselines
    feat_all, feature_cols = build_baseline_features(train_df, normalizer)
    train_mask = feat_all["unit"].isin(train_units).to_numpy()
    X_tr = feat_all.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_tr = rul_train[train_mask].to_numpy(dtype=float)

    # predict RUL at each test engine's final cycle
    feat_test, _ = build_baseline_features(test_df, normalizer)
    last_idx = feat_test.groupby("unit")["cycle"].idxmax().to_numpy()
    X_te = feat_test.loc[last_idx, feature_cols].to_numpy(dtype=float)

    baseline_results = train_baselines(X_tr, y_tr, X_te)

    for name, (model, pred) in baseline_results.items():
        pred_clipped = _clip_rul(pred)
        truth_clipped = _clip_rul(rul_test_true)
        metrics = score_rul(truth_clipped, pred_clipped)
        model_rows[name] = metrics
        summary[f"{name}_rmse"] = metrics["rmse"]
        summary[f"{name}_nasa"] = metrics["nasa_score"]
        print(f"  {name:18s} RMSE {metrics['rmse']:7.2f}  NASA {metrics['nasa_score']:7.2f}")
        if save:
            joblib.dump(model, MODELS_DIR / f"{name}_fd00{fd}.joblib")
            np.save(REPORTS_DIR / f"pred_{name}_fd00{fd}.npy", pred_clipped)

    # ----------------------------------------------------------- sequence
    seq_df = normalizer.transform(train_df)
    X_seq_train, y_seq_train, _ = build_windows(
        seq_df, rul_train, window, KEPT_SENSORS, units=train_units
    )
    # Validation uses windows across each held-out engine's FULL history (labels
    # span the piecewise range), so early stopping sees a representative target.
    X_val, y_val, _ = build_windows(
        seq_df, rul_train, window, KEPT_SENSORS, units=val_units
    )

    model = RULSequenceModel(
        n_features=len(KEPT_SENSORS),
        hidden_size=hidden,
        num_layers=num_layers,
        rnn_type=rnn_type,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  sequence ({rnn_type}) params={n_params:,}  train windows={X_seq_train.shape[0]}")

    seq_hist = train_sequence_model(
        model,
        X_seq_train,
        y_seq_train,
        X_val,
        y_val,
        epochs=epochs if not quick else 5,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    seq_test_df = normalizer.transform(test_df)
    X_seq_test, _, _ = build_last_windows(
        seq_test_df, window, KEPT_SENSORS, units=test_units, rul=None
    )
    seq_pred = predict_sequences(model, X_seq_test, device=device)
    seq_pred_clipped = _clip_rul(seq_pred)
    truth_clipped = _clip_rul(rul_test_true)
    seq_metrics = score_rul(truth_clipped, seq_pred_clipped)
    model_rows[f"{rnn_type}"] = seq_metrics
    summary[f"{rnn_type}_rmse"] = seq_metrics["rmse"]
    summary[f"{rnn_type}_nasa"] = seq_metrics["nasa_score"]
    summary[f"{rnn_type}_val_rmse"] = seq_hist["best_val_rmse"]
    print(
        f"  {rnn_type:18s} RMSE {seq_metrics['rmse']:7.2f}  NASA {seq_metrics['nasa_score']:7.2f}"
        f"  (val RMSE {seq_hist['best_val_rmse']:.2f})"
    )

    if save:
        torch.save(model.state_dict(), MODELS_DIR / f"sequence_{rnn_type}_fd00{fd}.pt")
        np.save(REPORTS_DIR / f"pred_{rnn_type}_fd00{fd}.npy", seq_pred_clipped)
        joblib.dump(
            {
                "normalizer": normalizer,
                "feature_cols": feature_cols,
                "window": window,
                "kept_sensors": KEPT_SENSORS,
                "rnn_type": rnn_type,
                "hidden_size": hidden,
                "num_layers": num_layers,
                "fd": fd,
            },
            MODELS_DIR / f"preprocessing_fd00{fd}.joblib",
        )

    summary["elapsed_sec"] = time.time() - t0

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        with open(REPORTS_DIR / f"results_fd00{fd}.json", "w") as f:
            json.dump(
                {
                    "config": {
                        "fd": fd,
                        "window": window,
                        "hidden": hidden,
                        "num_layers": num_layers,
                        "rnn_type": rnn_type,
                        "seed": SEED,
                        "n_train_units": int(train_units.shape[0]),
                        "n_val_units": int(val_units.shape[0]),
                        "n_test_units": int(test_units.shape[0]),
                    },
                    "models": model_rows,
                },
                f,
                indent=2,
            )
        np.save(REPORTS_DIR / f"truth_fd00{fd}.npy", truth_clipped)

    print(f"  done in {summary['elapsed_sec']:.1f}s")
    return model_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate C-MAPSS RUL models.")
    parser.add_argument("--fd", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rnn-type", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--quick", action="store_true", help="fewer epochs for a fast smoke test")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args()

    run(
        fd=args.fd,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        num_layers=args.num_layers,
        lr=args.lr,
        rnn_type=args.rnn_type,
        quick=args.quick,
        save=not args.no_save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
