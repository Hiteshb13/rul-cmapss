"""Model definitions.

Two families of models are implemented:

* **Baselines** (sklearn / xgboost): trained on engineered per-cycle features.
* **Sequence model** (PyTorch GRU/LSTM): trained directly on raw windowed
  sensor sequences, learning the temporal degradation pattern end-to-end.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .seeds import SEED


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def make_baseline_models(seed: int = SEED) -> dict[str, object]:
    """Return the configured baseline estimators keyed by display name."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Sequence model (PyTorch)
# ---------------------------------------------------------------------------
class RULSequenceModel(nn.Module):
    """GRU/LSTM regressor that maps a sensor window to a scalar RUL."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape ``(batch, seq_len, n_features)`` -> ``(batch,)``."""
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # take the final timestep's hidden state
        return self.head(last).squeeze(-1)


def predict_sequences(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """Run ``model`` over a batch of windows and return scalar RUL predictions."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, tensor.shape[0], batch_size):
            batch = tensor[i : i + batch_size].to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])
