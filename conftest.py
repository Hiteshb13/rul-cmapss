"""Pytest configuration.

Adds the project root to ``sys.path`` so tests can import the ``src`` package,
and provides a synthetic C-MAPSS-like fixture so feature/data tests run without
the full dataset download.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silence joblib core-detection warning

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import COLUMN_NAMES  # noqa: E402


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A small synthetic frame with 3 engines x 6 cycles, 21 sensors."""
    n_units, n_cycles = 3, 6
    n = n_units * n_cycles
    units = np.repeat(np.arange(1, n_units + 1), n_cycles)
    cycles = np.tile(np.arange(1, n_cycles + 1), n_units)

    rng = np.random.default_rng(42)
    sensors = rng.normal(size=(n, 21))
    # make one sensor perfectly linear (slope 2 per cycle) for slope tests
    sensors[:, 1] = 2.0 * cycles + 10.0

    df = pd.DataFrame(
        np.column_stack(
            [
                units,
                cycles,
                np.full(n, 100.0),  # setting_1
                np.full(n, 0.0),  # setting_2
                np.full(n, 518.0),  # setting_3
                sensors,
            ]
        ),
        columns=COLUMN_NAMES,
    )
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df
