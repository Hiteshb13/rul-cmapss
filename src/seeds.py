"""Reproducibility helpers.

Centralises the random seed so that every stochastic component of the
pipeline (train/val/test split, model initialisation, data shuffling) is
deterministic and reproducible across machines.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (optionally) PyTorch RNGs.

    Parameters
    ----------
    seed : int
        Random seed to use everywhere.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # avoid joblib core-detection warning
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional at seed time
        pass
