"""Create a small, git-friendly sample of the C-MAPSS data.

The full raw files are large and gitignored. This script writes a tiny subset
(first few engines of FD001) to ``data/sample/`` as CSV so the repo remains
self-contained and tests can run without the full download.

Usage
-----
    python scripts/make_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SAMPLE = PROJECT_ROOT / "data" / "sample"

sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw, load_rul  # noqa: E402

N_UNITS = 5


def main() -> int:
    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)

    train = load_raw("train", 1)
    test = load_raw("test", 1)
    rul = load_rul(1)

    train_sample = train[train["unit"] <= N_UNITS]
    test_sample = test[test["unit"] <= N_UNITS]
    rul_sample = rul[:N_UNITS]

    train_sample.to_csv(DATA_SAMPLE / "train_FD001_sample.csv", index=False)
    test_sample.to_csv(DATA_SAMPLE / "test_FD001_sample.csv", index=False)

    with open(DATA_SAMPLE / "RUL_FD001_sample.txt", "w") as f:
        f.write("\n".join(str(int(v)) for v in rul_sample) + "\n")

    print(f"Wrote sample (first {N_UNITS} units of FD001) to {DATA_SAMPLE}.")
    print(f"  train rows: {len(train_sample)}")
    print(f"  test rows : {len(test_sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
