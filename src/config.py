"""Central configuration and shared constants for the C-MAPSS RUL project.

Keeping constants here (instead of scattered through notebooks) means the
training pipeline, notebooks and tests all agree on column names, the sensor
drop-list, the piecewise-linear RUL cap and the file layout.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLE = PROJECT_ROOT / "data" / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# C-MAPSS schema
# ---------------------------------------------------------------------------
# Each row is: unit number, cycle, 3 operational settings, 21 sensor readings.
COLUMN_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"s{i}" for i in range(1, 22)]

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

# Constant / non-informative sensors identified during EDA (FD001). These
# carry no degradation signal and are dropped from the feature space. The list
# is confirmed empirically in notebooks/01_eda.ipynb; it is only centralised
# here so downstream code stays in sync.
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Sensors kept for modelling (informative under degradation).
KEPT_SENSORS = [s for s in SENSOR_COLS if s not in CONSTANT_SENSORS]

# ---------------------------------------------------------------------------
# Labelling / evaluation
# ---------------------------------------------------------------------------
# Piecewise-linear RUL: RUL is capped so the early, flat part of an engine's
# life does not dominate the regression target. Standard in C-MAPSS literature.
RUL_MAX = 125

# Number of operating-condition regimes (6 for FD002/FD004, 1 for FD001/FD003).
N_OPERATING_CONDITIONS = 6

# ---------------------------------------------------------------------------
# NASA asymmetric scoring function (Saxena et al., 2008)
# ---------------------------------------------------------------------------
# d = RUL_predicted - RUL_true
#   d < 0  -> early prediction (safe):  score = exp(-d / 13) - 1
#   d >= 0 -> late prediction (risky):  score = exp( d / 10) - 1
# Late predictions (over-estimating RUL) are penalised more heavily.
SCORE_A1 = 13.0
SCORE_A2 = 10.0

# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
# Maintenance-action threshold (cycles). When predicted RUL drops below this,
# an alert is raised.
ALERT_THRESHOLD = 30
