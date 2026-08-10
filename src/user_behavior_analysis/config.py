"""Global configuration: paths, constants, and column metadata."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = REPO_ROOT / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
MODEL_DIR = OUTPUT_DIR / "model"
METRICS_DIR = OUTPUT_DIR / "metrics"

RAW_DATA_PATH = RAW_DIR / "UserBehavior.csv"

# ---------------------------------------------------------------------------
# Raw data schema
# ---------------------------------------------------------------------------
COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
READ_DTYPES = {
    "user_id": "int32",
    "item_id": "int32",
    "category_id": "int32",
    "timestamp": "int64",
}
BEHAVIOR_TYPES = ["pv", "cart", "fav", "buy"]

# ---------------------------------------------------------------------------
# Analysis parameters (same methodology as the graduation thesis)
# ---------------------------------------------------------------------------
SEED = 42                 # global seed for sampling, splits, and models
SAMPLE_SIZE = 5000        # number of complete users to sample from the full data
SESSION_GAP_MINUTES = 30  # inactivity gap (minutes) that starts a new session
HIGH_VALUE_BUY_THRESHOLD = 3  # purchases per user for the "high-value" segment

# Time-based split for the churn model: features from days 1-7 predict the
# label from days 8-10, so feature and label windows never overlap.
TIME_SPLIT_FEATURE_DAYS = 7
TIME_SPLIT_LABEL_DAYS = 10

# Loss-path pattern names (definitions match the thesis):
LOSS_PATTERN_PURE_PV = "pure_pv"       # session contains only pv
LOSS_PATTERN_CART = "cart_abandoned"   # added to cart, never bought
LOSS_PATTERN_FAV = "fav_only"          # favorited, never cart/buy
