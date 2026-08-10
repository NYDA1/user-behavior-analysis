"""Data I/O helpers: typed loading, saving, logging, and timing."""

from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR

# Storage backend for intermediate tables. Parquet keeps dtypes and is much
# smaller/faster; set False to fall back to CSV when pyarrow is unavailable.
USE_PARQUET = True


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if USE_PARQUET else pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if USE_PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def log_step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def timed(label: str = ""):
    """Decorator that logs how long a pipeline step took."""

    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            log_step(f"{label or fn.__name__}: {time.perf_counter() - t0:.1f}s")
            return result

        return wrapper

    return deco


# ---------------------------------------------------------------------------
# Loaders for the processed tables
# ---------------------------------------------------------------------------
def load_sample() -> pd.DataFrame:
    """Load sampled raw events; converts timestamp to datetime and types behavior."""
    df = _read_table(PROCESSED_DIR / "user_sample.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["behavior_type"] = df["behavior_type"].astype("category")
    return df


def load_events() -> pd.DataFrame:
    return _read_table(PROCESSED_DIR / "events.parquet")


def load_sessions() -> pd.DataFrame:
    return _read_table(PROCESSED_DIR / "sessions.parquet")


def load_user_features() -> pd.DataFrame:
    return _read_table(PROCESSED_DIR / "user_features.parquet")


def load_model_predictions() -> pd.DataFrame:
    return _read_table(PROCESSED_DIR / "model_predictions.parquet")


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame into data/processed/ and log the row count."""
    path = PROCESSED_DIR / f"{name}{'.parquet' if USE_PARQUET else '.csv'}"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(df, path)
    log_step(f"saved {name} ({len(df):,} rows) -> {path.name}")
    return path


# ---------------------------------------------------------------------------
# JSON metrics
# ---------------------------------------------------------------------------
def _json_default(obj):
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    log_step(f"saved JSON -> {path}")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
