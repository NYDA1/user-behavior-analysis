"""Sessionization: split each user's event stream into sessions."""

from __future__ import annotations

import pandas as pd

from .config import SESSION_GAP_MINUTES


def split_sessions(
    events: pd.DataFrame, gap_minutes: int = SESSION_GAP_MINUTES
) -> pd.DataFrame:
    """Assign a ``session_id`` to every event.

    A new session starts when the gap to the user's previous event exceeds
    ``gap_minutes`` minutes, or at the user's first event. Pure function,
    no I/O. Returns a sorted copy of ``events`` with ``session_id`` added.

    A session is a continuous interaction window with at most a 30-minute
    idle gap between consecutive events.
    """
    df = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    gap_min = (
        df.groupby("user_id", observed=True)["timestamp"].diff().dt.total_seconds() / 60.0
    )
    is_new = gap_min.isna() | (gap_min > gap_minutes)

    session_num = is_new.groupby(df["user_id"], observed=True).cumsum()
    df["session_id"] = (
        df["user_id"].astype(str) + "_" + session_num.astype(str).str.zfill(3)
    )
    return df
