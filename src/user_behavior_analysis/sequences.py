"""Session-level aggregation: compressed behavior sequences and flags."""

from __future__ import annotations

import pandas as pd


def build_session_sequences(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events into one row per session.

    Columns:
        session_id, user_id, start, has_buy, has_cart, has_fav,
        compressed_sequence (consecutive duplicates collapsed, e.g.
        ``pv→pv→cart`` becomes ``pv→cart``), raw_length, compressed_length,
        duration_minutes, start_hour.

    Fully vectorized (shift/ne run detection), so it scales linearly with the
    number of events instead of looping over sessions.
    """
    df = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # A "run" starts at the first event of a session or when the behavior
    # differs from the previous event of the same session.
    same_session = df["session_id"] == df["session_id"].shift(1)
    prev_behavior = df["behavior_type"].shift(1)
    is_run_start = ~same_session | (df["behavior_type"] != prev_behavior)

    # --- session-level aggregates (vectorized) --------------------------------
    agg = df.groupby("session_id", sort=False, observed=True).agg(
        user_id=("user_id", "first"),
        start=("timestamp", "min"),
        has_buy=("behavior_type", lambda s: bool((s == "buy").any())),
        has_cart=("behavior_type", lambda s: bool((s == "cart").any())),
        has_fav=("behavior_type", lambda s: bool((s == "fav").any())),
        raw_length=("behavior_type", "size"),
        duration_minutes=("timestamp", lambda s: (s.max() - s.min()).total_seconds() / 60.0),
    )

    # --- compressed sequence per session --------------------------------------
    run_states = df.loc[is_run_start]
    compressed = (
        run_states.groupby("session_id", sort=False, observed=True)["behavior_type"]
        .agg(lambda s: "→".join(s.astype(object)))
    )
    compressed_length = run_states.groupby("session_id", sort=False, observed=True).size()

    sessions = agg.reset_index()
    sessions["compressed_sequence"] = sessions["session_id"].map(compressed)
    sessions["compressed_length"] = sessions["session_id"].map(compressed_length)
    sessions["start_hour"] = sessions["start"].dt.hour

    # Guarantee: compression never lengthens a sequence.
    assert (sessions["compressed_length"] <= sessions["raw_length"]).all()

    return sessions
