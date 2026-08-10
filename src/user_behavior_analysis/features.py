"""User-level feature engineering for churn prediction."""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# All features derive from the event/session history only. The label (whether
# the user bought) is deliberately kept out of this list: see make_labels()
# and the leak-guard assert at the end of build_user_features().
EVENT_FEATURE_COLS = [
    "total_events",
    "n_pv",
    "n_cart",
    "n_fav",
    "n_categories",
    "n_items",
    "n_active_days",
    "evening_ratio",
    "mean_inter_event_gap_min",
]
SESSION_FEATURE_COLS = [
    "n_sessions",
    "avg_session_len",
    "median_session_len",
    "avg_session_duration_min",
    "max_session_duration_min",
]
DERIVED_FEATURE_COLS = [
    "cart_rate",
    "fav_rate",
    "interaction_rate",
    "sessions_per_active_day",
    "avg_events_per_active_day",
    "activity_span_days",
    "items_per_session",
    "top1_category_share",
    "has_cart_and_fav",
]
FEATURE_COLS = EVENT_FEATURE_COLS + SESSION_FEATURE_COLS + DERIVED_FEATURE_COLS


def build_user_features(
    events: pd.DataFrame, sessions: pd.DataFrame, max_ts=None
) -> pd.DataFrame:
    """Build one row per user of aggregated behavior features.

    ``max_ts`` optionally ceilings the feature window (used by the time-split
    evaluation in the churn model). Labels are never part of the returned
    matrix; call :func:`make_labels` separately.
    """
    ev = events[events["timestamp"] <= max_ts] if max_ts is not None else events

    # --- event-level aggregates -----------------------------------------------
    ev = ev.copy()
    ev["_evening"] = (ev["timestamp"].dt.hour >= 18).astype("int8")

    event_features = ev.groupby("user_id", observed=True).agg(
        # total_events counts NON-purchase events only. total_events - n_pv -
        # n_cart - n_fav would otherwise equal n_buy exactly, letting a linear
        # model reconstruct the label from the four counts (label leakage).
        total_events=("behavior_type", lambda s: int((s != "buy").sum())),
        n_pv=("behavior_type", lambda s: int((s == "pv").sum())),
        n_cart=("behavior_type", lambda s: int((s == "cart").sum())),
        n_fav=("behavior_type", lambda s: int((s == "fav").sum())),
        n_categories=("category_id", "nunique"),
        n_items=("item_id", "nunique"),
        n_active_days=("timestamp", lambda s: s.dt.normalize().nunique()),
        evening_ratio=("_evening", "mean"),
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
    )

    # mean gap between consecutive events of the same user (minutes)
    ev_sorted = ev.sort_values(["user_id", "timestamp"])
    gap_min = (
        ev_sorted.groupby("user_id", observed=True)["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60.0
    )
    mean_gap = gap_min.groupby(ev_sorted["user_id"], observed=True).mean()
    event_features["mean_inter_event_gap_min"] = mean_gap

    # --- session-level aggregates ---------------------------------------------
    sess = sessions[sessions["start"] <= max_ts] if max_ts is not None else sessions
    session_features = sess.groupby("user_id", observed=True).agg(
        n_sessions=("raw_length", "size"),
        avg_session_len=("raw_length", "mean"),
        median_session_len=("raw_length", "median"),
        avg_session_duration_min=("duration_minutes", "mean"),
        max_session_duration_min=("duration_minutes", "max"),
    )

    # --- merge and derive ------------------------------------------------------
    features = event_features.join(session_features, how="left")
    for col in SESSION_FEATURE_COLS:
        features[col] = features[col].fillna(0.0)

    f = features
    pv = f["n_pv"].where(f["n_pv"] > 0, 1.0)  # avoid div-by-zero
    f["cart_rate"] = f["n_cart"] / pv
    f["fav_rate"] = f["n_fav"] / pv
    f["interaction_rate"] = (f["n_cart"] + f["n_fav"]) / f["total_events"]
    f["sessions_per_active_day"] = f["n_sessions"] / f["n_active_days"]
    f["avg_events_per_active_day"] = f["total_events"] / f["n_active_days"]
    f["activity_span_days"] = (f["last_ts"] - f["first_ts"]).dt.days + 1
    f["items_per_session"] = f["n_items"] / f["n_sessions"].where(f["n_sessions"] > 0, 1.0)
    f["has_cart_and_fav"] = ((f["n_cart"] > 0) & (f["n_fav"] > 0)).astype("int8")

    # top-1 category share: fraction of events in the user's most-viewed category
    cat_counts = ev.groupby(["user_id", "category_id"], observed=True).size()
    cat_sum = cat_counts.groupby("user_id", observed=True).sum()
    cat_max = cat_counts.groupby("user_id", observed=True).max()
    f["top1_category_share"] = (cat_max / cat_sum).reindex(f.index)

    # --- leak guard -------------------------------------------------------------
    leaks = {"n_buy", "buy", "bought"} & set(features.columns)
    assert not leaks, f"feature leak detected: {leaks} present in feature matrix"

    return features[FEATURE_COLS]


def make_labels(events: pd.DataFrame, users, min_ts=None, max_ts=None) -> pd.Series:
    """Binary label: 1 if the user bought at least once in the window."""
    ev = events
    if min_ts is not None:
        ev = ev[ev["timestamp"] >= min_ts]
    if max_ts is not None:
        ev = ev[ev["timestamp"] <= max_ts]
    buyers = set(ev.loc[ev["behavior_type"] == "buy", "user_id"].unique().tolist())
    return pd.Series([1 if u in buyers else 0 for u in users], index=users, name="bought")
