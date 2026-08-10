"""Funnel statistics, loss-path classification, and transition counts.

This module is the single source of truth for the funnel and loss-pattern
definitions used by the analysis scripts, the churn model, and the dashboard.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    BEHAVIOR_TYPES,
    LOSS_PATTERN_CART,
    LOSS_PATTERN_FAV,
    LOSS_PATTERN_PURE_PV,
)


def funnel_stats(events: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    """Conversion funnel at both the user level and the session level.

    User level: distinct users who ever performed each behavior.
    Session level: sessions that ever contained each behavior.
    """
    n_users = events["user_id"].nunique()

    users = {
        b: int(events.loc[events["behavior_type"] == b, "user_id"].nunique())
        for b in BEHAVIOR_TYPES
    }
    sessions_pv = len(sessions)
    sessions_cart = int(sessions["has_cart"].sum())
    sessions_buy = int(sessions["has_buy"].sum())

    return {
        "user_level": {
            "pv": users["pv"],
            "cart": users["cart"],
            "buy": users["buy"],
            "cart_rate_from_pv": users["cart"] / users["pv"] if users["pv"] else 0.0,
            "buy_rate_from_pv": users["buy"] / users["pv"] if users["pv"] else 0.0,
            "cart_to_buy": users["buy"] / users["cart"] if users["cart"] else 0.0,
            "n_users": n_users,
        },
        "session_level": {
            "pv": sessions_pv,
            "cart": sessions_cart,
            "buy": sessions_buy,
            "cart_rate_from_pv": sessions_cart / sessions_pv if sessions_pv else 0.0,
            "buy_rate_from_pv": sessions_buy / sessions_pv if sessions_pv else 0.0,
            "cart_to_buy": sessions_buy / sessions_cart if sessions_cart else 0.0,
            "pv_to_cart_loss": 1 - sessions_cart / sessions_pv if sessions_pv else 0.0,
            "cart_to_buy_loss": 1 - sessions_buy / sessions_cart if sessions_cart else 0.0,
        },
    }


def classify_loss_patterns(sessions: pd.DataFrame) -> pd.DataFrame:
    """Label non-buying sessions with a loss-pattern category.

    Patterns (canonical definitions used across the project):
        pure_pv         — session contains only page views
        cart_abandoned  — item added to cart but never bought
        fav_only        — item favorited, never added to cart or bought

    Buying sessions get ``loss_pattern = None``. Returns a copy with the
    new column; the three patterns are mutually exclusive and cover every
    non-buying session.
    """
    df = sessions.copy()
    comp = df["compressed_sequence"].astype("string")
    no_buy = ~df["has_buy"]

    pattern = pd.Series([None] * len(df), index=df.index, dtype="object")
    pattern.loc[no_buy & (comp == "pv")] = LOSS_PATTERN_PURE_PV
    pattern.loc[no_buy & comp.str.contains("cart", na=False)] = LOSS_PATTERN_CART
    pattern.loc[
        no_buy & comp.str.contains("fav", na=False) & ~comp.str.contains("cart", na=False)
    ] = LOSS_PATTERN_FAV
    df["loss_pattern"] = pattern

    # Every non-buying session must be assigned exactly one pattern.
    assert df.loc[no_buy, "loss_pattern"].notna().all()
    return df


def count_transitions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Count state transitions in compressed sequences.

    Adds ``start`` (first state of each session) and ``end`` (last state)
    pseudo-nodes so a full journey graph can be drawn. Returns a DataFrame
    with columns ``source, target, count``.
    """
    seqs = sessions["compressed_sequence"].astype("string")
    exploded = seqs.str.split("→").explode()

    transitions = pd.DataFrame(
        {"state": exploded.astype(object), "next": exploded.groupby(level=0).shift(-1)}
    )
    transitions["is_first"] = ~transitions.index.duplicated()

    # terminal edges: last state of each session -> end
    is_end = transitions["next"].isna()
    end_edges = transitions.loc[is_end, "state"].rename("end_state")

    # start pseudo-node -> first state of each session
    start_edges = transitions.loc[transitions["is_first"], "state"].rename("start_state")

    links = []
    link_counts = transitions.loc[~is_end].groupby(["state", "next"], observed=True).size()
    for (src, tgt), cnt in link_counts.items():
        links.append({"source": src, "target": tgt, "count": int(cnt)})
    for state, cnt in start_edges.value_counts().items():
        links.append({"source": "start", "target": state, "count": int(cnt)})
    for state, cnt in end_edges.value_counts().items():
        links.append({"source": state, "target": "end", "count": int(cnt)})

    return pd.DataFrame(links)
