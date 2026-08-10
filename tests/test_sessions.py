"""Unit tests for sessionization (30-minute inactivity gap)."""

import pandas as pd

from user_behavior_analysis.sessions import split_sessions

T0 = 1_500_000_000


def make_events(user: int, offsets_seconds: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [user] * len(offsets_seconds),
            "timestamp": pd.to_datetime([T0 + o for o in offsets_seconds], unit="s"),
            "behavior_type": ["pv"] * len(offsets_seconds),
        }
    )


def test_gap_below_30_minutes_stays_in_one_session():
    # 60s, 29min, 29min59s gaps — all within one session
    out = split_sessions(make_events(7, [0, 60, 29 * 60, 30 * 60 - 1]))
    assert out["session_id"].nunique() == 1


def test_gap_exactly_30_minutes_stays_in_one_session():
    # "> 30" is the split rule, so exactly 30 min does NOT split
    out = split_sessions(make_events(7, [0, 30 * 60]))
    assert out["session_id"].nunique() == 1


def test_gap_over_30_minutes_starts_new_session():
    out = split_sessions(make_events(7, [0, 30 * 60 + 1]))
    assert out["session_id"].nunique() == 2


def test_users_never_share_a_session():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "timestamp": pd.to_datetime([T0] * 3, unit="s"),
            "behavior_type": ["pv", "cart", "pv"],
        }
    )
    out = split_sessions(df)
    assert out["session_id"].nunique() == 2


def test_session_ids_are_stable_strings():
    out = split_sessions(make_events(7, [0, 31 * 60]))
    assert sorted(out["session_id"].unique()) == ["7_001", "7_002"]
