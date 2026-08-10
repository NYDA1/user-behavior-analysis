"""Unit tests for compressed behavior-sequence building."""

import pandas as pd

from user_behavior_analysis.sequences import build_session_sequences

T0 = 1_500_000_000


def test_consecutive_duplicates_are_compressed():
    events = pd.DataFrame(
        {
            "user_id": [1] * 7,
            "session_id": ["u1_001"] * 5 + ["u1_002"] * 2,
            "behavior_type": ["pv", "pv", "cart", "cart", "fav", "pv", "pv"],
            "timestamp": pd.to_datetime([T0 + i * 60 for i in range(7)], unit="s"),
        }
    )
    sessions = build_session_sequences(events).set_index("session_id")

    s1 = sessions.loc["u1_001"]
    assert s1["compressed_sequence"] == "pv→cart→fav"
    assert s1["raw_length"] == 5
    assert s1["compressed_length"] == 3
    assert s1["has_cart"] and s1["has_fav"] and not s1["has_buy"]

    s2 = sessions.loc["u1_002"]
    assert s2["compressed_sequence"] == "pv"
    assert s2["compressed_length"] == 1
    assert not s2["has_cart"]


def test_run_detection_resets_at_session_boundary():
    # Session 2 starts with the same behavior session 1 ended with — the
    # session boundary must still mark a new run.
    events = pd.DataFrame(
        {
            "user_id": [1] * 4,
            "session_id": ["u1_001"] * 2 + ["u1_002"] * 2,
            "behavior_type": ["pv", "pv", "pv", "cart"],
            "timestamp": pd.to_datetime([T0 + i * 60 for i in range(4)], unit="s"),
        }
    )
    sessions = build_session_sequences(events).set_index("session_id")
    assert sessions.loc["u1_002", "compressed_sequence"] == "pv→cart"
    assert sessions.loc["u1_002", "compressed_length"] == 2


def test_compression_never_lengthens():
    events = pd.DataFrame(
        {
            "user_id": [1, 2],
            "session_id": ["u1_001", "u2_001"],
            "behavior_type": ["pv", "buy"],
            "timestamp": pd.to_datetime([T0, T0], unit="s"),
        }
    )
    sessions = build_session_sequences(events)
    assert (sessions["compressed_length"] <= sessions["raw_length"]).all()


def test_buy_flag_detected():
    events = pd.DataFrame(
        {
            "user_id": [1] * 3,
            "session_id": ["u1_001"] * 3,
            "behavior_type": ["pv", "cart", "buy"],
            "timestamp": pd.to_datetime([T0 + i * 60 for i in range(3)], unit="s"),
        }
    )
    sessions = build_session_sequences(events)
    assert sessions.loc[0, "has_buy"]
    assert sessions.loc[0, "compressed_sequence"] == "pv→cart→buy"
