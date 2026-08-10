"""Unit tests for funnel statistics and loss-pattern classification."""

import pandas as pd

from user_behavior_analysis.funnel import classify_loss_patterns, count_transitions


def sessions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["a", "b", "c", "d", "e"],
            "has_buy": [False, False, False, False, True],
            "compressed_sequence": ["pv", "pv→cart", "pv→fav", "pv→cart→fav", "pv→cart→buy"],
        }
    )


def test_loss_patterns_match_definitions():
    out = classify_loss_patterns(sessions_frame()).set_index("session_id")
    assert out.loc["a", "loss_pattern"] == "pure_pv"
    assert out.loc["b", "loss_pattern"] == "cart_abandoned"
    assert out.loc["c", "loss_pattern"] == "fav_only"
    # contains cart -> cart_abandoned wins over fav_only
    assert out.loc["d", "loss_pattern"] == "cart_abandoned"
    # buying sessions are not loss patterns
    assert out.loc["e", "loss_pattern"] is None


def test_loss_patterns_cover_every_non_buying_session():
    out = classify_loss_patterns(sessions_frame())
    no_buy = out[~out["has_buy"]]
    assert no_buy["loss_pattern"].notna().all()
    assert no_buy["loss_pattern"].nunique() == 3  # mutually exclusive


def test_count_transitions_includes_start_and_end_pseudo_nodes():
    out = count_transitions(sessions_frame())
    by_edge = out.set_index(["source", "target"])["count"]

    assert by_edge[("start", "pv")] == 5  # every session starts with pv
    assert by_edge[("pv", "cart")] == 3
    assert by_edge[("pv", "end")] == 1  # session a: pv only
    assert by_edge[("cart", "end")] == 1  # session b
    assert by_edge[("fav", "end")] == 2  # sessions c and d
    assert by_edge[("cart", "fav")] == 1  # session d
