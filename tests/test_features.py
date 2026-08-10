"""Unit tests for user feature engineering and the label leak guard."""

import pandas as pd
import pytest

from user_behavior_analysis.features import FEATURE_COLS, build_user_features, make_labels

T0 = 1_500_000_000


def events_and_sessions():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "behavior_type": ["pv", "buy", "pv", "cart", "pv"],
            "category_id": [10, 10, 20, 30, 30],
            "item_id": [100, 100, 200, 300, 300],
            "timestamp": pd.to_datetime([T0 + i * 3600 for i in range(5)], unit="s"),
        }
    )
    sessions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "start": pd.to_datetime([T0, T0 + 2 * 3600], unit="s"),
            "raw_length": [2, 3],
            "duration_minutes": [60.0, 120.0],
        }
    )
    return events, sessions


def test_feature_matrix_has_no_buy_columns():
    events, sessions = events_and_sessions()
    features = build_user_features(events, sessions)
    assert set(features.columns) == set(FEATURE_COLS)
    assert "n_buy" not in features.columns
    assert "buy" not in features.columns
    assert "bought" not in features.columns


def test_feature_values():
    events, sessions = events_and_sessions()
    features = build_user_features(events, sessions)

    assert features.loc[1, "total_events"] == 2
    assert features.loc[1, "n_pv"] == 1
    assert features.loc[2, "total_events"] == 3
    assert features.loc[2, "n_cart"] == 1
    assert features.loc[2, "cart_rate"] == pytest.approx(1 / 2)
    assert features.loc[1, "n_sessions"] == 1


def test_max_ts_ceils_the_feature_window():
    events, sessions = events_and_sessions()
    # cut between user 1's two events (T0 and T0+1h)
    cut = pd.to_datetime(T0 + 1800, unit="s")
    features = build_user_features(events, sessions, max_ts=cut)
    # user 1's buy event falls after the cut and is excluded...
    assert features.loc[1, "total_events"] == 1
    assert features.loc[1, "n_pv"] == 1
    # ...and user 2 (all events after the cut) has no features at all
    assert 2 not in features.index


def test_labels_only_mark_actual_buyers():
    events, _ = events_and_sessions()
    labels = make_labels(events, users=[1, 2, 3])
    assert labels.loc[1] == 1  # bought
    assert labels.loc[2] == 0  # never bought
    assert labels.loc[3] == 0  # not even present in the data
