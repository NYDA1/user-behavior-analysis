"""03 · Analysis: funnels, loss paths, feature comparison, and sankey.

Reads the processed tables and writes every chart and metrics JSON consumed
by the dashboard (output/charts/ and output/metrics/).
"""

import pandas as pd

from user_behavior_analysis import funnel as f
from user_behavior_analysis import plots as P
from user_behavior_analysis.config import CHARTS_DIR, METRICS_DIR
from user_behavior_analysis.io import (
    load_events,
    load_sessions,
    load_user_features,
    log_step,
    save_json,
    timed,
)

# Metrics used in the converting-vs-churned comparison (mean per user group).
COMPARE_METRICS = [
    "avg_session_len",
    "n_cart",
    "n_fav",
    "avg_session_duration_min",
    "n_active_days",
    "n_sessions",
]


@timed("03 · analysis")
def run() -> None:
    events = load_events()
    sessions = load_sessions()
    user_features = load_user_features()

    # --- conversion funnel (user + session level) ----------------------------
    funnel = f.funnel_stats(events, sessions)
    save_json(funnel, METRICS_DIR / "funnel.json")
    P.plot_funnel(funnel["user_level"], CHARTS_DIR / "conversion_funnel_users.png", "user level")
    P.plot_funnel(funnel["session_level"], CHARTS_DIR / "conversion_funnel_sessions.png", "session level")
    log_step(
        "funnel: pv→cart %.2f%%, cart→buy %.2f%% (session level)"
        % (
            funnel["session_level"]["cart_rate_from_pv"] * 100,
            funnel["session_level"]["cart_to_buy"] * 100,
        )
    )

    # --- loss-path patterns (single source of truth: funnel.classify) --------
    sessions = f.classify_loss_patterns(sessions)
    no_buy = sessions[~sessions["has_buy"]]
    pattern_counts = no_buy["loss_pattern"].value_counts()
    loss_patterns = {str(k): int(v) for k, v in pattern_counts.items()}
    loss_patterns["total_non_buying"] = int(len(no_buy))
    save_json(loss_patterns, METRICS_DIR / "loss_patterns.json")
    P.plot_loss_patterns(sessions, CHARTS_DIR / "loss_pattern_distribution.png")

    # --- behavior distribution ------------------------------------------------
    P.plot_behavior_distribution(events, CHARTS_DIR / "behavior_distribution.png")

    # --- converting vs churned user features -----------------------------------
    converted = user_features[user_features["bought"] == 1]
    lost = user_features[user_features["bought"] == 0]
    compare = pd.DataFrame(
        {
            "converted": [float(converted[m].mean()) for m in COMPARE_METRICS],
            "lost": [float(lost[m].mean()) for m in COMPARE_METRICS],
        },
        index=COMPARE_METRICS,
    )
    save_json(compare.to_dict("index"), METRICS_DIR / "feature_compare.json")
    P.plot_feature_compare(compare, CHARTS_DIR / "user_feature_compare.png")

    # --- sankey journey (transitions table shared with the dashboard) ---------
    transitions = f.count_transitions(sessions)
    save_json(transitions.to_dict("records"), METRICS_DIR / "transitions.json")
    P.build_sankey_figure(transitions).write_html(CHARTS_DIR / "user_path_sankey.html")
    log_step(f"sankey: {len(transitions)} transition types")

    # --- overview KPIs ----------------------------------------------------------
    overview = {
        "n_events": int(len(events)),
        "n_users": int(events["user_id"].nunique()),
        "n_sessions": int(len(sessions)),
        "n_converted_users": int(user_features["bought"].sum()),
        "n_converted_sessions": int(sessions["has_buy"].sum()),
        "avg_session_length": round(float(sessions["raw_length"].mean()), 2),
        "date_range": [
            str(events["timestamp"].min().date()),
            str(events["timestamp"].max().date()),
        ],
    }
    save_json(overview, METRICS_DIR / "overview.json")
    log_step(f"overview: {overview}")


if __name__ == "__main__":
    run()
