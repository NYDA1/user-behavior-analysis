"""02 · ETL: events → sessions → sequences → user-level features.

Input:  data/processed/user_sample.parquet
Outputs:
    data/processed/events.parquet           events + session_id
    data/processed/sessions.parquet         one row per session (sequences, flags)
    data/processed/user_features.parquet    one row per user (features + label)
"""

from user_behavior_analysis import features as feat
from user_behavior_analysis import sequences as seqs
from user_behavior_analysis import sessions as sess
from user_behavior_analysis.io import load_sample, log_step, save_table, timed


@timed("02 · ETL")
def run() -> None:
    events = load_sample()
    log_step(f"events: {len(events):,}, users: {events['user_id'].nunique():,}")

    events = sess.split_sessions(events)
    save_table(events, "events")

    sessions = seqs.build_session_sequences(events)
    save_table(sessions, "sessions")
    log_step(
        f"sessions: {len(sessions):,} | avg length {sessions['raw_length'].mean():.1f} "
        f"| median {sessions['raw_length'].median():.0f} | "
        f"buying {sessions['has_buy'].sum():,} ({sessions['has_buy'].mean() * 100:.1f}%)"
    )

    user_features = feat.build_user_features(events, sessions)
    labels = feat.make_labels(events, users=user_features.index)
    user_features["bought"] = labels  # label column, kept out of FEATURE_COLS
    save_table(user_features, "user_features")
    log_step(
        f"users: {len(user_features):,} | converted {int(user_features['bought'].sum()):,} "
        f"({user_features['bought'].mean() * 100:.1f}%)"
    )


if __name__ == "__main__":
    run()
