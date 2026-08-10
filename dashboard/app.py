"""Interactive dashboard for the user behavior analysis.

Run with:  streamlit run dashboard/app.py

All data comes from the pipeline's artifacts (data/processed/*.parquet and
output/{metrics,model}/*.json) — the dashboard never recomputes analysis,
so it stays responsive on large samples.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from user_behavior_analysis.config import METRICS_DIR, MODEL_DIR, PROCESSED_DIR
from user_behavior_analysis.plots import (
    BEHAVIOR_COLORS,
    BEHAVIOR_ORDER,
    BASELINE,
    CONVERTED,
    FUNNEL_RAMP,
    LOSS_PATTERN_COLORS,
    LOSS_PATTERN_LABELS,
    LOSS_PATTERN_ORDER,
    LOST,
    MODEL_1,
    MODEL_2,
    build_sankey_figure,
)

st.set_page_config(page_title="User Behavior Analysis", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loaders (loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_overview():
    return load_json(METRICS_DIR / "overview.json")


@st.cache_data
def load_funnel():
    return load_json(METRICS_DIR / "funnel.json")


@st.cache_data
def load_loss_patterns():
    return load_json(METRICS_DIR / "loss_patterns.json")


@st.cache_data
def load_feature_compare():
    return load_json(METRICS_DIR / "feature_compare.json")


@st.cache_data
def load_transitions():
    records = load_json(METRICS_DIR / "transitions.json")
    return pd.DataFrame(records)


@st.cache_data
def load_model_metrics():
    return load_json(MODEL_DIR / "model_metrics.json")


@st.cache_data
def load_curves():
    return load_json(MODEL_DIR / "curves.json")


@st.cache_data
def load_importance():
    return load_json(MODEL_DIR / "feature_importance.json")


@st.cache_data
def load_events():
    return pd.read_parquet(PROCESSED_DIR / "events.parquet")


@st.cache_data
def load_user_features():
    return pd.read_parquet(PROCESSED_DIR / "user_features.parquet")


@st.cache_data
def load_predictions():
    return pd.read_parquet(PROCESSED_DIR / "model_predictions.parquet")


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def funnel_figure(stats: dict, title: str) -> go.Figure:
    stages = ["pv", "cart", "buy"]
    return go.Figure(
        go.Funnel(
            y=[s.upper() for s in stages],
            x=[stats[s] for s in stages],
            textinfo="value+percent initial",
            textposition="inside",
            marker=dict(color=FUNNEL_RAMP),
        )
    ).update_layout(title=title, height=420, margin=dict(t=60, b=20))


def loss_pattern_figure(counts: dict) -> go.Figure:
    order = [p for p in LOSS_PATTERN_ORDER if p in counts]
    return go.Figure(
        go.Bar(
            x=[LOSS_PATTERN_LABELS[p] for p in order],
            y=[counts[p] for p in order],
            marker_color=[LOSS_PATTERN_COLORS[p] for p in order],
            text=[f"{counts[p]:,} ({counts[p] / counts['total_non_buying'] * 100:.1f}%)" for p in order],
            textposition="outside",
        )
    ).update_layout(
        title="Loss-path patterns among non-buying sessions",
        yaxis_title="Number of sessions",
        height=420,
        margin=dict(t=60, b=20),
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview():
    ov = load_overview()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{ov['n_users']:,}")
    c2.metric("Events", f"{ov['n_events']:,}")
    c3.metric("Sessions", f"{ov['n_sessions']:,}")
    c4.metric("Buying sessions", f"{ov['n_converted_sessions']:,}")

    events = load_events()
    events["date"] = events["timestamp"].dt.date
    daily = events.groupby("date", observed=True)["behavior_type"].size()
    # Anchor the chart on the main observation window: a handful of stray
    # records exist weeks before the campaign period, which would otherwise
    # stretch the axis with zero-valued days.
    cutoff = events["timestamp"].quantile(0.01).normalize().date()
    daily = daily[daily.index >= cutoff]
    daily_fig = go.Figure(
        go.Bar(x=[str(d) for d in daily.index], y=daily.values, marker_color="#2a78d6")
    ).update_layout(title="Events per day", xaxis_title="Date", yaxis_title="Events", height=320)

    dist = events["behavior_type"].value_counts().reindex(BEHAVIOR_ORDER).dropna()
    dist_fig = go.Figure(
        go.Bar(
            x=dist.index,
            y=dist.values,
            marker_color=[BEHAVIOR_COLORS[b] for b in dist.index],
            text=[f"{v:,} ({v / dist.sum() * 100:.1f}%)" for v in dist.values],
            textposition="outside",
        )
    ).update_layout(title="Behavior type distribution", yaxis_title="Events", height=320)

    left, right = st.columns(2)
    left.plotly_chart(dist_fig, width="stretch")
    right.plotly_chart(daily_fig, width="stretch")


def page_funnel():
    funnel = load_funnel()
    level = st.selectbox("Granularity", ["session_level", "user_level"], format_func=lambda s: s.replace("_", " "))
    stats = funnel[level]
    st.plotly_chart(funnel_figure(stats, f"Conversion funnel ({level.replace('_', ' ')})"), width="stretch")
    st.dataframe(
        pd.DataFrame(
            {
                "Stage": ["PV", "Cart", "Buy"],
                "Count": [stats["pv"], stats["cart"], stats["buy"]],
                "Conversion (vs pv)": ["100%", f"{stats['cart_rate_from_pv'] * 100:.1f}%", f"{stats['buy_rate_from_pv'] * 100:.1f}%"],
                "Stage-to-stage": ["—", f"pv→cart {stats['cart_rate_from_pv'] * 100:.1f}%", f"cart→buy {stats['cart_to_buy'] * 100:.1f}%"],
            }
        ),
        hide_index=True,
    )


def page_loss_paths():
    counts = load_loss_patterns()
    st.plotly_chart(loss_pattern_figure(counts), width="stretch")

    sessions = load_events()
    top_seq = (
        sessions.groupby("session_id", observed=True)["behavior_type"]
        .agg(lambda s: "→".join(s))
        .value_counts()
    )
    st.subheader("Most frequent behavior sequences (all sessions)")
    st.dataframe(
        top_seq.head(20).rename("count").reset_index(),
        hide_index=True,
    )


def page_sankey():
    transitions = load_transitions()
    show_pseudo = st.checkbox("Show start/end nodes", value=True)
    st.plotly_chart(build_sankey_figure(transitions, show_pseudo=show_pseudo), width="stretch")


def page_feature_compare():
    compare = load_feature_compare()
    metric = st.selectbox("Metric", list(compare.keys()))
    vals = compare[metric]
    fig = go.Figure(
        go.Bar(
            x=["Converted", "Churned"],
            y=[vals["converted"], vals["lost"]],
            marker_color=[CONVERTED, LOST],
            text=[f"{vals['converted']:.2f}", f"{vals['lost']:.2f}"],
            textposition="outside",
        )
    ).update_layout(title=f"Mean {metric.replace('_', ' ')}", height=400, margin=dict(t=60, b=20))
    st.plotly_chart(fig, width="stretch")

    users = load_user_features()
    data = users[[metric, "bought"]].dropna()
    box = go.Figure()
    for label, mask, color in [("Converted", data["bought"] == 1, CONVERTED),
                               ("Churned", data["bought"] == 0, LOST)]:
        box.add_trace(go.Box(y=data.loc[mask, metric], name=label, marker_color=color))
    box.update_layout(title=f"Distribution of {metric.replace('_', ' ')}", height=400, margin=dict(t=60, b=20))
    st.plotly_chart(box, width="stretch")


def page_churn_model():
    metrics = load_model_metrics()
    curves = load_curves()
    importance = load_importance()
    preds = load_predictions()

    models = list(metrics["models"].keys())
    st.subheader("Model comparison")
    compare_rows = []
    for name in models:
        m = metrics["models"][name]
        compare_rows.append(
            {
                "Model": name,
                "ROC-AUC": f"{m['roc_auc']:.3f}",
                "PR-AUC": f"{m['pr_auc']:.3f}",
                "F1@0.5": f"{m['f1_0.5']:.3f}",
                "MCC": f"{m['mcc']:.3f}",
                "Precision@top10%": f"{m['precision_top10']:.3f}",
            }
        )
    st.dataframe(pd.DataFrame(compare_rows), hide_index=True)

    model = st.selectbox("Model", models, index=0)
    prob_col = f"prob_{model}"
    thresh = st.slider("Decision threshold", 0.05, 0.95, float(metrics["models"][model]["youden_threshold"]), 0.01)
    preds["pred"] = (preds[prob_col] >= thresh).astype(int)

    cm = pd.crosstab(preds["y_true"], preds["pred"])
    for col in (0, 1):
        if col not in cm.columns:
            cm[col] = 0
    cm = cm[[0, 1]].rename(index={0: "Not bought", 1: "Bought"}, columns={0: "Predicted no", 1: "Predicted yes"})

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            go.Figure(
                go.Heatmap(
                    z=cm.values,
                    x=cm.columns,
                    y=cm.index,
                    text=[[f"{v:,}" for v in row] for row in cm.values],
                    texttemplate="%{text}",
                    colorscale="Blues",
                    zmin=0,
                    showscale=False,
                )
            ).update_layout(title=f"Confusion matrix @ {thresh:.2f}", height=380, margin=dict(t=60, b=20)),
            width="stretch",
        )
    with right:
        c = curves[model]
        roc_fig = go.Figure(
            [
                go.Scatter(x=c["fpr"], y=c["tpr"], name=model, mode="lines", line=dict(color=MODEL_1, width=2)),
                go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(color=BASELINE, dash="dash")),
            ]
        ).update_layout(title=f"ROC (AUC {c['auc']:.3f})", height=380, margin=dict(t=60, b=20),
                        xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(roc_fig, width="stretch")

    imp = pd.Series(importance[model]).sort_values(ascending=True).tail(15)
    st.plotly_chart(
        go.Figure(
            go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=MODEL_2)
        ).update_layout(title=f"Feature importance — {model}", height=480, margin=dict(t=60, b=20)),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Funnel": page_funnel,
    "Loss paths": page_loss_paths,
    "Sankey": page_sankey,
    "Feature compare": page_feature_compare,
    "Churn model": page_churn_model,
}

st.sidebar.title("📊 User Behavior")
st.sidebar.caption("Taobao UserBehavior dataset · sessionized · churn modeling")
page = st.sidebar.radio("Section", list(PAGES))
PAGES[page]()
