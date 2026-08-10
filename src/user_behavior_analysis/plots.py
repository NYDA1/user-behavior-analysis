"""Unified chart style, palette, and plotting helpers.

Matplotlib charts, the Plotly Sankey figure, and the Streamlit dashboard all
reference the palette constants in this module, so every visualisation in the
project renders with the same colors. Labels are English (the dataset's own
behavior terms), which also avoids CJK font issues on headless systems.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared palette
# ---------------------------------------------------------------------------
BEHAVIOR_COLORS = {
    "pv": "#2a78d6",
    "cart": "#eb6834",
    "fav": "#1baf7a",
    "buy": "#008300",
}
CONVERTED = "#008300"  # users who bought
LOST = "#e34948"  # users who never bought
FUNNEL_RAMP = ["#86b6ef", "#3987e5", "#104281"]  # sequential blues
MODEL_1 = "#2a78d6"
MODEL_2 = "#eb6834"
BASELINE = "#898781"  # grey for baselines / axes

LOSS_PATTERN_COLORS = {
    "pure_pv": "#2a78d6",
    "cart_abandoned": "#eb6834",
    "fav_only": "#1baf7a",
}
LOSS_PATTERN_LABELS = {
    "pure_pv": "Pure page views",
    "cart_abandoned": "Cart abandoned",
    "fav_only": "Favorited only",
}
LOSS_PATTERN_ORDER = ["pure_pv", "cart_abandoned", "fav_only"]

BEHAVIOR_ORDER = ["pv", "cart", "fav", "buy"]


def setup_matplotlib() -> None:
    """Apply the project-wide matplotlib style (English labels, subtle grid)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#898781",
            "axes.labelcolor": "#2c2c2c",
            "axes.titlecolor": "#2c2c2c",
            "axes.titlesize": 13,
            "axes.grid": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_behavior_distribution(events: pd.DataFrame, path) -> None:
    """Bar chart of the share of each behavior type."""
    counts = events["behavior_type"].value_counts()
    counts = counts.reindex(BEHAVIOR_ORDER).dropna().astype(int)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        counts.index, counts.values, color=[BEHAVIOR_COLORS[b] for b in counts.index]
    )
    ax.set_title("Behavior type distribution")
    ax.set_ylabel("Number of events")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,} ({val / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_funnel(level_stats: dict, path, level: str) -> None:
    """Horizontal funnel bar chart for user- or session-level stats."""
    stages = ["pv", "cart", "buy"]
    values = [level_stats[s] for s in stages]
    labels = [f"{s.upper()} ({v:,})" for s, v in zip(stages, values)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=FUNNEL_RAMP)
    base = values[0]

    for i, (bar, val) in enumerate(zip(bars, values)):
        if i > 0:
            ax.text(
                bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{val / base * 100:.1f}% of pv",
                va="center",
                ha="center",
                color="white",
                fontsize=9,
            )
    ax.set_title(f"Conversion funnel — {level.replace('_', ' ')}")
    ax.set_xlabel("Count")
    ax.set_xlim(0, base * 1.1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_patterns(sessions: pd.DataFrame, path) -> None:
    """Bar chart of the three loss-path patterns among non-buying sessions."""
    no_buy = sessions[~sessions["has_buy"]]
    counts = no_buy["loss_pattern"].value_counts()
    counts = counts.reindex(LOSS_PATTERN_ORDER).dropna().astype(int)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        [LOSS_PATTERN_LABELS[p] for p in counts.index],
        counts.values,
        color=[LOSS_PATTERN_COLORS[p] for p in counts.index],
    )
    ax.set_title("Loss-path patterns among non-buying sessions")
    ax.set_ylabel("Number of sessions")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,} ({val / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_compare(compare: pd.DataFrame, path) -> None:
    """2x3 faceted bar chart comparing converting vs churned users per metric.

    Each metric gets its own panel and its own axis, fixing the original
    chart's mixed-unit problem (session length, cart count, and session
    duration on one shared axis).
    """
    metrics = compare.index.tolist()
    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        converted = compare.loc[metric, "converted"]
        lost = compare.loc[metric, "lost"]
        bars = ax.bar(
            ["Converted", "Churned"],
            [converted, lost],
            color=[CONVERTED, LOST],
            width=0.55,
        )
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("Mean value")
        for bar, val in zip(bars, [converted, lost]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:,.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, max(converted, lost) * 1.2)

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Behavioral features: converting vs churned users", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model evaluation charts
# ---------------------------------------------------------------------------
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # LR, RF, XGB


def plot_roc_curves(curves: dict, path, title: str = "ROC curves") -> None:
    """Overlay ROC curves; `curves` maps model name -> {fpr, tpr, auc}."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["fpr"], c["tpr"], lw=1.8, color=MODEL_COLORS[i % 3], label=f"{name} (AUC {c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=BASELINE, label="Random")
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pr_curves(curves: dict, prevalence: float, path, title: str = "Precision-recall curves") -> None:
    """Overlay PR curves with the positive-prevalence baseline."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["recall"], c["precision"], lw=1.8, color=MODEL_COLORS[i % 3], label=f"{name} (PR-AUC {c['pr_auc']:.3f})")
    ax.axhline(prevalence, ls="--", lw=1, color=BASELINE, label=f"Prevalence ({prevalence:.1%})")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, path, title: str = "Confusion matrix") -> None:
    """Heatmap of a 2x2 confusion matrix with counts and row-normalized %."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Not bought", "Bought"])
    ax.set_yticks([0, 1], ["Not bought", "Bought"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    row_tot = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / row_tot[i, 0] if row_tot[i, 0] else 0.0
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1%})", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(top_by_model: dict, path, top_n: int = 12) -> None:
    """Horizontal bar panels, one per model, of the top-N feature importances."""
    models = list(top_by_model.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, series) in zip(axes, top_by_model.items()):
        s = series.sort_values(ascending=True).tail(top_n)
        ax.barh(s.index, s.values, color=MODEL_COLORS[models.index(name) % 3])
        ax.set_title(f"{name} — feature importance")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Top feature importances by model", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


setup_matplotlib()


# ---------------------------------------------------------------------------
# Sankey (Plotly) — shared by the analysis script and the dashboard
# ---------------------------------------------------------------------------
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)' (plotly 6 rejects 8-digit hex)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_sankey_figure(transitions: pd.DataFrame, show_pseudo: bool = True):
    """Plotly Sankey figure from a transitions table (source/target/count).

    Node order follows the journey: start -> pv/cart/fav/buy -> end.
    """
    import plotly.graph_objects as go

    nodes = ["start"] + BEHAVIOR_ORDER + ["end"]
    node_map = {n: i for i, n in enumerate(nodes)}
    node_colors = ["#9aa0a6"] + [BEHAVIOR_COLORS[b] for b in BEHAVIOR_ORDER] + ["#9aa0a6"]

    df = transitions[transitions["source"].isin(node_map) & transitions["target"].isin(node_map)]
    if not show_pseudo:
        df = df[~df["source"].isin(["start", "end"]) & ~df["target"].isin(["start", "end"])]

    sources = [node_map[s] for s in df["source"]]
    targets = [node_map[t] for t in df["target"]]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=nodes,
                color=node_colors,
                pad=16,
                thickness=20,
                line=dict(color="#ffffff", width=1),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=df["count"].tolist(),
                color=[_hex_to_rgba(node_colors[node_map[t]], 0.4) for t in df["target"]],
            ),
        )
    )
    fig.update_layout(
        title="User journey through behaviors (session level)",
        font=dict(size=12),
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


setup_matplotlib()
