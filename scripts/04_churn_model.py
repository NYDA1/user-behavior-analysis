"""04 · Churn prediction: converting vs churned users.

Binary classification on user-level behavior features predicting whether a
user makes a purchase at all. Two evaluation setups guard against leakage:

  --split group (default)
      Stratified random split by user (one row per user, so the split is
      inherently user-disjoint).

  --split time
      Features come from days 1-7 of the observation window, the label from
      days 8-10. Feature and label windows never overlap, which is a stronger
      claim than a random split: a user's purchase history cannot leak into
      the features.

Class imbalance (~8-10% positive) is handled with class_weight / scale_pos_weight.
Evaluation focuses on ROC-AUC and PR-AUC (the honest metric under imbalance),
plus F1 / MCC at a Youden-optimal threshold and precision@top-10%.

Outputs (output/model/): roc_auc.png, pr_auc.png, confusion_matrix.png,
feature_importance.png, model_metrics.json, feature_importance.json,
curves.json and data/processed/model_predictions.parquet.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from user_behavior_analysis import features as feat
from user_behavior_analysis import plots as P
from user_behavior_analysis.config import (
    MODEL_DIR,
    PROCESSED_DIR,
    SEED,
    TIME_SPLIT_FEATURE_DAYS,
    TIME_SPLIT_LABEL_DAYS,
)
from user_behavior_analysis.io import (
    load_events,
    load_sessions,
    load_user_features,
    log_step,
    save_json,
    timed,
)

THRESHOLD = 0.5
TOP_DECILE = 0.1  # precision@top-10%


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["group", "time"], default="group",
                    help="evaluation split: group (random, default) or time (no overlap)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-xgboost", action="store_true", help="skip XGBoost even if installed")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_split(args, events, sessions, user_features):
    """Return (X_train, X_test, y_train, y_test, test_users, label)."""
    if args.split == "group":
        X = user_features[feat.FEATURE_COLS].fillna(0.0)
        y = user_features["bought"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed
        )
        test_users = X_test.index.tolist()
        split_label = "stratified random user split (test %.0f%%)" % (args.test_size * 100)
    else:  # time split: days 1-7 features -> days 8-10 label
        # The dataset's main window starts 2017-11-24; a handful of stray
        # records exist weeks earlier. Anchor the window at the 1% quantile
        # date so the split lands on the main period.
        day0 = events["timestamp"].quantile(0.01).normalize()
        feat_cut = day0 + pd.Timedelta(days=TIME_SPLIT_FEATURE_DAYS)
        label_cut = day0 + pd.Timedelta(days=TIME_SPLIT_LABEL_DAYS)

        X = feat.build_user_features(
            events[events["timestamp"] <= feat_cut], sessions, max_ts=feat_cut
        ).fillna(0.0)
        y = feat.make_labels(events, users=X.index, min_ts=feat_cut, max_ts=label_cut).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed
        )
        test_users = X_test.index.tolist()
        split_label = (
            f"time split (features ≤ day {TIME_SPLIT_FEATURE_DAYS}, "
            f"label days {TIME_SPLIT_FEATURE_DAYS + 1}-{TIME_SPLIT_LABEL_DAYS})"
        )

    log_step(f"split: {split_label} | train {len(X_train):,} users, test {len(X_test):,} users")
    return X_train, X_test, y_train, y_test, test_users, split_label


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models(X_train, y_train, args):
    models = {
        # StandardScaler: feature scales range from 0-1 ratios to 0-800 counts;
        # scaling fixes lbfgs convergence and makes LR coefficients comparable.
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=args.seed, n_jobs=-1,
        ),
    }
    if not args.no_xgboost:
        try:
            from xgboost import XGBClassifier

            neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
            models["XGBoost"] = XGBClassifier(
                scale_pos_weight=neg / max(pos, 1),
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=args.seed,
                n_jobs=-1,
            )
            log_step("XGBoost available — included in comparison")
        except ImportError:
            log_step("xgboost not installed — running with LR + RandomForest only")
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    # Youden's J: threshold maximizing tpr - fpr
    youden = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[youden])

    y_pred_50 = (y_prob >= THRESHOLD).astype(int)
    y_pred_best = (y_prob >= best_threshold).astype(int)

    # precision@top-10%: among the 10% most-likely users, how many convert
    n_top = max(int(np.ceil(len(y_test) * TOP_DECILE)), 1)
    top_idx = np.argsort(y_prob)[::-1][:n_top]
    precision_top = float(y_test.iloc[top_idx].mean())

    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(auc(recall, precision)),
        "precision_0.5": float(precision_score(y_test, y_pred_50, zero_division=0)),
        "recall_0.5": float(recall_score(y_test, y_pred_50, zero_division=0)),
        "f1_0.5": float(f1_score(y_test, y_pred_50, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred_50)),
        "youden_threshold": best_threshold,
        "f1_youden": float(f1_score(y_test, y_pred_best, zero_division=0)),
        "precision_top10": precision_top,
        "curves": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                   "precision": precision.tolist(), "recall": recall.tolist()},
    }


def run(args) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    events = load_events()
    sessions = load_sessions()
    user_features = load_user_features()

    X_train, X_test, y_train, y_test, test_users, split_label = prepare_split(
        args, events, sessions, user_features
    )
    prevalence = float(y_test.mean())

    models = build_models(X_train, y_train, args)
    for name, model in models.items():
        model.fit(X_train, y_train)

    results = [evaluate_model(name, model, X_test, y_test) for name, model in models.items()]

    # ------------------------------------------------------------------ outputs
    metrics_out = {
        "split": args.split,
        "split_description": split_label,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "prevalence": prevalence,
        "models": {r["model"]: {k: v for k, v in r.items() if k != "curves"} for r in results},
    }
    save_json(metrics_out, MODEL_DIR / "model_metrics.json")

    curves_out = {r["model"]: {**r["curves"], "auc": r["roc_auc"], "pr_auc": r["pr_auc"]} for r in results}
    save_json(curves_out, MODEL_DIR / "curves.json")

    # charts
    P.plot_roc_curves(curves_out, MODEL_DIR / "roc_auc.png",
                      title=f"ROC curves — {args.split} split")
    P.plot_pr_curves(curves_out, prevalence, MODEL_DIR / "pr_auc.png",
                     title=f"Precision-recall curves — {args.split} split")

    # confusion matrix of the best model (by PR-AUC) at the Youden threshold
    best = max(results, key=lambda r: r["pr_auc"])
    model = models[best["model"]]
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, (y_prob >= best["youden_threshold"]).astype(int))
    P.plot_confusion_matrix(
        cm, MODEL_DIR / "confusion_matrix.png",
        title=f"Confusion matrix — {best['model']} @ Youden threshold ({best['youden_threshold']:.3f})",
    )

    # feature importance per model
    importance = {}
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importance[name] = pd.Series(model.feature_importances_, index=feat.FEATURE_COLS)
        else:  # linear model: |standardized coefficient| (last step of pipeline)
            coefs = pd.Series(np.abs(model[-1].coef_[0]), index=feat.FEATURE_COLS)
            importance[name] = coefs / coefs.sum()
    save_json({name: s.sort_values(ascending=False).to_dict() for name, s in importance.items()},
              MODEL_DIR / "feature_importance.json")
    P.plot_feature_importance(importance, MODEL_DIR / "feature_importance.png")

    # per-user predictions for the dashboard
    pred_df = pd.DataFrame({"user_id": test_users, "y_true": y_test.values})
    for name, model in models.items():
        pred_df[f"prob_{name}"] = model.predict_proba(X_test)[:, 1]
    pred_df["split"] = args.split
    pred_df.to_parquet(PROCESSED_DIR / "model_predictions.parquet", index=False)
    log_step(f"saved model predictions for {len(pred_df):,} test users")

    # ------------------------------------------------------------------ console
    log_step(f"\n=== Churn prediction ({args.split} split) ===")
    log_step(f"prevalence (bought in test): {prevalence:.2%}")
    for r in results:
        log_step(
            f"  {r['model']:<20} ROC-AUC {r['roc_auc']:.3f} | PR-AUC {r['pr_auc']:.3f} "
            f"| F1@0.5 {r['f1_0.5']:.3f} | MCC {r['mcc']:.3f} "
            f"| precision@top10% {r['precision_top10']:.3f} (vs {prevalence:.3f} baseline)"
        )


if __name__ == "__main__":
    run(parse_args())
