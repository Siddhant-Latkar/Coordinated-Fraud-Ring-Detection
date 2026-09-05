"""
Trains the risk model on a chronological (never shuffled) split and
prints a scorecard: precision, recall, PR-AUC, a confusion matrix, and
an estimated rupee cost/benefit for false positives vs. fraud caught.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.decisioning import choose_action, combine_scores
from src.features import FEATURES, build_features
from src.rules import evaluate_rules

# Placeholder assumptions -- replace with real numbers from the
# merchant/ops team before trusting the "net value" line in production.
REVIEW_COST_PER_FALSE_POSITIVE = 50.0  # cost of a manual review / step-up friction, in rupees
DECISION_THRESHOLD = 0.5


def _score_test_set_through_deployed_pipeline(model, test_df):
    """
    Scores each held-out row through the same combine_scores() +
    evaluate_rules() + choose_action() functions used in src/scoring.py,
    so the reported metrics match what /transactions and the webhook
    actually decide on, not just the raw model.
    """
    model_scores = model.predict_proba(test_df[FEATURES])[:, 1]
    actions = []

    for model_score, (_, row) in zip(model_scores, test_df.iterrows()):
        stats = {
            "device_unique_users_24h": row["device_unique_users_24h"],
            "ip_unique_users_24h": row["ip_unique_users_24h"],
            "device_unique_users_30d": row["device_unique_users_30d"],
            "ip_unique_users_30d": row["ip_unique_users_30d"],
            "device_txn_count_24h": row["device_txn_count_24h"],
            "ip_txn_count_24h": row["ip_txn_count_24h"],
        }
        rule_score, _ = evaluate_rules({"amount": row["amount"]}, stats)
        final_score = combine_scores(model_score, row["graph_risk_score"], rule_score)
        actions.append(choose_action(final_score))

    flagged = pd.Series([action != "allow" for action in actions], index=test_df.index)
    return flagged, actions


def main():
    df = build_features(pd.read_csv("data/transactions.csv"))

    # Chronological split ONLY. Shuffling time-series/fraud data leaks
    # future velocity and identity-sharing context into training.
    cut = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        random_state=42,
    ).fit(train_df[FEATURES], train_df.label)

    proba = model.predict_proba(test_df[FEATURES])[:, 1]
    preds = proba > DECISION_THRESHOLD

    pr_auc = average_precision_score(test_df.label, proba)
    model_precision = precision_score(test_df.label, preds, zero_division=0)
    model_recall = recall_score(test_df.label, preds, zero_division=0)
    model_tn, model_fp, model_fn, model_tp = confusion_matrix(test_df.label, preds).ravel()

    # The number that matters: what /transactions and the Razorpay
    # webhook actually decide, running every row through the real
    # model+graph+rule blend and allow/review/step_up thresholds --
    # not just the raw model at an arbitrary 0.5 cutoff.
    flagged, _ = _score_test_set_through_deployed_pipeline(model, test_df)
    precision = precision_score(test_df.label, flagged, zero_division=0)
    recall = recall_score(test_df.label, flagged, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_df.label, flagged).ravel()

    avg_fraud_amount = (
        test_df.loc[test_df.label == 1, "amount"].mean()
        if test_df.label.sum() > 0
        else 0.0
    )
    fraud_amount_caught = tp * avg_fraud_amount
    review_cost = fp * REVIEW_COST_PER_FALSE_POSITIVE
    net_value = fraud_amount_caught - review_cost

    print("=== Held-out test set: last 20% chronologically, never seen in training ===")
    print(f"PR-AUC (model only): {pr_auc:.4f}")
    print()
    print("--- Raw ML model alone, threshold 0.5 (diagnostic only -- NOT what's deployed) ---")
    print(f"Precision:           {model_precision:.4f}")
    print(f"Recall:              {model_recall:.4f}")
    print(f"Confusion matrix:    TP={model_tp}  FP={model_fp}  FN={model_fn}  TN={model_tn}")
    print()
    print("--- DEPLOYED pipeline: model + graph + rules blend, allow/review/step_up thresholds ---")
    print("--- (this is what src/scoring.py, /transactions, and the Razorpay webhook use) ---")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"Confusion matrix:    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(
        f"Est. fraud caught:   Rs.{fraud_amount_caught:,.0f}  "
        f"({tp} transactions @ ~Rs.{avg_fraud_amount:,.0f} avg)"
    )
    print(
        f"Est. review cost:    Rs.{review_cost:,.0f}  "
        f"({fp} legitimate txns flagged @ Rs.{REVIEW_COST_PER_FALSE_POSITIVE:.0f}/review)"
    )
    print(f"Net estimated value: Rs.{net_value:,.0f}")

    if pr_auc > 0.999:
        print(
            "\nWARNING: near-perfect PR-AUC. On real transaction data this "
            "almost always means leakage or an unrealistically separable "
            "synthetic dataset -- not a production-ready result. Re-validate "
            "on out-of-sample real (or more realistic synthetic) data before "
            "quoting this number to anyone."
        )

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, "models/risk_model.joblib")


if __name__ == "__main__":
    main()
