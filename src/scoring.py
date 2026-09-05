"""
The single scoring pipeline: entity stats -> features -> model + rules
-> final advisory decision -> persisted record.

Both app/api.py's HTTP endpoint and app/webhook.py's async Razorpay
consumer call score_transaction(), so there is exactly one place that
combines the model score, graph score, and rule score into a decision.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src.database import (
    get_entity_stats,
    get_transaction_by_id,
    insert_transaction,
    to_iso,
)
from src.decisioning import choose_action, combine_scores
from src.realtime_features import build_realtime_features
from src.rules import evaluate_rules

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "risk_model.joblib"

_MODEL_BUNDLE_CACHE = None


class ModelNotTrainedError(RuntimeError):
    pass


class InvalidTransactionError(ValueError):
    """
    Raised for a payload that can't be scored safely -- e.g. a
    non-positive amount. This matters beyond input hygiene: a
    negative/zero amount produces amount_log = log1p(negative) = NaN,
    and the model was observed to silently return a low, seemingly
    valid score for that NaN input rather than erroring. That's a real
    gap an attacker could exploit deliberately (submit a nonsensical
    amount specifically to force a low risk score), so this is
    rejected up front rather than left to "the model probably handles
    it".
    """
    pass


def get_model_bundle():
    """
    Loads the model once and caches it in memory. Re-loading a
    joblib file on every request is fine for a demo, but adds real,
    measurable latency under production webhook volume.
    """
    global _MODEL_BUNDLE_CACHE

    if _MODEL_BUNDLE_CACHE is None:
        if not MODEL_PATH.exists():
            raise ModelNotTrainedError(
                "Model not found. Run: python -m src.train"
            )
        _MODEL_BUNDLE_CACHE = joblib.load(MODEL_PATH)

    return _MODEL_BUNDLE_CACHE


def _record_to_response(record: dict) -> dict:
    evidence = record["evidence"]
    if isinstance(evidence, str):
        evidence = json.loads(evidence)

    return {
        "transaction_id": record["transaction_id"],
        "timestamp": record["timestamp"],
        "risk_score": round(record["final_risk_score"], 4),
        "model_score": record["model_score"],
        "graph_score": record["graph_score"],
        "rule_score": record["rule_score"],
        "action": record["action"],
        "evidence": evidence,
        "entity_stats_24h": None,
        "advisory": True,
        "duplicate": True,
    }


def score_transaction(payload: dict) -> dict:
    """
    payload must contain: user_id, device_id, ip_id, merchant_id,
    amount, and optionally transaction_id / timestamp.

    Returns the same response shape the /transactions endpoint returns,
    and persists the scored transaction to the database.

    Re-scoring an already-seen transaction_id is idempotent: it returns
    the originally stored decision instead of raising, which matters
    for real Razorpay webhook retries (Razorpay does not guarantee
    exactly-once delivery).
    """
    payload = dict(payload)  # never mutate the caller's dict

    amount = payload.get("amount")
    if amount is None or not np.isfinite(amount) or amount <= 0:
        raise InvalidTransactionError(
            f"amount must be a finite positive number, got: {amount!r}"
        )

    if not payload.get("transaction_id"):
        payload["transaction_id"] = f"tx_live_{uuid.uuid4().hex[:12]}"

    existing = get_transaction_by_id(payload["transaction_id"])
    if existing is not None:
        return _record_to_response(existing)

    timestamp = payload.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if isinstance(timestamp, datetime):
        timestamp = to_iso(timestamp)
    payload["timestamp"] = timestamp

    stats = get_entity_stats(
        user_id=payload["user_id"],
        device_id=payload["device_id"],
        ip_id=payload["ip_id"],
        merchant_id=payload["merchant_id"],
        timestamp=payload["timestamp"],
    )

    model_bundle = get_model_bundle()
    model = model_bundle["model"]
    model_features = model_bundle["features"]

    live_features = build_realtime_features(payload, stats)

    model_score = float(model.predict_proba(live_features[model_features])[0, 1])
    graph_score = float(live_features["graph_risk_score"].iloc[0])
    rule_score, evidence = evaluate_rules(payload, stats)

    final_risk_score = combine_scores(model_score, graph_score, rule_score)
    action = choose_action(final_risk_score)

    evidence = list(evidence)
    evidence.insert(0, f"ML model score: {model_score:.4f}")
    evidence.insert(1, f"Graph relationship score: {graph_score:.4f}")
    evidence.insert(2, f"Rule score: {rule_score:.4f}")

    record = {
        **payload,
        "model_score": model_score,
        "graph_score": graph_score,
        "rule_score": rule_score,
        "final_risk_score": final_risk_score,
        "action": action,
        "evidence": json.dumps(evidence),
        # Explicit, not incidental: app/webhook.py sets
        # _razorpay_capture_state based on which Razorpay event
        # triggered this (payment.authorized vs payment.captured), so
        # /reviews/{id}/decision later knows whether "release" means
        # "don't capture" or "issue a refund". Defaults to "authorized"
        # for the plain /transactions demo endpoint, which has no real
        # Razorpay payment behind it.
        "capture_state": payload.get("_razorpay_capture_state", "authorized"),
    }

    try:
        insert_transaction(record)
    except Exception as error:
        # Race: two callers scored the same brand-new transaction_id at
        # the same time. Whoever loses the race just returns the
        # winner's already-persisted result instead of failing.
        existing = get_transaction_by_id(payload["transaction_id"])
        if existing is not None:
            return _record_to_response(existing)
        raise

    return {
        "transaction_id": payload["transaction_id"],
        "timestamp": payload["timestamp"],
        "risk_score": round(final_risk_score, 4),
        "model_score": model_score,
        "graph_score": graph_score,
        "rule_score": rule_score,
        "action": action,
        "evidence": evidence,
        "entity_stats_24h": stats,
        "advisory": True,
        "duplicate": False,
    }
