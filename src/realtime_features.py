"""
Live-serving feature builder. Every field read from `stats` must come
from src/database.get_entity_stats(), and every name/formula here must
match src/features.py exactly, or the model scores production traffic
on a feature distribution it was never evaluated against.
"""

import numpy as np
import pandas as pd

from src.features import GRAPH_SCORE_SATURATION


def build_realtime_features(transaction: dict, stats: dict) -> pd.DataFrame:
    """
    Builds the feature vector the trained model expects, using the
    exact same field names and formulas as src/features.py.
    """
    shared_identity_30d = (
        stats["device_unique_users_30d"] + stats["ip_unique_users_30d"]
    )
    graph_risk_score = min(shared_identity_30d / GRAPH_SCORE_SATURATION, 1.0)

    row = {
        "amount_log": np.log1p(transaction["amount"]),
        "user_txn_count_24h": float(stats["user_txn_count_24h"]),
        "device_txn_count_24h": float(stats["device_txn_count_24h"]),
        "ip_txn_count_24h": float(stats["ip_txn_count_24h"]),
        "merchant_txn_count_24h": float(stats["merchant_txn_count_24h"]),
        "device_unique_users_24h": float(stats["device_unique_users_24h"]),
        "ip_unique_users_24h": float(stats["ip_unique_users_24h"]),
        "device_unique_users_30d": float(stats["device_unique_users_30d"]),
        "ip_unique_users_30d": float(stats["ip_unique_users_30d"]),
        "graph_risk_score": graph_risk_score,
    }

    # Keep every computed column (not just the model's FEATURES) so
    # callers can still read graph_risk_score etc. for evidence. The
    # API slices down to the model's own feature list before scoring.
    return pd.DataFrame([row])
