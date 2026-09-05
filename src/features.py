"""
Training-time feature engineering.

DESIGN RULE: every feature here must use only information available
strictly before the transaction happened, and must have a byte-for-byte
equivalent in src/database.get_entity_stats() + src/realtime_features.py.
Otherwise the model trains on signals it never sees at serving time
(train/serve skew).

WHY TWO WINDOWS (24h AND 30d): this dataset's ring members transact a
median of ~4.5 days apart, not in a tight burst -- only ~16% of
consecutive same-ring-device transactions fall inside a 24h window of
each other. A 24h-only model catches roughly 29% of rings; adding the
30-day window is what catches the slow-drip pattern. Real coordinated
fraud shows up both ways -- fast card-testing bursts and slow-drip
identity reuse -- so a production system needs both window lengths.
"""

from collections import deque

import numpy as np
import pandas as pd

FEATURES = [
    "amount_log",
    "user_txn_count_24h",
    "device_txn_count_24h",
    "ip_txn_count_24h",
    "merchant_txn_count_24h",
    "device_unique_users_24h",
    "ip_unique_users_24h",
    "device_unique_users_30d",
    "ip_unique_users_30d",
    "graph_risk_score",
]

WINDOW_24H_NS = pd.Timedelta(hours=24).to_timedelta64()
WINDOW_30D_NS = pd.Timedelta(days=30).to_timedelta64()

# How many distinct people sharing one device/IP in the 30d window
# before the graph score saturates at 1.0. A knob to tune against real
# data once you have it -- kept as one named constant, used identically
# in src/realtime_features.py.
GRAPH_SCORE_SATURATION = 10.0


def _prior_window_stats(df: pd.DataFrame, entity_col: str, window_ns, need_unique_users: bool):
    """
    For every row, computes -- using ONLY transactions strictly BEFORE
    it in time, never the row itself and never a future row -- the:
      - count of prior transactions on the same entity in the window
      - count of distinct users behind that entity in the window

    This mirrors exactly the SQL in src/database.py
    ("timestamp < current AND timestamp >= current - window"), so a
    model trained on this can never see a different feature
    distribution than what the live API computes.
    """
    n = len(df)
    counts = np.zeros(n, dtype=float)
    uniques = np.zeros(n, dtype=float)

    for _, group in df.groupby(entity_col, sort=False):
        idx = group.index.to_numpy()
        times = group["timestamp"].to_numpy()
        users = group["user_id"].to_numpy()

        window = deque()  # (time, user) currently inside the trailing window
        user_counter: dict = {}

        for pos in range(len(idx)):
            t = times[pos]
            cutoff = t - window_ns

            while window and window[0][0] < cutoff:
                _, old_user = window.popleft()
                user_counter[old_user] -= 1
                if user_counter[old_user] == 0:
                    del user_counter[old_user]

            # Record state BEFORE this row's own edge is added -> "prior"
            counts[idx[pos]] = len(window)
            if need_unique_users:
                uniques[idx[pos]] = len(user_counter)

            window.append((t, users[pos]))
            user_counter[users[pos]] = user_counter.get(users[pos], 0) + 1

    return counts, uniques


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values("timestamp").reset_index(drop=True)

    x["amount_log"] = np.log1p(x["amount"])

    # Short window: burst velocity, all four entities.
    for col, out_count in [
        ("user_id", "user_txn_count_24h"),
        ("device_id", "device_txn_count_24h"),
        ("ip_id", "ip_txn_count_24h"),
        ("merchant_id", "merchant_txn_count_24h"),
    ]:
        counts, _ = _prior_window_stats(x, col, WINDOW_24H_NS, need_unique_users=False)
        x[out_count] = counts

    # Short-window identity sharing (catches fast card-testing bursts).
    _, x["device_unique_users_24h"] = _prior_window_stats(
        x, "device_id", WINDOW_24H_NS, need_unique_users=True
    )
    _, x["ip_unique_users_24h"] = _prior_window_stats(
        x, "ip_id", WINDOW_24H_NS, need_unique_users=True
    )

    # Long-window identity sharing (catches slow-drip device/IP reuse).
    _, x["device_unique_users_30d"] = _prior_window_stats(
        x, "device_id", WINDOW_30D_NS, need_unique_users=True
    )
    _, x["ip_unique_users_30d"] = _prior_window_stats(
        x, "ip_id", WINDOW_30D_NS, need_unique_users=True
    )

    # graph_risk_score: distinct people who touched this device OR this
    # IP in the last 30 days. Computed from the same numbers the live
    # API has (see src/database.py + src/realtime_features.py), so
    # there is no separate "offline graph" the model relies on that
    # production can't reproduce in real time.
    shared_identity_30d = x["device_unique_users_30d"] + x["ip_unique_users_30d"]
    x["graph_risk_score"] = np.minimum(shared_identity_30d / GRAPH_SCORE_SATURATION, 1.0)

    return x
