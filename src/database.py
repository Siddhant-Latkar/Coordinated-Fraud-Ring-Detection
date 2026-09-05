"""
SQLite-backed transaction store + windowed entity stats.

"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Overridable so tests (and CI) never write into the real demo database.
DATABASE_PATH = Path(os.environ.get("RISK_DB_PATH", "data/risk_manager.db"))


def to_iso(dt: datetime) -> str:
    """
    Canonical timestamp format used everywhere a transaction time is
    stored or compared. Always includes microseconds explicitly so two
    ISO-8601 strings can be compared as plain text and get the correct
    chronological order -- no DB-side date parsing required, which is
    what let SQLite's datetime() silently truncate sub-second bursts
    to whole seconds in an earlier version of this file.
    """
    return dt.isoformat(timespec="microseconds")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


@contextmanager
def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                ip_id TEXT NOT NULL,
                merchant_id TEXT NOT NULL,
                amount REAL NOT NULL,

                model_score REAL NOT NULL,
                graph_score REAL NOT NULL,
                rule_score REAL NOT NULL,
                final_risk_score REAL NOT NULL,

                action TEXT NOT NULL,
                evidence TEXT NOT NULL,

                capture_state TEXT,
                resolution TEXT,
                resolved_at TEXT
            )
        """)

        # Idempotent migration for databases created before capture-state
        # / resolution tracking existed. SQLite has no "ADD COLUMN IF NOT
        # EXISTS", so we just try and swallow the "duplicate column" error.
        for column_def in ("capture_state TEXT", "resolution TEXT", "resolved_at TEXT"):
            try:
                connection.execute(f"ALTER TABLE transactions ADD COLUMN {column_def}")
            except sqlite3.OperationalError as error:
                if "duplicate column" not in str(error).lower():
                    raise

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
            ON transactions(timestamp)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_user
            ON transactions(user_id)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_device
            ON transactions(device_id)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_ip
            ON transactions(ip_id)
        """)


def get_pending_reviews():
    """
    Transactions that were held (action in review/step_up) and never
    resolved by a human -- i.e. still sitting in Razorpay's 'authorized'
    state, unresolved, waiting to either be captured or to lapse into
    Razorpay's automatic refund once the capture window closes.
    """
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM transactions
            WHERE action IN ('review', 'step_up')
              AND resolution IS NULL
            ORDER BY timestamp DESC
        """).fetchall()

    return [dict(row) for row in rows]


def mark_transaction_resolved(transaction_id: str, resolution: str):
    with get_connection() as connection:
        connection.execute("""
            UPDATE transactions
            SET resolution = ?, resolved_at = ?
            WHERE transaction_id = ?
        """, (resolution, to_iso(datetime.now(timezone.utc)), transaction_id))


def insert_transaction(record: dict):
    record = dict(record)
    # Explicit, not an accidental pass-through: the plain /transactions
    # demo endpoint never sets this (there's no real Razorpay payment
    # behind it), so it defaults to "authorized" -- harmless there since
    # that path never calls /reviews/{id}/decision against a real
    # payment anyway. Real webhook traffic always sets this explicitly
    # in app/webhook.py based on which event triggered scoring.
    record.setdefault("capture_state", "authorized")

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO transactions (
                transaction_id,
                timestamp,
                user_id,
                device_id,
                ip_id,
                merchant_id,
                amount,
                model_score,
                graph_score,
                rule_score,
                final_risk_score,
                action,
                evidence,
                capture_state
            )
            VALUES (
                :transaction_id,
                :timestamp,
                :user_id,
                :device_id,
                :ip_id,
                :merchant_id,
                :amount,
                :model_score,
                :graph_score,
                :rule_score,
                :final_risk_score,
                :action,
                :evidence,
                :capture_state
            )
        """, record)


def get_entity_stats(
    user_id: str,
    device_id: str,
    ip_id: str,
    merchant_id: str,
    timestamp: str,
) -> dict:
    """
    Retrieves recent history STRICTLY BEFORE the current transaction
    time, using the preceding window for live velocity/identity
    features.

    The keys returned here are named to match src/features.py exactly
    (user_txn_count_24h, device_txn_count_24h, ip_txn_count_24h,
    merchant_txn_count_24h, device_unique_users_24h/_30d,
    ip_unique_users_24h/_30d). Do not rename one side without the
    other -- that's how train/serve skew creeps back in.
    """
    now = parse_iso(timestamp)
    now_iso = to_iso(now)

    with get_connection() as connection:
        def _entity_counts(column: str, value: str, lookback: timedelta):
            window_start_iso = to_iso(now - lookback)
            row = connection.execute(f"""
                SELECT
                    COUNT(*) AS txn_count,
                    COUNT(DISTINCT user_id) AS unique_users
                FROM transactions
                WHERE {column} = ?
                  AND timestamp >= ?
                  AND timestamp < ?
            """, (value, window_start_iso, now_iso)).fetchone()
            return int(row["txn_count"]), int(row["unique_users"])

        day = timedelta(hours=24)
        month = timedelta(days=30)

        user_count, _ = _entity_counts("user_id", user_id, day)
        device_count, device_unique_users_24h = _entity_counts("device_id", device_id, day)
        ip_count, ip_unique_users_24h = _entity_counts("ip_id", ip_id, day)
        merchant_count, _ = _entity_counts("merchant_id", merchant_id, day)

        # 30-day window: catches slow-drip device/IP reuse that a 24h
        # velocity check misses entirely (see src/features.py docstring
        # for why both windows matter).
        _, device_unique_users_30d = _entity_counts("device_id", device_id, month)
        _, ip_unique_users_30d = _entity_counts("ip_id", ip_id, month)

        return {
            "user_txn_count_24h": user_count,
            "device_txn_count_24h": device_count,
            "ip_txn_count_24h": ip_count,
            "merchant_txn_count_24h": merchant_count,
            "device_unique_users_24h": device_unique_users_24h,
            "ip_unique_users_24h": ip_unique_users_24h,
            "device_unique_users_30d": device_unique_users_30d,
            "ip_unique_users_30d": ip_unique_users_30d,
        }


def get_transaction_by_id(transaction_id: str):
    with get_connection() as connection:
        row = connection.execute("""
            SELECT * FROM transactions WHERE transaction_id = ?
        """, (transaction_id,)).fetchone()

    return dict(row) if row else None


def get_recent_transactions(limit: int = 200):
    with get_connection() as connection:
        # Plain string ORDER BY, no DB-side date function: safe because
        # every timestamp is written via to_iso(), which always includes
        # explicit microseconds, so ISO-8601 strings sort chronologically
        # as plain text.
        rows = connection.execute("""
            SELECT *
            FROM transactions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]
