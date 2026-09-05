"""
Stores the device_id/ip_id/user_id captured on the checkout page at
order-creation time, keyed by Razorpay's order_id, so the webhook
handler (which only receives order_id) can join them back in.

Shares get_connection() with src/database.py -- same SQLite file,
separate table.
"""

from src.database import get_connection


def initialize_checkout_sessions_table():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS checkout_sessions (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                ip_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_checkout_context(order_id: str, user_id: str, device_id: str, ip_id: str, created_at: str):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO checkout_sessions (order_id, user_id, device_id, ip_id, created_at)
            VALUES (:order_id, :user_id, :device_id, :ip_id, :created_at)
            ON CONFLICT(order_id) DO UPDATE SET
                user_id=excluded.user_id,
                device_id=excluded.device_id,
                ip_id=excluded.ip_id
        """, {
            "order_id": order_id,
            "user_id": user_id,
            "device_id": device_id,
            "ip_id": ip_id,
            "created_at": created_at,
        })


def get_checkout_context(order_id: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute("""
            SELECT * FROM checkout_sessions WHERE order_id = ?
        """, (order_id,)).fetchone()

    return dict(row) if row else None
