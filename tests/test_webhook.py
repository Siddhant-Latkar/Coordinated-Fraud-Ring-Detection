"""
Tests for the real Razorpay automation path: /checkout/create-order ->
webhook ingestion -> async scoring -> auto-capture/hold -> the human
/reviews/{id}/decision resolution endpoint.

api.razorpay.com is not reachable from this environment (and there are
no real Razorpay credentials to test with), so every Razorpay HTTP call
is mocked via unittest.mock.patch. What IS verified for real: signature
verification, event routing, the checkout_sessions device/IP join, the
full scoring pipeline, database persistence, and the capture/hold/
resolve decision logic -- everything except the actual bytes sent over
the wire to Razorpay's servers.
"""

import hashlib
import hmac
import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["RISK_DB_PATH"] = _tmp_db.name
os.environ["RAZORPAY_WEBHOOK_SECRET"] = "whsec_test_secret"
os.environ["RAZORPAY_KEY_ID"] = "rzp_test_fake"
os.environ["RAZORPAY_KEY_SECRET"] = "fake_secret"

from fastapi.testclient import TestClient  # noqa: E402

from src.checkout_sessions import initialize_checkout_sessions_table  # noqa: E402
from src.database import get_pending_reviews, initialize_database  # noqa: E402

initialize_database()
initialize_checkout_sessions_table()

from app.api import app  # noqa: E402
from app.webhook import resolve_transaction  # noqa: E402

WEBHOOK_SECRET = "whsec_test_secret"


def sign(raw_body: bytes) -> str:
    return hmac.new(WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()


def make_webhook_body(event: str, payment_id: str, order_id: str = "", email: str = "u@example.com", amount_paise: int = 90000):
    return json.dumps({
        "event": event,
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "order_id": order_id,
                    "email": email,
                    "amount": amount_paise,
                    "currency": "INR",
                    "notes": {"merchant_id": "m_test"},
                }
            }
        }
    }).encode()


@pytest.fixture(scope="module")
def client():
    # Module-scoped: a fresh TestClient per test would tear down and
    # recreate the event loop, breaking the module-level asyncio.Queue
    # in app/webhook.py.
    with TestClient(app) as test_client:
        yield test_client


def test_webhook_rejects_bad_signature(client):
    body = make_webhook_body("payment.authorized", "pay_bad_sig")
    response = client.post("/webhooks/razorpay", content=body, headers={
        "x-razorpay-signature": "not-the-real-signature",
        "x-razorpay-event-id": "evt_bad_sig",
    })
    assert response.status_code == 400


def test_webhook_ignores_unhandled_event_types(client):
    body = json.dumps({"event": "refund.processed", "payload": {}}).encode()
    response = client.post("/webhooks/razorpay", content=body, headers={
        "x-razorpay-signature": sign(body),
        "x-razorpay-event-id": "evt_unhandled",
    })
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


def test_duplicate_event_id_is_not_reprocessed(client):
    body = make_webhook_body("payment.authorized", "pay_dup_001", order_id="order_dup")

    with patch("app.webhook.capture_payment") as mock_capture:
        first = client.post("/webhooks/razorpay", content=body, headers={
            "x-razorpay-signature": sign(body),
            "x-razorpay-event-id": "evt_dup_001",
        })
        second = client.post("/webhooks/razorpay", content=body, headers={
            "x-razorpay-signature": sign(body),
            "x-razorpay-event-id": "evt_dup_001",
        })
        time.sleep(0.3)

    assert first.status_code == 200
    assert second.json()["status"] == "duplicate_ignored"
    assert mock_capture.call_count == 1


def test_checkout_order_then_ring_burst_escalates_and_holds(client):
    """
    The core proof: create real checkout orders (capturing device
    fingerprint), then simulate Razorpay sending payment.authorized for
    each -- several sharing the same device but different accounts.
    The first should auto-capture; later ones in the ring should be
    held (review/step_up), never captured, and show up as pending
    reviews.
    """
    def fake_create_order(amount_rupees, receipt, notes=None):
        fake_create_order.n = getattr(fake_create_order, "n", 0) + 1
        return {"id": f"order_test_ring_{fake_create_order.n}", "amount": int(amount_rupees * 100), "currency": "INR"}

    captured = []

    def fake_capture(payment_id, amount_rupees, currency="INR"):
        captured.append(payment_id)
        return {"id": payment_id, "status": "captured"}

    with patch("app.api.create_order", side_effect=fake_create_order), \
         patch("app.webhook.capture_payment", side_effect=fake_capture):

        order_ids = []
        for i in range(1, 6):
            response = client.post("/checkout/create-order", json={
                "user_id": f"ring_test_user_{i}@example.com",
                "device_id": "shared_ring_device_test",
                "amount": 900,
            })
            assert response.status_code == 200
            order_ids.append(response.json()["order_id"])

        for i, order_id in enumerate(order_ids, start=1):
            body = make_webhook_body(
                "payment.authorized",
                payment_id=f"pay_test_ring_{i}",
                order_id=order_id,
                email=f"ring_test_user_{i}@example.com",
            )
            resp = client.post("/webhooks/razorpay", content=body, headers={
                "x-razorpay-signature": sign(body),
                "x-razorpay-event-id": f"evt_test_ring_{i}",
            })
            assert resp.status_code == 200

        time.sleep(0.5)

    # First member of the burst looked like a normal, isolated
    # transaction and should have been auto-captured.
    assert "pay_test_ring_1" in captured

    # Later members shared a device with several other accounts and
    # should have been held, not captured.
    pending_ids = {row["transaction_id"] for row in get_pending_reviews()}
    assert "pay_test_ring_5" in pending_ids
    for held_id in pending_ids:
        assert held_id not in captured


def test_resolve_transaction_approve_captures_held_payment():
    with patch("app.webhook.capture_payment") as mock_capture:
        mock_capture.return_value = {"status": "captured"}
        result = resolve_transaction("pay_test_ring_5", "approve")

    assert result["status"] == "resolved"
    assert result["resolution"] == "captured"
    mock_capture.assert_called_once()


def test_resolve_transaction_release_does_not_capture():
    result = resolve_transaction("pay_test_ring_4", "release")

    assert result["status"] == "resolved"
    assert result["resolution"] == "released_uncaptured"


def test_resolve_transaction_is_idempotent():
    # pay_test_ring_5 was already resolved above.
    result = resolve_transaction("pay_test_ring_5", "approve")
    assert result["status"] == "already_resolved"


def test_resolve_transaction_unknown_id():
    result = resolve_transaction("does_not_exist_at_all", "approve")
    assert result["status"] == "not_found"


def test_missing_razorpay_credentials_gives_clean_502(client):
    with patch.dict(os.environ, {"RAZORPAY_KEY_ID": "", "RAZORPAY_KEY_SECRET": ""}):
        with patch("app.api.create_order") as mock_create_order:
            from app.razorpay_client import RazorpayAPIError
            mock_create_order.side_effect = RazorpayAPIError("credentials not set")

            response = client.post("/checkout/create-order", json={
                "user_id": "u1", "device_id": "d1", "amount": 100
            })

    assert response.status_code == 502
