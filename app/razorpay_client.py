"""
Thin wrapper around the real Razorpay REST API.

Written directly against Razorpay's documented API contract and
unit-tested against a mocked HTTP client -- not yet smoke-tested
against a live Razorpay account. Run the two functions below once with
TEST MODE keys (Dashboard -> Settings -> API Keys -> Generate Test Key)
before trusting this in production.

Docs: https://razorpay.com/docs/api/orders/create/
      https://razorpay.com/docs/api/payments/capture/
      https://razorpay.com/docs/payments/payments/capture-settings/
"""

import os

import httpx

RAZORPAY_API_BASE = "https://api.razorpay.com/v1"


class RazorpayAPIError(RuntimeError):
    pass


def get_key_id() -> str:
    return os.environ.get("RAZORPAY_KEY_ID", "")


def _auth():
    key_id = os.environ.get("RAZORPAY_KEY_ID", "")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise RazorpayAPIError(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set. "
            "Get them from Dashboard -> Settings -> API Keys."
        )
    return (key_id, key_secret)


def create_order(amount_rupees: float, receipt: str, notes: dict | None = None) -> dict:
    """
    Creates a Razorpay order with capture held back (payment_capture:0),
    so the payment lands in 'authorized' state after the bank approves
    it, and stays there until we explicitly capture it. This is what
    gives the risk engine a real window to act in, instead of finding
    out about a transaction after the money already moved.
    """
    response = httpx.post(
        f"{RAZORPAY_API_BASE}/orders",
        auth=_auth(),
        json={
            "amount": int(round(amount_rupees * 100)),  # rupees -> paise
            "currency": "INR",
            "receipt": receipt,
            "payment_capture": 0,  # manual capture -- see module docstring
            "notes": notes or {},
        },
        timeout=10.0,
    )

    if response.status_code >= 400:
        raise RazorpayAPIError(f"create_order failed ({response.status_code}): {response.text}")

    return response.json()


def capture_payment(payment_id: str, amount_rupees: float, currency: str = "INR") -> dict:
    """
    Moves a payment from 'authorized' to 'captured'. The amount must
    exactly match the authorized amount (Razorpay rejects partial
    mismatches for this call). Call this only for an 'allow' decision.
    """
    response = httpx.post(
        f"{RAZORPAY_API_BASE}/payments/{payment_id}/capture",
        auth=_auth(),
        json={
            "amount": int(round(amount_rupees * 100)),  # rupees -> paise
            "currency": currency,
        },
        timeout=10.0,
    )

    if response.status_code >= 400:
        raise RazorpayAPIError(f"capture_payment failed ({response.status_code}): {response.text}")

    return response.json()


def refund_payment(payment_id: str, amount_rupees: float, notes: dict | None = None) -> dict:
    """
    Refunds an ALREADY-CAPTURED payment. Only meaningful for a payment
    that reached 'captured' before scoring ever saw it (an auto-capture
    account, or a held payment whose capture window already lapsed) --
    for a still-'authorized' held payment, don't refund it, just leave
    it uncaptured and it auto-refunds on its own via Razorpay's normal
    timeout. Called from the human-decision endpoint when someone
    reviews a flagged transaction and decides to reverse it.
    """
    response = httpx.post(
        f"{RAZORPAY_API_BASE}/payments/{payment_id}/refund",
        auth=_auth(),
        json={
            "amount": int(round(amount_rupees * 100)),
            "speed": "normal",
            "notes": notes or {},
        },
        timeout=10.0,
    )

    if response.status_code >= 400:
        raise RazorpayAPIError(f"refund_payment failed ({response.status_code}): {response.text}")

    return response.json()


def notify_slack(text: str) -> bool:
    """
    Fire-and-forget Slack alert via an Incoming Webhook
    (https://api.slack.com/messaging/webhooks). Returns False (never
    raises) if SLACK_WEBHOOK_URL isn't configured, so a missing/broken
    alert integration never breaks the actual scoring/capture pipeline
    -- getting money right matters more than getting notified.
    """
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not slack_webhook_url:
        return False

    try:
        response = httpx.post(slack_webhook_url, json={"text": text}, timeout=5.0)
        return response.status_code < 400
    except httpx.HTTPError:
        return False
