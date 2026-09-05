"""
Real Razorpay webhook ingestion, wired to real consequences.

Design points that matter in production and are easy to get wrong:

1. VERIFY THE RAW BODY. Razorpay signs the exact bytes it sent
   (HMAC-SHA256, hex, keyed with your webhook secret, header
   X-Razorpay-Signature). If you parse the body to JSON and re-serialize
   before verifying, whitespace/key-order differences break the
   signature. We read `await request.body()` before any JSON parsing.

2. ACK FAST, SCORE ASYNC. Every event that gets a non-2xx response is
   treated by Razorpay as a delivery failure and retried with
   exponential backoff. Running model inference (or anything slow or
   flaky) inside the handler risks timeouts turning into duplicate
   retries. We push to an in-process queue and return 200 immediately;
   a background worker does the actual scoring AND the real action.

3. DEDUPE BY x-razorpay-event-id. Razorpay does not guarantee
   exactly-once delivery or in-order delivery. We keep a small
   in-memory set of processed event IDs (swap for a Redis SET or a
   dedicated DB table in production -- this in-memory version resets
   on restart, which is fine for a demo, not for production).

4. WE HANDLE BOTH payment.authorized AND payment.captured, because
   Razorpay accounts default to AUTO-CAPTURE and most merchants never
   change that. The two events mean very different things for what we
   can actually do:
     - payment.authorized (manual capture, via /checkout/create-order's
       payment_capture:0): the payment is on HOLD. review/step_up means
       we genuinely PREVENT the money from moving -- we just never call
       capture. This is the strong case; see app/razorpay_client.py.
     - payment.captured (auto-capture, the default): the money has
       ALREADY moved by the time we see it. review/step_up here can
       only mean "flag for a human, who can choose to refund" --
       reactive, not preventive. We still score and alert on it rather
       than ignoring this event type.

5. amount IS IN PAISE in the webhook payload; divide by 100.

6. device_id / ip_id come from src.checkout_sessions, populated by the
   /checkout/create-order endpoint in app/api.py at order-creation
   time -- not from the webhook payload, which never contains them.
"""

import asyncio
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.razorpay_client import RazorpayAPIError, capture_payment, notify_slack, refund_payment
from src.checkout_sessions import get_checkout_context
from src.database import get_transaction_by_id, mark_transaction_resolved
from src.scoring import InvalidTransactionError, ModelNotTrainedError, score_transaction

router = APIRouter()

# Read fresh at request time rather than cached, so an env var set
# after import still takes effect.
def _webhook_secret() -> str:
    return os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")

# In-memory dedup/queue -- resets on restart. Swap for Redis/a DB table
# and a real queue (SQS, Celery, etc.) before production use.
_seen_event_ids: set[str] = set()
_scoring_queue: "asyncio.Queue[dict]" = asyncio.Queue()


def verify_razorpay_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    if not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _lookup_checkout_context(order_id: str) -> dict:
    """
    Reads the device/IP captured at order-creation time (see
    app/api.py's /checkout/create-order). Falls back to a clearly-fake
    placeholder if this order was never created through that endpoint
    (e.g. a payment link, or a test event) -- logged loudly so a
    missing integration is obvious rather than silently scoring on
    garbage.
    """
    context = get_checkout_context(order_id)
    if context is not None:
        return {"device_id": context["device_id"], "ip_id": context["ip_id"]}

    print(
        f"webhook: no checkout_sessions row for order_id={order_id!r} -- "
        f"this order wasn't created via /checkout/create-order, so "
        f"device/IP signals are meaningless for this transaction."
    )
    return {
        "device_id": f"unknown_device_for_{order_id}",
        "ip_id": f"unknown_ip_for_{order_id}",
    }


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    event_id = request.headers.get("x-razorpay-event-id", "")

    if not verify_razorpay_signature(raw_body, signature, _webhook_secret()):
        # Do NOT process an unverified body. Returning 400 (not 500)
        # tells Razorpay this is a permanent rejection, not a transient
        # failure worth retrying.
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    if event_id and event_id in _seen_event_ids:
        # Already handled this exact event on a prior delivery attempt.
        # Ack it again without reprocessing.
        return {"status": "duplicate_ignored"}

    payload = json.loads(raw_body)
    event = payload.get("event", "")

    if event not in {"payment.authorized", "payment.captured", "payment.failed"}:
        # Ack everything else too -- an unhandled event type is not an
        # error, and a non-2xx here just triggers pointless retries.
        return {"status": "ignored", "event": event}

    payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})

    if event == "payment.failed":
        return {"status": "logged_failed"}

    # event is payment.authorized or payment.captured. capture_state
    # tells the worker (and later, a human resolving a held review)
    # which Razorpay action is still possible for this transaction:
    # "authorized" -> capture_payment() is available, refund is not.
    # "captured"   -> refund_payment() is available, capture is moot.
    capture_state = "authorized" if event == "payment.authorized" else "captured"

    order_id = payment_entity.get("order_id", "")
    checkout_context = _lookup_checkout_context(order_id)

    transaction_payload = {
        "transaction_id": payment_entity.get("id"),
        "user_id": payment_entity.get("email") or payment_entity.get("contact") or "unknown_user",
        "device_id": checkout_context["device_id"],
        "ip_id": checkout_context["ip_id"],
        "merchant_id": payment_entity.get("notes", {}).get("merchant_id", "default_merchant"),
        "amount": payment_entity.get("amount", 0) / 100.0,  # paise -> rupees
        "_razorpay_currency": payment_entity.get("currency", "INR"),
        "_razorpay_capture_state": capture_state,
    }

    if event_id:
        _seen_event_ids.add(event_id)

    await _scoring_queue.put(transaction_payload)

    # Ack immediately. Scoring AND the real capture/hold decision
    # happen in the background worker below.
    return {"status": "queued"}


async def run_scoring_worker():
    """
    Background consumer for the scoring queue. Started once at app
    startup (see app/api.py). In production, replace this loop with a
    real worker process (Celery/RQ/an SQS consumer) so scoring survives
    an API-process restart and can scale independently of ingestion.
    """
    while True:
        transaction_payload = await _scoring_queue.get()
        try:
            result = score_transaction(transaction_payload)
            await _act_on_decision(result, transaction_payload)
        except ModelNotTrainedError:
            print("Scoring worker: model not trained yet, dropping transaction.")
        except InvalidTransactionError as error:
            print(
                f"Scoring worker: rejected invalid transaction "
                f"{transaction_payload.get('transaction_id')}: {error}"
            )
        except Exception as error:  # noqa: BLE001
            print(f"Scoring worker error for {transaction_payload.get('transaction_id')}: {error}")
        finally:
            _scoring_queue.task_done()


async def _act_on_decision(result: dict, transaction_payload: dict):
    """
    This is the piece that makes the system an actual risk *manager*
    instead of a risk *logger*: it changes what happens to the money.

      allow, capture_state=authorized -> capture the payment now.
      allow, capture_state=captured   -> already captured, nothing to do.
      review/step_up, authorized      -> do nothing (leave it
                                          'authorized'); alert a human,
                                          who can capture or release it
                                          via /reviews/{id}/decision
                                          within Razorpay's capture
                                          window (a few days).
      review/step_up, captured        -> money already moved (this
                                          account is on auto-capture);
                                          alert a human, who can refund
                                          it via the same endpoint.

    Never auto-refund/auto-release silently -- every non-'allow'
    outcome surfaces to a human and waits for an explicit decision.
    That's the defense-only, advisory line this project has held
    throughout: the model recommends, it doesn't get unilateral
    authority over someone's money.
    """
    action = result["action"]
    transaction_id = result["transaction_id"]
    amount = transaction_payload["amount"]
    currency = transaction_payload.get("_razorpay_currency", "INR")
    capture_state = transaction_payload.get("_razorpay_capture_state", "authorized")

    if action == "allow":
        if capture_state == "authorized":
            try:
                capture_payment(transaction_id, amount, currency)
                print(f"[CAPTURED] {transaction_id} | risk={result['risk_score']}")
            except RazorpayAPIError as error:
                # Scoring said "allow" but the capture call itself failed
                # (network issue, already-expired auth window, etc). This
                # needs a human, not a silent retry loop.
                print(f"[CAPTURE FAILED] {transaction_id}: {error} -- needs manual capture via Dashboard.")
        else:
            print(f"[ALLOW] {transaction_id} | risk={result['risk_score']} | already captured, no action needed")
        return

    # review or step_up: intentionally left unresolved for a human.
    urgency = "URGENT" if action == "step_up" else "REVIEW"
    print(
        f"[{urgency}] {transaction_id} | risk={result['risk_score']} | "
        f"action={action} | capture_state={capture_state} | "
        f"evidence={result['evidence']}"
    )

    resolve_hint = (
        "capture it (approve) or leave it to lapse/auto-refund (release)"
        if capture_state == "authorized"
        else "mark it fine (approve) or refund it (release)"
    )
    alert_sent = notify_slack(
        f":rotating_light: *{urgency}* -- transaction `{transaction_id}` "
        f"needs a decision (risk {result['risk_score']}, {capture_state}).\n"
        f"Evidence: {', '.join(result['evidence'])}\n"
        f"Resolve: POST /reviews/{transaction_id}/decision "
        f'{{"decision": "approve"}} or {{"decision": "release"}} -- {resolve_hint}.'
    )
    if not alert_sent:
        print(
            f"[{urgency}] Slack alert not sent (SLACK_WEBHOOK_URL not "
            f"configured) -- this transaction is only visible in this "
            f"log and the dashboard's Pending Reviews panel until "
            f"someone resolves it or Razorpay's capture window lapses."
        )


def resolve_transaction(transaction_id: str, decision: str) -> dict:
    """
    The other half of the human-in-the-loop loop: called after a person
    actually looks at a held review/step_up transaction (from a Slack
    alert, the dashboard's Pending Reviews panel, or curl against the
    /reviews/{id}/decision route below -- both call this same function,
    same pattern as src.scoring.score_transaction being shared between
    app/api.py and app/dashboard.py).

    decision="approve" -> capture it if still authorized; no-op if
                           already captured (the human is saying "this
                           was fine").
    decision="release" -> if authorized, do NOT capture (leave it to
                           lapse into Razorpay's automatic refund); if
                           already captured, issue a real refund.

    This never happens automatically -- see _act_on_decision's
    docstring. A human's explicit decision is required either way.
    """
    if decision not in ("approve", "release"):
        raise ValueError(f"decision must be 'approve' or 'release', got: {decision!r}")

    record = get_transaction_by_id(transaction_id)
    if record is None:
        return {"transaction_id": transaction_id, "status": "not_found"}

    if record.get("resolution"):
        return {
            "transaction_id": transaction_id,
            "status": "already_resolved",
            "resolution": record["resolution"],
        }

    if record.get("action") == "allow":
        # Nothing was ever held here -- an "allow" transaction was
        # already auto-captured (or was already-captured to begin
        # with) by _act_on_decision at scoring time. Calling
        # capture_payment again would hit a real "already captured"
        # error from Razorpay. This function is only meaningful for a
        # review/step_up transaction that's still waiting on a human.
        return {
            "transaction_id": transaction_id,
            "status": "no_action_needed",
            "detail": "This transaction was scored 'allow' and already settled automatically.",
        }

    capture_state = record.get("capture_state", "authorized")
    amount = record["amount"]

    if decision == "approve":
        if capture_state == "captured":
            resolution = "confirmed_captured"
        else:
            capture_payment(transaction_id, amount, "INR")
            resolution = "captured"
    else:
        if capture_state == "captured":
            refund_payment(transaction_id, amount, notes={"reason": "manual_review_release"})
            resolution = "refunded"
        else:
            # Never captured, and the human is saying "reverse this" --
            # there's nothing to call. Just don't capture it; Razorpay
            # auto-refunds an uncaptured authorization once its capture
            # window lapses.
            resolution = "released_uncaptured"

    mark_transaction_resolved(transaction_id, resolution)
    return {"transaction_id": transaction_id, "status": "resolved", "resolution": resolution}


class ReviewDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(approve|release)$")


@router.post("/reviews/{transaction_id}/decision")
def resolve_review(transaction_id: str, body: ReviewDecisionRequest):
    try:
        result = resolve_transaction(transaction_id, body.decision)
    except RazorpayAPIError as error:
        raise HTTPException(status_code=502, detail=f"Razorpay action failed: {error}")

    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"No transaction found: {transaction_id}")

    return result
