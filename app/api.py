import asyncio
import math
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.razorpay_client import RazorpayAPIError, create_order, get_key_id
from app.webhook import router as razorpay_webhook_router, run_scoring_worker
from src.checkout_sessions import initialize_checkout_sessions_table, save_checkout_context
from src.database import initialize_database, to_iso
from src.scoring import (
    MODEL_PATH,
    InvalidTransactionError,
    ModelNotTrainedError,
    score_transaction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    initialize_checkout_sessions_table()
    # Real Razorpay webhook traffic is scored asynchronously by this
    # background task (see app/webhook.py) instead of inline in the
    # request handler, so a slow scoring pass never risks a webhook
    # timing out and Razorpay retrying it as a duplicate delivery.
    worker_task = asyncio.create_task(run_scoring_worker())
    yield
    worker_task.cancel()


app = FastAPI(
    title="Fraud-Ring Risk API",
    version="1.0.0",
    description="Defense-only advisory fraud-risk scoring API",
    lifespan=lifespan,
)

app.include_router(razorpay_webhook_router)


def _sanitize_non_finite_floats(value):
    """
    Recursively replaces NaN/Infinity with their string form.

    Python's json.loads accepts NaN/Infinity as a non-standard
    extension, so a request with amount:NaN parses and pydantic
    correctly rejects it -- but the stdlib json encoder can't
    serialize NaN/Infinity, so echoing the rejected value back in the
    422 error body raises an unhandled exception. This avoids that
    without changing which requests are accepted or rejected.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # "nan", "inf", "-inf" -- all valid JSON strings
    if isinstance(value, dict):
        return {key: _sanitize_non_finite_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_finite_floats(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_non_finite_floats(exc.errors())},
    )


class TransactionRequest(BaseModel):
    transaction_id: str | None = None
    timestamp: datetime | None = None

    user_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    ip_id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)

    amount: float = Field(gt=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "service": "real-time-risk-manager"
    }


@app.post("/transactions")
def score_transaction_endpoint(transaction: TransactionRequest):
    """
    Scores one transaction, persists it, and returns an advisory
    action. This is a thin HTTP wrapper around src.scoring, which is
    the same pipeline app/webhook.py uses for real Razorpay webhook
    traffic -- one scoring path, two entry points.
    """
    try:
        return score_transaction(transaction.model_dump())
    except InvalidTransactionError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except ModelNotTrainedError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not score transaction: {error}"
        )


class CreateOrderRequest(BaseModel):
    user_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    amount: float = Field(gt=0)


@app.post("/checkout/create-order")
def create_order_endpoint(order_request: CreateOrderRequest, request: Request):
    """
    Call this from YOUR OWN checkout page BEFORE opening Razorpay
    Checkout -- not the /transactions endpoint above, which is for
    scoring a transaction you already have full details for.

    This is the other half of the loop the webhook needs: Razorpay's
    webhook payload never contains device_id or IP, so they have to be
    captured here, at order-creation time, and joined back in later
    using the order_id (see src/checkout_sessions.py and
    app/webhook.py's _lookup_checkout_context).

    device_id must come from a client-side fingerprint your frontend
    JS collects (e.g. FingerprintJS) and sends in the request body.
    ip_id is captured server-side here, from the request itself --
    never trust a client-supplied IP.
    """
    ip_id = request.client.host if request.client else "unknown_ip"
    receipt = f"rcpt_{uuid.uuid4().hex[:16]}"

    try:
        order = create_order(
            amount_rupees=order_request.amount,
            receipt=receipt,
            notes={"user_id": order_request.user_id, "merchant_id": "default_merchant"},
        )
    except RazorpayAPIError as error:
        raise HTTPException(status_code=502, detail=f"Razorpay order creation failed: {error}")

    save_checkout_context(
        order_id=order["id"],
        user_id=order_request.user_id,
        device_id=order_request.device_id,
        ip_id=ip_id,
        created_at=to_iso(datetime.now(timezone.utc)),
    )

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "razorpay_key_id": get_key_id(),  # frontend needs this for Checkout.js
    }
