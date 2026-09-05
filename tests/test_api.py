import os
import tempfile

import pytest

# Point the API at a throwaway database BEFORE importing app.api, so
# tests never touch the real demo data used for the live dashboard.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["RISK_DB_PATH"] = _tmp_db.name

from fastapi.testclient import TestClient  # noqa: E402
from app.api import app  # noqa: E402
from src.database import initialize_database  # noqa: E402

# TestClient(app) without a `with` block doesn't run FastAPI's startup
# handler in every Starlette version, so make sure the table exists
# explicitly rather than depending on that side effect.
initialize_database()

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200


def test_single_transaction_scores_without_crashing():
    """
    This is the exact call the live dashboard makes. It used to return
    a 500 on every single request because graph_risk_score was dropped
    before the API tried to read it back out.
    """
    response = client.post("/transactions", json={
        "user_id": "u_test_001",
        "device_id": "d_test_001",
        "ip_id": "ip_test_001",
        "merchant_id": "m_test_001",
        "amount": 500,
    })
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["action"] in {"allow", "review", "step_up"}


def test_non_finite_amount_returns_clean_422_not_500():
    """
    NaN/Infinity amounts must be rejected with 422, not crash with 500.
    See app/api.py's handle_validation_error for the fix.
    """
    for raw_amount in (b"NaN", b"Infinity", b"-Infinity"):
        body = (
            b'{"user_id":"u_x","device_id":"d_x","ip_id":"ip_x",'
            b'"merchant_id":"m_x","amount":' + raw_amount + b"}"
        )
        response = client.post(
            "/transactions", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code in (400, 422), (
            f"amount={raw_amount!r} returned {response.status_code}, expected a clean "
            f"400/422 rejection, not a crash"
        )


def test_ring_burst_raises_risk_score():
    """
    Simulates a coordinated ring: several different users transacting
    from the SAME device and SAME IP in quick succession. Risk for the
    later transactions in the burst must clearly exceed an isolated,
    unrelated transaction -- this is the actual capability the track
    asks for, proven end-to-end through the real API and database.
    """
    shared_device = "d_ring_shared"
    shared_ip = "ip_ring_shared"

    last_response = None
    for i in range(6):
        last_response = client.post("/transactions", json={
            "user_id": f"u_ring_{i}",
            "device_id": shared_device,
            "ip_id": shared_ip,
            "merchant_id": "m_ring_target",
            "amount": 900,
        })
        assert last_response.status_code == 200

    ring_score = last_response.json()["risk_score"]

    baseline_score = client.post("/transactions", json={
        "user_id": "u_normal_1",
        "device_id": "d_normal_1",
        "ip_id": "ip_normal_1",
        "merchant_id": "m_normal_1",
        "amount": 900,
    }).json()["risk_score"]

    assert ring_score > baseline_score
    assert ring_score >= 0.5  # should trigger at least "review"
