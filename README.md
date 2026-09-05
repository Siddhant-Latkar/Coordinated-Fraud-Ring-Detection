# Coordinated Fraud-Ring Risk Manager

> Advisory-only fraud-risk scoring for merchants. Combines a supervised
> ML model, identity-sharing ("graph-lite") signals, and transparent
> rules to classify each transaction as `allow`, `review`, or
> `step_up`, with a real Razorpay webhook integration behind it.

## The problem

Most fraud checks look at one transaction at a time: amount, location,
device. Coordinated rings don't work that way. One attacker reuses a
single device or IP across many different, often stolen, identities --
and deliberately spaces the transactions out, sometimes days apart, to
stay under any single velocity check. Looked at individually, every
one of those transactions looks completely normal. This is fraud
that's only visible across time, not in any one transaction.

## The solution

```
Checkout -> Razorpay order -> Payment event
                  |
     Verify signature + attach device/IP
                  |
   Score: ML model + graph signal + rules -> one blended risk score
                  |
        -----------------------------
        |             |             |
      allow         review       step_up
        |             |             |
     capture         hold          hold
                       \___________/
                             |
                Human approves or releases
```

We track how many *different* users share one device or IP within a
24-hour window and a 30-day window, and blend that with a trained ML
model and a transparent rule layer into one risk score: 60% model, 25%
graph, 15% rules. The formula is plain arithmetic, not a black box --
anyone can hand-check it against `src/decisioning.py`. The dashboard,
the API, and the Razorpay webhook all call the exact same
`score_transaction()` function -- there is no second copy of the
scoring logic anywhere to drift out of sync.

**This system is strictly advisory and defense-only.** It never blocks
a payment on its own: `allow` auto-captures, but `review` and
`step_up` only hold a payment and put it in front of a human, who
approves or releases it. There is no offense-capable code anywhere in
this project -- no exploit generation, no attack tooling, nothing that
acts on another system. It scores and recommends; a human makes the
final call.

## Run it yourself

No Razorpay account is needed for the local demo.

```bash
pip install -r requirements.txt
python -m src.generate_data      # only if data/transactions.csv is missing
python -m src.train               # trains models/risk_model.joblib, prints the scorecard below
uvicorn app.api:app --reload      # API on http://127.0.0.1:8000
streamlit run app/dashboard.py    # dashboard on http://localhost:8501
```

## Quick tests to verify it actually works

1. **A normal transaction.** POST to `/transactions` with any values.
   Expect a risk score and an `allow` decision.
2. **A coordinated ring.** POST several transactions with different
   `user_id` values but the same `device_id`/`ip_id`, one after
   another. Expect the risk score to climb and escalate to `review` or
   `step_up` by the third or fourth transaction.
3. **Human review.** After step 2 triggers a hold, open the
   dashboard's "Pending human reviews" panel -- the transaction should
   appear there, with Approve/Release buttons.
4. **Invalid input.** POST an `amount` of `NaN`. Expect a clean `422`
   rejection, never a `500`.
5. **The automated suite:** `pytest tests/ -v` runs 13 tests covering
   all four of the above end-to-end, plus Razorpay webhook signature
   verification, event deduplication, and checkout-context capture,
   with Razorpay's HTTP calls mocked.

## Architecture

```
src/generate_data.py       synthetic transactions + 25 embedded fraud rings
src/features.py            offline, leak-free feature engineering (training)
src/database.py            SQLite store + live 24h/30d windowed entity stats
src/realtime_features.py   builds the same feature vector at serve time
src/rules.py               transparent, human-readable rule layer
src/decisioning.py         blend weights + allow/review/step_up thresholds
src/scoring.py             the one scoring pipeline every entry point calls
src/train.py               trains the model, prints the honest scorecard
src/checkout_sessions.py   stores device/IP captured at checkout, keyed by order_id
app/api.py                 FastAPI: /transactions, /checkout/create-order
app/webhook.py             /webhooks/razorpay ingestion + review resolution
app/razorpay_client.py     create_order / capture_payment / refund_payment / Slack alerts
app/dashboard.py           Streamlit: live demo + pending reviews + historical investigation
```

## Honest metrics

Held-out, chronological, never-shuffled test set. These numbers are
for the **actual deployed decision** -- model + graph + rule blend --
not just the raw model in isolation. "Flagged" means `review` or
`step_up`; an `allow` transaction is a negative prediction.

| Metric | Value |
|---|---|
| PR-AUC (raw model) | 0.987 |
| Precision (deployed pipeline) | 0.931 |
| Recall (deployed pipeline) | 0.982 |
| False positives | 8 legitimate transactions flagged per ~2,000 |
| Est. review cost | ~Rs.400 (at Rs.50/manual review, a placeholder) |
| Est. fraud caught | ~Rs.172,178 |

For comparison, the raw ML model alone (no graph/rule blend, a plain
0.5 threshold) scores precision 0.939 / recall 0.982 on the same test
set. Both sets of numbers print every time you run `python -m
src.train`, so this table can't silently drift from what the code
computes. These are synthetic-data results, not production
performance.

**Why precision is 0.93, not closer to 0.97:** ring amounts are mixed
across three bands (Rs.20-150, Rs.450-1200, Rs.1500-5000) instead of
one narrow range, so the model can't use amount as a shortcut for
identity-sharing. That's a harder, more realistic problem, and the
lower number reflects that rather than a regression.

**Why two time windows, 24h and 30d:** this dataset's ring pattern is
device/IP reuse spread over weeks, not a same-day burst -- the median
gap between transactions on the same ring device is ~4.5 days. A
24h-only check catches ~29% of rings; the 30-day window recovers the
rest.

## What broke, and how we fixed it

Both of these were caught by deliberately trying to break the system,
not by accident:

- **A crash on malformed input.** Sending a non-finite `amount`
  (`NaN`) to `/transactions` returned a 500 instead of a clean
  rejection. Fixed in `app/api.py`, with a regression test in
  `tests/test_api.py`.
- **A model blind spot for low-value fraud.** An earlier version of
  the training data used one narrow amount range for every fraud ring,
  so the model learned that range as a shortcut instead of learning
  the identity-sharing pattern itself. Fixed by mixing three realistic
  amount bands into the data generator and retraining; the metrics
  above are from the corrected model.

## Known limitations

- Trained entirely on synthetic data -- these numbers establish that
  the approach works on a controlled, labeled dataset, not that it
  transfers to real traffic without retraining on real chargebacks.
- The blend weights, action thresholds, and review-cost estimate
  (`src/decisioning.py`, `src/train.py`) are placeholders, not tuned
  against real operating costs.
- A system that can only see 24h/30d of history cannot catch a ring's
  very first transaction -- there's no prior pattern yet for any
  detector to use. This is a structural limit of behavior-based
  detection, not something tunable away.
- The Razorpay integration is written against the documented API
  contract and unit-tested against a mocked HTTP client, not yet
  smoke-tested against a live account.
