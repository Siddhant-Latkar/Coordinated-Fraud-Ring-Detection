import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import sqlite3

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from app.webhook import resolve_transaction
from src.database import get_pending_reviews
from src.decisioning import choose_action as get_action
from src.decisioning import GRAPH_WEIGHT, MODEL_WEIGHT, RULE_WEIGHT
from src.features import build_features
from src.rules import evaluate_rules
from src.scoring import InvalidTransactionError, ModelNotTrainedError, score_transaction


st.set_page_config(
    page_title="Risk Manager",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Coordinated Fraud-Ring Risk Manager")
st.caption(
    "Defense-only fraud-risk prototype | "
    "All decisions are advisory: allow, review, or step-up verification."
)

DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"
DATABASE_PATH = PROJECT_ROOT / "data" / "risk_manager.db"


# -------------------------------------------------------------------
# Real-time transaction input form
# -------------------------------------------------------------------

st.subheader("Submit a live transaction")

with st.form("live_transaction_form", clear_on_submit=False):
    form_left, form_right = st.columns(2)

    with form_left:
        live_user_id = st.text_input(
            "User ID",
            value="live_user_001"
        )

        live_device_id = st.text_input(
            "Device ID",
            value="device_demo_001"
        )

        live_ip_id = st.text_input(
            "IP address / ID",
            value="ip_demo_001"
        )

    with form_right:
        live_merchant_id = st.text_input(
            "Merchant ID",
            value="merchant_demo"
        )

        live_amount = st.number_input(
            "Transaction amount (₹)",
            min_value=1.0,
            value=500.0,
            step=50.0
        )

    submit_live_transaction = st.form_submit_button(
        "Score and save transaction"
    )


if submit_live_transaction:
    payload = {
        "user_id": live_user_id.strip(),
        "device_id": live_device_id.strip(),
        "ip_id": live_ip_id.strip(),
        "merchant_id": live_merchant_id.strip(),
        "amount": float(live_amount)
    }

    if not all([
        payload["user_id"],
        payload["device_id"],
        payload["ip_id"],
        payload["merchant_id"]
    ]):
        st.error("User ID, device ID, IP ID, and merchant ID are required.")

    else:
        try:
            # In-process call, not HTTP -- same src.scoring.score_transaction()
            # pipeline used by app/api.py and the Razorpay webhook.
            result = score_transaction(payload)

            action = result["action"]
            score = result["risk_score"]

            if result.get("duplicate"):
                st.info(
                    "This transaction_id was already scored earlier; "
                    "showing the original decision instead of re-scoring."
                )

            if action == "allow":
                st.success(
                    f"Decision: ALLOW | Final risk score: {score:.4f}"
                )

            elif action == "review":
                st.warning(
                    f"Decision: REVIEW | Final risk score: {score:.4f}"
                )

            else:
                st.error(
                    f"Decision: STEP-UP VERIFICATION | "
                    f"Final risk score: {score:.4f}"
                )

            result_col1, result_col2, result_col3 = st.columns(3)

            result_col1.metric(
                "ML model score",
                f"{result.get('model_score', 0):.4f}"
            )

            result_col2.metric(
                "Graph score",
                f"{result.get('graph_score', 0):.4f}"
            )

            result_col3.metric(
                "Final risk score",
                f"{score:.4f}"
            )

            st.write("Live evidence")

            for item in result.get("evidence", []):
                st.write(f"- {item}")

            with st.expander("Full response"):
                st.json(result)

        except InvalidTransactionError as error:
            st.error(f"Invalid transaction: {error}")

        except ModelNotTrainedError:
            st.error(
                "Model not found. Run this once in a terminal: "
                "python -m src.train"
            )

        except Exception as error:
            st.error(f"Could not score transaction: {error}")


st.divider()


# -------------------------------------------------------------------
# Helper function: load real-time scored records from SQLite
# -------------------------------------------------------------------

def load_live_transactions():
    if not DATABASE_PATH.exists():
        return pd.DataFrame()

    try:
        connection = sqlite3.connect(DATABASE_PATH)

        live_df = pd.read_sql_query(
            """
            SELECT *
            FROM transactions
            ORDER BY timestamp DESC
            """,
            connection
        )

        connection.close()

        if not live_df.empty:
            live_df["timestamp"] = pd.to_datetime(
                live_df["timestamp"],
                errors="coerce"
            )

        return live_df

    except Exception as error:
        st.warning(f"Could not load real-time database records: {error}")
        return pd.DataFrame()


# -------------------------------------------------------------------
# Live real-time monitoring section
# -------------------------------------------------------------------

st.subheader("Live scored transactions")

live_df = load_live_transactions()

if live_df.empty:
    st.info(
        "No live transactions are stored yet. Fill in the form above "
        "and click \"Score and save transaction\" to see a result here."
    )

else:
    live_total = len(live_df)
    live_review = int((live_df["action"] == "review").sum())
    live_step_up = int((live_df["action"] == "step_up").sum())
    live_avg_risk = live_df["final_risk_score"].mean()

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Live transactions", f"{live_total:,}")
    metric2.metric("Review queue", f"{live_review:,}")
    metric3.metric("Step-up queue", f"{live_step_up:,}")
    metric4.metric("Average live risk", f"{live_avg_risk:.4f}")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        live_risk_chart = px.histogram(
            live_df,
            x="final_risk_score",
            color="action",
            nbins=30,
            title="Live risk-score distribution",
            color_discrete_map={
                "allow": "#2ca02c",
                "review": "#ffae00",
                "step_up": "#d62728"
            }
        )

        live_risk_chart.update_layout(
            xaxis_title="Final risk score",
            yaxis_title="Transaction count"
        )

        st.plotly_chart(
            live_risk_chart,
            use_container_width=True
        )

    with chart_right:
        action_counts = (
            live_df["action"]
            .value_counts()
            .rename_axis("action")
            .reset_index(name="count")
        )

        live_action_chart = px.pie(
            action_counts,
            names="action",
            values="count",
            title="Live advisory decisions",
            color="action",
            color_discrete_map={
                "allow": "#2ca02c",
                "review": "#ffae00",
                "step_up": "#d62728"
            }
        )

        st.plotly_chart(
            live_action_chart,
            use_container_width=True
        )

    st.write("Most recent real-time decisions")

    live_display_columns = [
        "transaction_id",
        "timestamp",
        "user_id",
        "device_id",
        "ip_id",
        "merchant_id",
        "amount",
        "model_score",
        "graph_score",
        "rule_score",
        "final_risk_score",
        "action",
        "evidence"
    ]

    available_columns = [
        column
        for column in live_display_columns
        if column in live_df.columns
    ]

    st.dataframe(
        live_df[available_columns],
        use_container_width=True,
        height=350
    )


st.divider()


# -------------------------------------------------------------------
# Pending human reviews: held review/step_up transactions from a real
# Razorpay webhook, resolved here via app.webhook.resolve_transaction().
# -------------------------------------------------------------------

st.subheader("Pending human reviews")
st.caption(
    "Transactions from a real Razorpay webhook that were held "
    "(review/step_up) and are still waiting on a decision. Nothing "
    "here happens automatically -- approve captures the payment (or "
    "confirms an already-captured one), release lets it lapse "
    "uncaptured or issues a refund if it was already captured."
)

pending_reviews = get_pending_reviews()

if not pending_reviews:
    st.info(
        "No pending reviews. This fills up from real Razorpay webhook "
        "traffic (see app/webhook.py) -- the \"Submit a live "
        "transaction\" form above scores transactions directly and "
        "never creates a pending review, since it has no real Razorpay "
        "payment behind it to hold."
    )

else:
    for pending in pending_reviews:
        evidence = json.loads(pending["evidence"])
        badge = "🔴" if pending["action"] == "step_up" else "🟡"

        with st.expander(
            f"{badge} {pending['transaction_id']} | "
            f"{pending['action']} | risk {pending['final_risk_score']:.4f} | "
            f"₹{pending['amount']:,.2f}"
        ):
            st.write(f"User: {pending['user_id']} | Device: {pending['device_id']} | IP: {pending['ip_id']}")
            st.write(f"Capture state: {pending.get('capture_state', 'authorized')}")
            st.write("Evidence:")
            for item in evidence:
                st.write(f"- {item}")

            approve_col, release_col = st.columns(2)

            if approve_col.button("Approve (capture)", key=f"approve_{pending['transaction_id']}"):
                try:
                    result = resolve_transaction(pending["transaction_id"], "approve")
                    st.success(f"Resolved: {result['status']} ({result.get('resolution', '')})")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not resolve: {error}")

            if release_col.button("Release (refund/lapse)", key=f"release_{pending['transaction_id']}"):
                try:
                    result = resolve_transaction(pending["transaction_id"], "release")
                    st.success(f"Resolved: {result['status']} ({result.get('resolution', '')})")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not resolve: {error}")


st.divider()


# -------------------------------------------------------------------
# Historical synthetic dataset analysis
# -------------------------------------------------------------------

st.subheader("Historical synthetic-data analysis")

if not DATA_PATH.exists():
    st.error("Data file not found. Run: python -m src.generate_data")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python -m src.train")
    st.stop()


@st.cache_data
def load_historical_features(data_path: str, model_path: str):
    """
    Builds the full model_score + graph_score + rule_score + blended
    risk_score for every historical row, using the SAME
    src.rules.evaluate_rules() and src.decisioning.combine_scores()
    logic the live scoring path uses -- not a second hand-rolled
    approximation. Cached because this recomputes ~10k rule
    evaluations (~1.5s) and would otherwise re-run on every sidebar
    interaction.
    """
    raw_df = pd.read_csv(data_path)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["features"]

    built = build_features(raw_df)
    built["model_score"] = model.predict_proba(built[feature_names])[:, 1]

    def _row_rules(row):
        stats = {
            "device_unique_users_24h": row["device_unique_users_24h"],
            "ip_unique_users_24h": row["ip_unique_users_24h"],
            "device_unique_users_30d": row["device_unique_users_30d"],
            "ip_unique_users_30d": row["ip_unique_users_30d"],
            "device_txn_count_24h": row["device_txn_count_24h"],
            "ip_txn_count_24h": row["ip_txn_count_24h"],
        }
        rule_score, evidence = evaluate_rules({"amount": row["amount"]}, stats)
        return pd.Series({
            "rule_score": rule_score,
            "rule_evidence": json.dumps(evidence),
        })

    rule_results = built.apply(_row_rules, axis=1)
    built["rule_score"] = rule_results["rule_score"]
    built["rule_evidence"] = rule_results["rule_evidence"]

    # Same blend formula as src.decisioning.combine_scores(), so
    # historical and live scoring never disagree.
    built["risk_score"] = (
        MODEL_WEIGHT * built["model_score"]
        + GRAPH_WEIGHT * built["graph_risk_score"]
        + RULE_WEIGHT * built["rule_score"]
    ).clip(0, 1)

    built["action"] = built["risk_score"].apply(get_action)

    return built


features_df = load_historical_features(str(DATA_PATH), str(MODEL_PATH))


# Sidebar filters
st.sidebar.header("Historical data filters")
st.sidebar.caption(
    "Applies to the metrics, charts, explorer table, and transaction "
    "detail panel below. The \"Most suspicious shared entities\" panel "
    "always shows the full dataset, since it's a structural scan for "
    "shared devices/IPs rather than a decision-based view."
)

action_filter = st.sidebar.multiselect(
    "Historical decision",
    options=["allow", "review", "step_up"],
    default=["allow", "review", "step_up"]
)

label_filter = st.sidebar.multiselect(
    "Historical actual label",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda value: (
        "Normal" if value == 0 else "Synthetic fraud"
    )
)

filtered = features_df[
    features_df["action"].isin(action_filter)
    & features_df["label"].isin(label_filter)
]


# Historical metrics -- reflect the sidebar filter (see caption above)
total_transactions = len(filtered)
fraud_count = int(filtered["label"].sum())
review_count = int((filtered["action"] == "review").sum())
step_up_count = int((filtered["action"] == "step_up").sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Filtered transactions", f"{total_transactions:,}")
col2.metric("Filtered fraud labels", f"{fraud_count:,}")
col3.metric("Filtered reviews", f"{review_count:,}")
col4.metric("Filtered step-up", f"{step_up_count:,}")


# Historical charts -- also reflect the sidebar filter
left, right = st.columns(2)

if filtered.empty:
    st.info("No historical transactions match the selected filters.")

else:
    with left:
        st.subheader("Historical risk-score distribution")

        fig = px.histogram(
            filtered,
            x="risk_score",
            color="action",
            nbins=40,
            title="Blended risk scores (model + graph + rules)",
            color_discrete_map={
                "allow": "#2ca02c",
                "review": "#ffae00",
                "step_up": "#d62728"
            }
        )

        fig.update_layout(
            xaxis_title="Risk score",
            yaxis_title="Transaction count"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Historical decision distribution")

        decision_counts = (
            filtered["action"]
            .value_counts()
            .rename_axis("action")
            .reset_index(name="count")
        )

        fig = px.pie(
            decision_counts,
            names="action",
            values="count",
            title="Advisory decisions",
            color="action",
            color_discrete_map={
                "allow": "#2ca02c",
                "review": "#ffae00",
                "step_up": "#d62728"
            }
        )

        st.plotly_chart(fig, use_container_width=True)


# Suspicious historical entities -- intentionally always the FULL
# dataset (see sidebar caption): this is a structural scan for shared
# devices/IPs, not a view of the model's decisions, so filtering it by
# decision/label would hide the exact rings it's meant to surface.
st.subheader("Most suspicious shared historical entities")
st.caption("Always shows the full dataset, independent of the sidebar filters.")

entity_col1, entity_col2 = st.columns(2)

with entity_col1:
    suspicious_devices = (
        features_df.groupby("device_id")
        .agg(
            transactions=("transaction_id", "count"),
            users=("user_id", "nunique"),
            max_risk=("risk_score", "max"),
            fraud_labels=("label", "sum")
        )
        .query("users > 1")
        .sort_values(
            ["fraud_labels", "max_risk"],
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    st.write("Devices shared by multiple users")
    st.dataframe(suspicious_devices, use_container_width=True)

with entity_col2:
    suspicious_ips = (
        features_df.groupby("ip_id")
        .agg(
            transactions=("transaction_id", "count"),
            users=("user_id", "nunique"),
            max_risk=("risk_score", "max"),
            fraud_labels=("label", "sum")
        )
        .query("users > 1")
        .sort_values(
            ["fraud_labels", "max_risk"],
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    st.write("IP addresses shared by multiple users")
    st.dataframe(suspicious_ips, use_container_width=True)


# Historical transaction explorer
st.subheader("Historical transaction explorer")

display_columns = [
    "transaction_id",
    "timestamp",
    "user_id",
    "device_id",
    "ip_id",
    "merchant_id",
    "amount",
    "model_score",
    "graph_risk_score",
    "rule_score",
    "risk_score",
    "action",
    "label"
]

st.dataframe(
    filtered.sort_values(
        "risk_score",
        ascending=False
    )[display_columns].head(200),
    use_container_width=True,
    height=450
)


# Historical transaction investigation
st.subheader("Historical transaction details")

transaction_ids = filtered["transaction_id"].tolist()

if transaction_ids:
    selected_id = st.selectbox(
        "Select a historical transaction",
        transaction_ids[:500]
    )

    selected = features_df[
        features_df["transaction_id"] == selected_id
    ].iloc[0]

    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)

    detail_col1.metric(
        "Model score",
        f"{selected['model_score']:.4f}"
    )

    detail_col2.metric(
        "Graph score",
        f"{selected['graph_risk_score']:.4f}"
    )

    detail_col3.metric(
        "Blended risk score",
        f"{selected['risk_score']:.4f}"
    )

    detail_col4.metric(
        "Decision",
        selected["action"]
    )

    st.caption(
        "Actual label: "
        + ("Synthetic fraud" if selected["label"] == 1 else "Normal")
    )

    # Real evidence from src.rules.evaluate_rules() -- not a second,
    # hand-rolled approximation. An earlier version of this panel used
    # its own ad-hoc thresholds and never looked at the 30-day
    # device/IP signal at all, so it showed only "high amount" for
    # hundreds of rows that are actually caught fraud-ring members.
    evidence = json.loads(selected["rule_evidence"])
    evidence = [item for item in evidence if item != "No high-confidence rule signal detected"]
    evidence.append(f"ML model score: {selected['model_score']:.4f}")
    evidence.append(f"Graph relationship score: {selected['graph_risk_score']:.4f}")
    if not evidence:
        evidence.append("No high-confidence rule signal detected")

    st.write("Risk evidence")

    for item in evidence:
        st.write(f"- {item}")

    timeline = features_df[
        (features_df["user_id"] == selected["user_id"])
        | (features_df["device_id"] == selected["device_id"])
        | (features_df["ip_id"] == selected["ip_id"])
    ].sort_values("timestamp")

    st.write("Related historical activity timeline")

    st.dataframe(
        timeline[display_columns].tail(100),
        use_container_width=True
    )

else:
    st.info("No historical transactions match the selected filters.")
