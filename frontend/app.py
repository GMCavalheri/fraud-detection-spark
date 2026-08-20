import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")


@st.cache_data(ttl=30)
def api_get(path, params=None):
    r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def render_overview():
    st.title("Overview")
    try:
        summary = api_get("/stats/summary")
    except requests.HTTPError:
        st.warning("No scored data yet. Run the Spark pipeline first: `docker compose run --rm spark-pipeline`.")
        return

    total = summary["total_transactions"]
    fraud_rate = summary["predicted_fraud_count"] / total if total else 0
    auc = (summary.get("model_metrics") or {}).get("auc_roc")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Predicted Fraud Rate", f"{fraud_rate:.2%}")
    c3.metric("$ Flagged as Fraud", f"${summary['flagged_amount_usd']:,.0f}")
    c4.metric("Model AUC-ROC", f"{auc:.3f}" if auc else "n/a")

    ts = pd.DataFrame(api_get("/stats/timeseries"))
    if not ts.empty:
        ts["fraud_rate"] = ts["predicted_fraud_count"] / ts["total_transactions"]
        fig = px.line(ts, x="event_date", y="fraud_rate", markers=True, title="Predicted Fraud Rate Over Time")
        fig.update_yaxes(tickformat=".1%", title="Fraud rate")
        fig.update_xaxes(title="Date")
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        cats = pd.DataFrame(api_get("/stats/categories"))
        if not cats.empty:
            top = cats.sort_values("predicted_fraud_count", ascending=False).head(10)
            fig2 = px.bar(top, x="category", y="predicted_fraud_count", title="Flagged Transactions by Category")
            fig2.update_xaxes(title="")
            fig2.update_yaxes(title="Flagged count")
            st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        sample = pd.DataFrame(api_get("/transactions", {"limit": 2000})["items"])
        if not sample.empty:
            fig3 = px.histogram(sample, x="amount_usd", nbins=50, title="Amount Distribution (recent 2,000 txns)")
            fig3.update_xaxes(title="Amount (USD)")
            st.plotly_chart(fig3, use_container_width=True)


def render_data_quality():
    st.title("Data Quality Report")
    st.caption("What the Spark ETL step found and fixed across the three messy source systems.")
    try:
        report = api_get("/reports/data-quality")
    except requests.HTTPError:
        st.warning("Report not available yet - run the Spark pipeline first.")
        return

    raw = report["raw_row_counts"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows ingested", f"{report['total_raw_rows']:,}")
    c2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
    c3.metric("Final cleaned rows", f"{report['final_cleaned_rows']:,}")

    st.subheader("Issues found and fixed")
    issues = {
        "Duplicate rows removed": report["duplicates_removed"],
        "Negative amounts corrected": report["negative_amounts_corrected"],
        "Missing merchant imputed": report["missing_merchant_imputed"],
        "Missing city imputed": report["missing_city_imputed"],
        "Invalid/future timestamps dropped": report["invalid_or_future_timestamps_dropped"],
        "Corrupt JSON lines dropped": raw["mobile_events_corrupt_lines_dropped"],
    }
    fig = px.bar(x=list(issues.keys()), y=list(issues.values()), title="Rows Affected per Issue Type")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Rows")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw source breakdown"):
        st.json(raw)


def render_model_performance():
    st.title("Model Performance")
    try:
        summary = api_get("/stats/summary")
    except requests.HTTPError:
        st.warning("No metrics yet - run the Spark pipeline first.")
        return

    metrics = summary.get("model_metrics")
    if not metrics:
        st.warning("Model metrics not found.")
        return

    st.caption(
        f"{metrics['model_type']} trained on {metrics['train_rows']:,} rows "
        f"({metrics['train_date_range'][0]} to {metrics['train_date_range'][1]}), tested on "
        f"{metrics['test_rows']:,} rows ({metrics['test_date_range'][0]} to {metrics['test_date_range'][1]}). "
        "Time-based split - the model is only ever evaluated on days after the ones it trained on."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    c2.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
    c3.metric("Precision", f"{metrics['precision']:.3f}")
    c4.metric("Recall", f"{metrics['recall']:.3f}")
    c5.metric("F1", f"{metrics['f1']:.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        cm = metrics["confusion_matrix"]
        z = [
            [cm["true_negative"], cm["false_positive"]],
            [cm["false_negative"], cm["true_positive"]],
        ]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["Predicted Legit", "Predicted Fraud"],
                y=["Actual Legit", "Actual Fraud"],
                text=z,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig_cm.update_layout(title="Confusion Matrix (test set)")
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_b:
        fi = pd.DataFrame(metrics["feature_importances"], columns=["feature", "importance"]).sort_values("importance")
        fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="Feature Importances")
        st.plotly_chart(fig_fi, use_container_width=True)


def render_transaction_explorer():
    st.title("Transaction Explorer")
    meta = api_get("/meta")

    with st.form("filters"):
        c1, c2, c3, c4 = st.columns(4)
        category = c1.selectbox("Category", ["All"] + meta["categories"])
        label = c2.selectbox("Predicted", ["All", "Fraud", "Legit"])
        min_amt = c3.number_input("Min amount", min_value=0.0, value=0.0)
        max_amt = c4.number_input("Max amount", min_value=0.0, value=0.0, help="0 = no max")
        st.form_submit_button("Search")

    params = {"limit": 200}
    if category != "All":
        params["category"] = category
    if label == "Fraud":
        params["predicted_label"] = 1
    elif label == "Legit":
        params["predicted_label"] = 0
    if min_amt > 0:
        params["min_amount"] = min_amt
    if max_amt > 0:
        params["max_amount"] = max_amt

    data = api_get("/transactions", params)
    st.caption(f"{data['total']:,} matching transactions (showing first {len(data['items'])})")
    df = pd.DataFrame(data["items"])
    if not df.empty:
        df["fraud?"] = df["predicted_label"].map({1: "🚩 Fraud", 0: "Legit"})
        st.dataframe(
            df[[
                "transaction_id", "event_ts", "account_id", "amount_usd", "category",
                "merchant", "city", "channel", "fraud?", "fraud_probability", "rule_flags",
            ]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No transactions match these filters.")


def render_live_demo():
    st.title("Live Fraud Scoring Demo")
    st.caption("Scores a hypothetical transaction against the exact Spark MLlib model trained by the pipeline.")
    meta = api_get("/meta")

    with st.form("score_form"):
        account_id = st.text_input(
            "Account ID (optional - leave blank to simulate a brand-new card)",
            help="e.g. ACCT-116213. If it exists, the API pulls that account's real recent activity from Postgres.",
        )
        c1, c2 = st.columns(2)
        amount = c1.number_input("Amount (USD)", min_value=0.01, value=50.0, step=10.0)
        category = c2.selectbox("Category", meta["categories"])
        c3, c4 = st.columns(2)
        channel = c3.selectbox("Channel", meta["channels"])
        city = c4.selectbox("City (optional, enables geo-velocity check)", ["(unknown)"] + meta["cities"])
        submitted = st.form_submit_button("Score Transaction", type="primary")

    if not submitted:
        return

    payload = {"amount_usd": amount, "category": category, "channel": channel}
    if account_id.strip():
        payload["account_id"] = account_id.strip()
    if city != "(unknown)":
        payload["city"] = city

    with st.spinner("Scoring with the Spark MLlib model..."):
        resp = requests.post(f"{API_BASE_URL}/score", json=payload, timeout=60)

    if resp.status_code != 200:
        st.error(f"Scoring failed: {resp.text}")
        return

    result = resp.json()
    prob = result["fraud_probability"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#b3261e" if prob >= 0.5 else "#1e7a3d"},
                "steps": [
                    {"range": [0, 50], "color": "#d9f2df"},
                    {"range": [50, 100], "color": "#fbdada"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))

    col_g, col_r = st.columns([1, 1])
    with col_g:
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader("🚩 FLAGGED AS FRAUD" if result["predicted_label"] == 1 else "✅ Looks legit")
        st.write(
            f"**Used real account history:** "
            f"{'Yes' if result['used_account_history'] else 'No (scored as a brand-new card)'}"
        )
        st.write("**Rule-based flags:** " + (", ".join(result["rule_flags"]) or "none"))
        with st.expander("Features fed to the model"):
            st.json(result["features_used"])


def main():
    st.sidebar.title("🛡️ Fraud Detection")
    st.sidebar.caption("Synthetic messy data → Spark ETL/MLlib → FastAPI → Streamlit")
    page = st.sidebar.radio(
        "Section",
        ["Overview", "Data Quality Report", "Model Performance", "Transaction Explorer", "Live Fraud Scoring Demo"],
    )
    st.sidebar.divider()
    st.sidebar.markdown("[View source on GitHub](https://github.com/GMCavalheri/fraud-detection-spark)")

    try:
        api_get("/health")
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API_BASE_URL}: {e}")
        st.stop()

    pages = {
        "Overview": render_overview,
        "Data Quality Report": render_data_quality,
        "Model Performance": render_model_performance,
        "Transaction Explorer": render_transaction_explorer,
        "Live Fraud Scoring Demo": render_live_demo,
    }
    pages[page]()


if __name__ == "__main__":
    main()
