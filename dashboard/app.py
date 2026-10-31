"""Streamlit dashboard for Cashflow & Risk Assistant.

4-tab modern financial intelligence interface:
  1. Upload & Overview (Unified ingestion, category breakdown charts, ledger)
  2. Cash Flow Forecast (Explainable 30/60/90-day time-series projections & alerts)
  3. Anomalies (Z-Score & IsolationForest outliers with human-readable rationale)
  4. Ask a Question (Compute-first affordability recommendations)
"""

import sys
from html import escape
from pathlib import Path

# Add project root to sys.path so src imports work smoothly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import (
    SAMPLE_DATA_DIR,
    CATEGORIES,
    DEFAULT_HORIZONS,
    DEFAULT_SAFETY_BUFFER,
    DEFAULT_ZSCORE_THRESHOLD,
)
from src.ingestion.loader import load_files, load_file
from src.categorization.categorizer import categorize_transactions
from src.forecasting.forecaster import forecast_cashflow
from src.anomaly.anomaly_detector import detect_anomalies
from src.insight.insight_agent import answer_affordability_question, parse_affordability_question

# -------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Cashflow & Risk Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-color);
    background: var(--background-color);
}

/* Header & Banner styling */
.main-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    color: #f8fafc;
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(30, 27, 75, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.main-header h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff !important;
}
.main-header p {
    margin: 0.4rem 0 0 0;
    color: #dfe7ff;
    font-size: 1.05rem;
}

/* Metric Cards */
.metric-card {
    background: var(--secondary-background-color);
    color: var(--text-color);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.12);
}
.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-color);
    opacity: 0.75;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-color);
    line-height: 1.2;
}
.metric-subtext {
    font-size: 0.8rem;
    color: var(--text-color);
    opacity: 0.7;
    margin-top: 0.3rem;
}

/* Verdict Banners */
.verdict-banner-safe {
    background: linear-gradient(135deg, rgba(6, 95, 70, 0.96) 0%, rgba(4, 120, 87, 0.95) 100%);
    color: #ecfdf5;
    padding: 1.4rem 1.8rem;
    border-radius: 12px;
    border-left: 8px solid #10b981;
    margin: 1rem 0;
}
.verdict-banner-safe h3,
.verdict-banner-safe p,
.verdict-banner-risky h3,
.verdict-banner-risky p,
.verdict-banner-not-recommended h3,
.verdict-banner-not-recommended p {
    color: inherit;
}
.verdict-banner-risky {
    background: linear-gradient(135deg, rgba(146, 64, 14, 0.95) 0%, rgba(180, 83, 9, 0.95) 100%);
    color: #fffbeb;
    padding: 1.4rem 1.8rem;
    border-radius: 12px;
    border-left: 8px solid #f59e0b;
    margin: 1rem 0;
}
.verdict-banner-not-recommended {
    background: linear-gradient(135deg, rgba(136, 19, 55, 0.96) 0%, rgba(190, 18, 60, 0.95) 100%);
    color: #fff1f2;
    padding: 1.4rem 1.8rem;
    border-radius: 12px;
    border-left: 8px solid #f43f5e;
    margin: 1rem 0;
}

/* Chips & Tags */
.badge-inflow {
    background-color: rgba(16, 185, 129, 0.18);
    color: #d1fae5;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-outflow {
    background-color: rgba(239, 68, 68, 0.18);
    color: #fecaca;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Tab container tweak */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sanitize_markdown_text(raw_text: str) -> str:
    """Escape HTML and prevent LaTeX-like injections while preserving markdown formatting."""
    return escape(raw_text, quote=False).replace("$", r"\$")


# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "df_transactions" not in st.session_state:
    st.session_state.df_transactions = pd.DataFrame()

if "categorized" not in st.session_state:
    st.session_state.categorized = False

if "data_source_label" not in st.session_state:
    st.session_state.data_source_label = "No data loaded"


# -------------------------------------------------------------
# Sidebar: Controls & Data Source
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Advisor Controls")
    st.caption("Configure working capital assumptions & data feeds")

    # Safety Buffer Setting
    safety_buffer = st.number_input(
        "Minimum Safety Buffer (₹)",
        min_value=0.0,
        max_value=1000000.0,
        value=float(DEFAULT_SAFETY_BUFFER),
        step=5000.0,
        help="The minimum reserve cash cushion your business should maintain at all times.",
    )

    # Current Bank Balance Override
    use_manual_balance = st.checkbox("Set Manual Starting Balance", value=False)
    manual_balance = 0.0
    if use_manual_balance:
        manual_balance = st.number_input(
            "Current Cash Balance (₹)",
            value=125000.0,
            step=10000.0,
        )

    # Anomaly Threshold
    z_threshold = st.slider(
        "Anomaly Sensitivity (Z-Score)",
        min_value=1.5,
        max_value=4.0,
        value=2.0,
        step=0.25,
        help="Lower values flag more moderate deviations; higher values flag only extreme spikes.",
    )

    st.markdown("---")
    st.markdown("### 📂 Sample Demonstration")
    st.caption("Pre-configured 6-month retail dataset with realistic recurring rent, payroll, and injected anomalies.")

    if st.button("🚀 Try with sample data", use_container_width=True, type="primary"):
        with st.spinner("Loading synthetic invoices, expenses, sales, and bank statements..."):
            bank_statement = SAMPLE_DATA_DIR / "bank_statement.csv"
            if not bank_statement.exists():
                st.error("Sample data not found in data/sample/. Generating now...")
                from src.synthetic_data.generate_sample_data import generate_all_sample_data
                generate_all_sample_data()

            # The generated bank statement consolidates the detail files; loading
            # all four together would count the same activity multiple times.
            df_raw = load_file(bank_statement)
            df_cat = categorize_transactions(df_raw)
            st.session_state.df_transactions = df_cat
            st.session_state.categorized = True
            st.session_state.data_source_label = f"Sample Retail Bank Ledger ({len(df_cat)} records)"
            st.success(f"Loaded {len(df_cat)} records from the consolidated sample ledger.")
            st.rerun()

    st.markdown("---")
    st.caption("Powered by OpenRouter Free Tier Fallback & Explainable Time-Series AI")


# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>💼 Cashflow & Risk Assistant</h1>
        <p>AI-Powered Financial Intelligence: Explainable Cash Flow Forecasting, Spend Anomaly Audits, and Grounded Affordability Decisions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Tabs Interface
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Upload & Overview",
    "📈 2. Cash Flow Forecast",
    "🚨 3. Spending Anomalies",
    "💡 4. Ask Affordability Advisor",
])


# =============================================================
# TAB 1: UPLOAD & OVERVIEW
# =============================================================
with tab1:
    st.subheader("Data Ingestion & Unified Ledger")

    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_files = st.file_uploader(
            "Upload Financial Files (Invoices, Expenses, POS Sales, Bank Statements as CSV / Excel)",
            accept_multiple_files=True,
            type=["csv", "xlsx", "xls"],
            help="You can upload multiple files at once. The engine will normalize different headers into a unified schema.",
        )

        if uploaded_files:
            if st.button("📥 Ingest & Categorize Uploaded Files", type="primary"):
                with st.spinner("Ingesting files, reconciling dates, and categorizing via LLM..."):
                    df_ingested = load_files(uploaded_files)
                    df_cat = categorize_transactions(df_ingested)
                    st.session_state.df_transactions = df_cat
                    st.session_state.categorized = True
                    st.session_state.data_source_label = f"User Uploads ({len(uploaded_files)} files, {len(df_cat)} rows)"
                    st.success(f"Successfully processed {len(df_cat)} transactions!")
                    st.rerun()

    with col_info:
        st.info(
            f"**Current Active Source:** {st.session_state.data_source_label}\n\n"
            f"**Supported Formats:**\n"
            f"- Vendor Expense Spreadsheets\n"
            f"- Daily POS / Sales Registers\n"
            f"- B2B Invoices (Paid / Pending)\n"
            f"- Bank Checking Statements (Debit/Credit)"
        )

    df = st.session_state.df_transactions

    if not df.empty:
        st.markdown("---")
        # High level KPIs
        inflow_total = float(df[df["direction"] == "inflow"]["amount"].sum())
        outflow_total = float(df[df["direction"] == "outflow"]["amount"].sum())
        net_cash = inflow_total - outflow_total
        tx_count = len(df)
        date_range_str = f"{df['date'].min().strftime('%b %d, %Y')} — {df['date'].max().strftime('%b %d, %Y')}"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Inflow (Revenue)", f"₹{inflow_total:,.2f}")
        with kpi2:
            st.metric("Total Outflow (Expenses)", f"₹{outflow_total:,.2f}")
        with kpi3:
            st.metric("Net Operating Cash Flow", f"₹{net_cash:+,.2f}", delta=f"₹{net_cash:,.2f}")
        with kpi4:
            st.metric("Transactions Ingested", f"{tx_count:,}", help=f"Recorded period: {date_range_str}")

        # Charts Section
        st.markdown("#### 📂 Spending & Revenue Category Breakdown")
        chart_c1, chart_c2 = st.columns(2)

        # Expense by Category
        df_outflows = df[df["direction"] == "outflow"]
        if not df_outflows.empty:
            cat_out = df_outflows.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            with chart_c1:
                fig_out = px.pie(
                    cat_out,
                    names="category",
                    values="amount",
                    title="Outflow Breakdown by Category",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_out.update_traces(textposition="inside", textinfo="percent+label")
                fig_out.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
                st.plotly_chart(fig_out, use_container_width=True)

        # Daily Inflow vs Outflow timeline
        daily_summary = df.groupby(["date", "direction"])["amount"].sum().unstack(fill_value=0).reset_index()
        with chart_c2:
            fig_bar = go.Figure()
            if "inflow" in daily_summary.columns:
                fig_bar.add_trace(go.Bar(x=daily_summary["date"], y=daily_summary["inflow"], name="Inflow", marker_color="#10b981"))
            if "outflow" in daily_summary.columns:
                fig_bar.add_trace(go.Bar(x=daily_summary["date"], y=daily_summary["outflow"], name="Outflow", marker_color="#ef4444"))
            fig_bar.update_layout(
                title="Daily Cash Velocity (Inflow vs Outflow)",
                barmode="group",
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                margin=dict(t=40, b=10, l=10, r=10),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Transaction Table with Filter
        st.markdown("#### 📋 Unified Transaction Ledger")
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            dir_filter = st.selectbox("Filter Direction", ["All", "Inflow", "Outflow"])
        with f_col2:
            cat_options = ["All"] + sorted([str(c) for c in df["category"].dropna().unique()])
            cat_filter = st.selectbox("Filter Category", cat_options)
        with f_col3:
            search_query = st.text_input("Search Description / Vendor", "")

        df_filtered = df.copy()
        if dir_filter != "All":
            df_filtered = df_filtered[df_filtered["direction"] == dir_filter.lower()]
        if cat_filter != "All":
            df_filtered = df_filtered[df_filtered["category"] == cat_filter]
        if search_query.strip():
            df_filtered = df_filtered[df_filtered["description"].str.contains(search_query, case=False, na=False)]

        # Display formatted table
        display_df = df_filtered.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df["amount"] = display_df["amount"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(display_df, use_container_width=True, height=350)
    else:
        st.warning("👈 Please upload files or click 'Try with sample data' in the sidebar to get started.")


# =============================================================
# TAB 2: CASH FLOW FORECAST
# =============================================================
with tab2:
    st.subheader("Explainable Cash Flow Forecast (30 / 60 / 90 Days)")

    if df.empty:
        st.warning("Please load or upload transaction data first in Tab 1.")
    else:
        f_c1, f_c2 = st.columns([1, 3])
        with f_c1:
            horizon_choice = st.selectbox("Forecast Horizon", [30, 60, 90], index=0, format_func=lambda x: f"Next {x} Days")
            current_bal_input = manual_balance if use_manual_balance else None

        # Run Forecast
        with st.spinner("Computing time-series trend + weekly seasonality + monthly recurrence anchors..."):
            f_result = forecast_cashflow(
                df=df,
                horizon_days=horizon_choice,
                current_balance=current_bal_input,
                safety_buffer=safety_buffer,
            )

        # Metrics Callout
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Starting Balance", f"₹{f_result['current_balance']:,.2f}")
        with m2:
            lowest_val = f_result["lowest_balance"]
            st.metric(
                "Projected Lowest Balance",
                f"₹{lowest_val:,.2f}",
                delta=f"Occurs {f_result['lowest_balance_date']}",
                delta_color="inverse" if lowest_val < safety_buffer else "normal",
            )
        with m3:
            st.metric("Final Projected Balance", f"₹{f_result['final_projected_balance']:,.2f}")
        with m4:
            st.metric("Net Projected Cash Change", f"₹{f_result['total_projected_net']:+,.2f}")

        # Shortage Alert Banners
        if f_result["shortage_flag"]:
            st.error(
                f"⚠️ **CRITICAL CASH SHORTAGE DETECTED**: Your cash balance is projected to turn negative "
                f"(₹{f_result['lowest_balance']:,.2f}) on **{f_result['lowest_balance_date']}** "
                f"({f_result['lowest_balance_days_ahead']} days from now). Immediate working capital intervention required!"
            )
        elif f_result["safety_buffer_breach"]:
            st.warning(
                f"⚡ **SAFETY BUFFER BREACH WARNING**: Cash balance will drop to **₹{f_result['lowest_balance']:,.2f}** "
                f"on **{f_result['lowest_balance_date']}**, falling below your minimum safety reserve of ₹{safety_buffer:,.2f}."
            )
        else:
            st.success(
                f"✅ **HEALTHY CASH RUNWAY**: Projected lowest balance is **₹{f_result['lowest_balance']:,.2f}** "
                f"(on {f_result['lowest_balance_date']}), staying comfortably above your ₹{safety_buffer:,.2f} safety buffer."
            )

        # Plot Forecast
        hist_df = f_result["historical_daily"]
        fore_df = f_result["forecast_daily"]

        fig_fc = go.Figure()

        # Historical curve
        fig_fc.add_trace(go.Scatter(
            x=hist_df["date"],
            y=hist_df["historical_balance"],
            mode="lines",
            name="Historical Cash Balance",
            line=dict(color="#3b82f6", width=2.5),
        ))

        # Projected curve
        fig_fc.add_trace(go.Scatter(
            x=fore_df["date"],
            y=fore_df["projected_balance"],
            mode="lines+markers",
            name=f"Projected Balance ({horizon_choice}d)",
            line=dict(color="#8b5cf6", width=3, dash="dash"),
            marker=dict(size=4),
        ))

        # Safety Buffer Horizontal Line
        fig_fc.add_hline(
            y=safety_buffer,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text=f"Safety Buffer (₹{safety_buffer:,.0f})",
            annotation_position="top right",
        )

        # Zero Line
        fig_fc.add_hline(
            y=0,
            line_color="#ef4444",
            line_width=1.5,
            annotation_text="Zero Cash Deficit Line",
            annotation_position="bottom right",
        )

        # Lowest Point Pinpoint
        lowest_date_dt = pd.to_datetime(f_result["lowest_balance_date"])
        fig_fc.add_trace(go.Scatter(
            x=[lowest_date_dt],
            y=[f_result["lowest_balance"]],
            mode="markers+text",
            name="Lowest Projected Point",
            text=[f" Min: ₹{f_result['lowest_balance']:,.0f}"],
            textposition="top right",
            marker=dict(color="#ef4444", size=12, symbol="diamond"),
        ))

        fig_fc.update_layout(
            title=f"Cash Balance Trajectory & {horizon_choice}-Day Forward Forecast",
            xaxis_title="Timeline",
            yaxis_title="Cash Balance (₹)",
            hovermode="x unified",
            height=440,
            margin=dict(t=50, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Explainable Rationale Box
        with st.expander("🔍 See Detailed Explainable Time-Series Rationale", expanded=True):
            st.markdown(f"**Forecaster Narrative Summary:**\n\n{f_result['summary_text']}")
            meta = f_result["explainable_factors"]
            st.markdown(
                f"- **Underlying Trend:** {meta['trend_direction'].capitalize()} ({meta['daily_trend_slope']:+.2f} ₹/day drift)\n"
                f"- **Weekend Retail Multiplier:** Weekend net flows average ₹{meta['weekend_vs_weekday_diff']:+,.2f} vs weekdays\n"
                f"- **Detected Recurring Outflow Anchors (Days of Month):** {meta['significant_recurrence_days'] or 'None identified'}"
            )


# =============================================================
# TAB 3: ANOMALIES
# =============================================================
with tab3:
    st.subheader("Statistical & Pattern Outlier Audits")
    st.caption("Flags transactions that deviate significantly from category norms using Z-scores and IsolationForest.")

    if df.empty:
        st.warning("Please load or upload transaction data first in Tab 1.")
    else:
        with st.spinner("Auditing spending variance across categories..."):
            anomalies = detect_anomalies(df, z_threshold=z_threshold)

        if not anomalies:
            st.success("🎉 No unusual transactions or high-variance anomalies detected at current sensitivity threshold.")
        else:
            a_col1, a_col2 = st.columns([1, 1])
            total_anom_spend = sum(a["amount"] for a in anomalies)
            with a_col1:
                st.metric("Total Flagged Outliers", f"{len(anomalies)} transactions")
            with a_col2:
                st.metric("Total Anomalous Outflow", f"₹{total_anom_spend:,.2f}")

            st.markdown("#### 🚨 Flagged Irregular Expenses")

            for anom in anomalies:
                severity_label = "High Priority" if anom["severity"] == "High" else "Moderate Priority"
                severity_icon = "🚨" if anom["severity"] == "High" else "⚠️"

                with st.container(border=True):
                    col_title, col_badge = st.columns([4, 1])
                    with col_title:
                        st.markdown(f"**{anom['category']} — ₹{anom['amount']:,.2f}**")
                    with col_badge:
                        st.caption(f"{severity_icon} {severity_label}")

                    st.caption(
                        f"Description: {anom['description']} | Date: {anom['date']} | Method: {anom['detection_method']}"
                    )
                    st.markdown(f"💡 **Plain-Language Finding:** {anom['reason']}")


# =============================================================
# TAB 4: ASK AFFORDABILITY ADVISOR
# =============================================================
with tab4:
    st.subheader("Grounded Affordability & Purchasing Advisor")
    st.markdown(
        """
        Ask plain-language financial questions like *"Can I afford to purchase ₹80,000 of inventory next month?"*.
        <br><strong>The Explainability Principle:</strong> The advisor computes exact numeric balance trajectories and thresholds in Python <em>first</em>,
        and uses the LLM <em>only</em> to formulate a clear, transparent explanation of the already-computed math.
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("Please load or upload transaction data first in Tab 1.")
    else:
        # Precompute forecast and anomalies for query context
        current_bal_val = manual_balance if use_manual_balance else None
        f_context = forecast_cashflow(df, horizon_days=90, current_balance=current_bal_val, safety_buffer=safety_buffer)
        anom_context = detect_anomalies(df, z_threshold=z_threshold)

        st.markdown("##### ⚡ Quick Example Questions")
        example_cols = st.columns(3)
        sample_q1 = "Can I afford to purchase ₹80,000 of inventory next month?"
        sample_q2 = "Can I afford to spend ₹35,000 on an ad campaign in 14 days?"
        sample_q3 = "Can I afford to buy a ₹1,20,000 commercial equipment unit next quarter?"

        chosen_question = None
        with example_cols[0]:
            if st.button(f"📦 {sample_q1}", use_container_width=True):
                chosen_question = sample_q1
        with example_cols[1]:
            if st.button(f"📢 {sample_q2}", use_container_width=True):
                chosen_question = sample_q2
        with example_cols[2]:
            if st.button(f"🏭 {sample_q3}", use_container_width=True):
                chosen_question = sample_q3

        question_input = st.text_input(
            "Type your affordability question:",
            value=chosen_question or sample_q1,
            placeholder="e.g. Can I afford ₹80,000 inventory next month?",
        )

        # Optional amount / window manual override
        with st.expander("🛠️ Optional Parameter Overrides"):
            p_amt, p_win, p_item = parse_affordability_question(question_input)
            st.caption(f"Auto-parsed: Amount: ₹{p_amt or 0:,.2f} | Window: {p_win} days | Item: {p_item}")
            manual_amt = st.number_input("Override Purchase Amount (₹)", value=float(p_amt or 0.0), step=5000.0)

        if st.button("🧠 Evaluate Affordability", type="primary"):
            if not question_input.strip():
                st.error("Please enter a question.")
            else:
                amt_to_use = manual_amt if manual_amt > 0 else p_amt
                with st.spinner("Executing cash flow math first, then generating grounded explanation..."):
                    try:
                        ans = answer_affordability_question(
                            question=question_input,
                            amount=amt_to_use,
                            forecast_result=f_context,
                            current_balance=f_context["current_balance"],
                            anomalies=anom_context,
                            safety_buffer=safety_buffer,
                        )

                        # Verdict Banner
                        v = ans["verdict"]
                        if v == "safe":
                            st.success(
                                f"✅ VERDICT: SAFE TO PURCHASE\n\n"
                                f"You can comfortably proceed with this ₹{ans['purchase_amount']:,.2f} expenditure. "
                                f"Your cash reserves are projected to remain well above your ₹{ans['safety_buffer']:,.2f} safety buffer."
                            )
                        elif v == "risky":
                            st.warning(
                                f"⚠️ VERDICT: PROCEED WITH CAUTION (RISKY)\n\n"
                                f"This ₹{ans['purchase_amount']:,.2f} purchase is positive, but dips below your "
                                f"desired ₹{ans['safety_buffer']:,.2f} safety cushion."
                            )
                        else:
                            st.error(
                                f"🛑 VERDICT: NOT RECOMMENDED (DEFICIT RISK)\n\n"
                                f"Deducting ₹{ans['purchase_amount']:,.2f} will plunge your working capital into a negative "
                                f"deficit of ₹{ans['post_purchase_min_balance']:,.2f}."
                            )

                        # Mathematical Breakdown Cards
                        st.markdown("#### 🔢 Computed Mathematical Audit (Pre-LLM Math)")
                        bc1, bc2, bc3, bc4 = st.columns(4)
                        with bc1:
                            st.metric("Purchase Outflow", f"₹{ans['purchase_amount']:,.2f}", delta=f"{ans['window_days']}d Horizon")
                        with bc2:
                            st.metric("Pre-Purchase Low", f"₹{ans['pre_purchase_min_balance']:,.2f}", delta=f"on {ans['pre_purchase_min_date']}")
                        with bc3:
                            delta_post = ans["post_purchase_min_balance"] - ans["pre_purchase_min_balance"]
                            st.metric("Post-Purchase Low", f"₹{ans['post_purchase_min_balance']:,.2f}", delta=f"₹{delta_post:,.2f}")
                        with bc4:
                            st.metric("Safety Buffer Delta", f"₹{ans['buffer_delta']:+,.2f}", delta=f"Buffer: ₹{ans['safety_buffer']:,.2f}")

                        # Natural Language Explanation Card
                        st.markdown("#### 📜 Grounded Financial Advisor Explanation")
                        with st.container(border=True):
                            st.markdown(sanitize_markdown_text(ans["explanation"]))

                    except Exception as e:
                        st.error(f"Error evaluating affordability: {e}")
