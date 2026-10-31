from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.anomaly.anomaly_detector import detect_anomalies
from src.categorization.categorizer import categorize_transactions
from src.forecasting.forecaster import forecast_cashflow
from src.ingestion.loader import load_file
from src.insight.insight_agent import answer_affordability_question


def test_dashboard_affordability_advisor_renders_with_sample_data():
    at = AppTest.from_file("dashboard/app.py")
    at.run(timeout=30)
    assert not at.exception

    assert len(at.sidebar.button) > 0
    at.sidebar.button[0].click()
    at.run(timeout=30)
    assert not at.exception

    assert len(at.tabs) >= 4

    app_source = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "with st.container(border=True):" in app_source
    assert 'st.markdown(sanitize_markdown_text(ans["explanation"]))' in app_source
    assert "background:#ffffff" not in app_source.lower()
    assert "VERDICT" in app_source

    question = "Can I afford ₹80,000 of inventory next month?"
    bank_statement = Path("data/sample/bank_statement.csv")
    df = categorize_transactions(load_file(bank_statement))
    f_context = forecast_cashflow(df, horizon_days=90, current_balance=None, safety_buffer=25000.0)
    anom_context = detect_anomalies(df, z_threshold=2.0)
    ans = answer_affordability_question(
        question=question,
        amount=80000.0,
        forecast_result=f_context,
        current_balance=f_context["current_balance"],
        anomalies=anom_context,
        safety_buffer=25000.0,
    )

    assert ans["verdict"] in {"safe", "risky", "not recommended"}
    assert "80,000" in ans["explanation"] or "₹80,000" in ans["explanation"]
    assert "inventory" in ans["explanation"].lower()
