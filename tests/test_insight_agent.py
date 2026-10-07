"""Unit tests for insight explainability agent."""

import pandas as pd
import pytest

from src.insight.insight_agent import (
    parse_affordability_question,
    answer_affordability_question,
)


def test_parse_affordability_question():
    """Verify regex parser extracts amount, window, and item correctly."""
    amt, win, item = parse_affordability_question("Can I afford to purchase ₹80,000 of inventory next month?")
    assert amt == 80000.0
    assert win == 30
    assert "inventory" in item.lower()

    amt2, win2, item2 = parse_affordability_question("Can I buy 50k of new equipment next week?")
    assert amt2 == 50000.0
    assert win2 == 7
    assert "equipment" in item2.lower()

    amt3, win3, item3 = parse_affordability_question("Can I afford 1.5 lakh marketing campaign in 60 days?")
    assert amt3 == 150000.0
    assert win3 == 60


def test_insight_verdict_safe():
    """Verify 'safe' verdict when balance comfortably stays above safety buffer."""
    forecast_daily = pd.DataFrame([
        {"date": f"2026-04-{i:02d}", "projected_balance": 300000.0} for i in range(1, 31)
    ])
    f_res = {
        "forecast_daily": forecast_daily,
        "current_balance": 300000.0,
        "safety_buffer": 50000.0,
    }

    res = answer_affordability_question(
        question="Can I afford to purchase Rs. 80,000 of inventory next month?",
        forecast_result=f_res,
        current_balance=300000.0,
        safety_buffer=50000.0,
    )

    # Post min balance: 300,000 - 80,000 = 220,000 >= 50,000
    assert res["verdict"] == "safe"
    assert res["post_purchase_min_balance"] == 220000.0
    assert res["purchase_amount"] == 80000.0
    assert "80,000" in res["explanation"]


def test_insight_verdict_risky():
    """Verify 'risky' verdict when balance is positive but dips below safety buffer."""
    forecast_daily = pd.DataFrame([
        {"date": f"2026-04-{i:02d}", "projected_balance": 100000.0} for i in range(1, 31)
    ])
    f_res = {
        "forecast_daily": forecast_daily,
        "current_balance": 100000.0,
        "safety_buffer": 50000.0,
    }

    res = answer_affordability_question(
        question="Can I afford to purchase Rs. 80,000 of inventory next month?",
        forecast_result=f_res,
        current_balance=100000.0,
        safety_buffer=50000.0,
    )

    # Post min balance: 100,000 - 80,000 = 20,000. 0 <= 20,000 < 50,000 (risky!)
    assert res["verdict"] == "risky"
    assert res["post_purchase_min_balance"] == 20000.0
    assert "RISKY" in res["explanation"].upper()


def test_insight_verdict_not_recommended():
    """Verify 'not recommended' verdict when balance dips below zero."""
    forecast_daily = pd.DataFrame([
        {"date": f"2026-04-{i:02d}", "projected_balance": 50000.0} for i in range(1, 31)
    ])
    f_res = {
        "forecast_daily": forecast_daily,
        "current_balance": 50000.0,
        "safety_buffer": 50000.0,
    }

    res = answer_affordability_question(
        question="Can I afford to purchase Rs. 80,000 of inventory next month?",
        forecast_result=f_res,
        current_balance=50000.0,
        safety_buffer=50000.0,
    )

    # Post min balance: 50,000 - 80,000 = -30,000 < 0
    assert res["verdict"] == "not recommended"
    assert res["post_purchase_min_balance"] == -30000.0
    assert "NOT RECOMMENDED" in res["explanation"].upper()


def test_insight_missing_amount_raises_error():
    """Verify ValueError is raised if no purchase amount can be identified."""
    with pytest.raises(ValueError, match="Could not determine a valid purchase amount"):
        answer_affordability_question("Can I afford to buy some things?")


def test_insight_requires_a_complete_forecast():
    with pytest.raises(ValueError, match="A cash flow forecast is required"):
        answer_affordability_question(
            "Can I afford to purchase ₹80,000 of inventory next month?"
        )

    with pytest.raises(ValueError, match="must cover the full 30-day"):
        answer_affordability_question(
            "Can I afford to purchase ₹80,000 of inventory next month?",
            forecast_result={
                "forecast_daily": pd.DataFrame(
                    [{"date": "2026-04-01", "projected_balance": 100000.0}]
                ),
                "current_balance": 100000.0,
            },
        )
