"""Unit tests for cash flow forecaster."""

from datetime import date, timedelta
import pandas as pd
import pytest

from src.forecasting.forecaster import aggregate_daily_cashflow, forecast_cashflow


@pytest.fixture
def sample_tx_df():
    """Generates 30 days of sample transactions."""
    rows = []
    base_date = date(2026, 1, 1)
    for i in range(30):
        d = (base_date + timedelta(days=i)).isoformat()
        # Daily sales
        rows.append({
            "date": d,
            "description": "Daily retail sales",
            "amount": 10000.0,
            "direction": "inflow",
            "source_file": "sales.csv",
            "category": "Sales Revenue",
        })
        # Routine daily expenses
        rows.append({
            "date": d,
            "description": "Daily operational expense",
            "amount": 7000.0,
            "direction": "outflow",
            "source_file": "expenses.csv",
            "category": "Other",
        })
    return pd.DataFrame(rows)


def test_aggregate_daily_cashflow(sample_tx_df):
    """Verify daily aggregation produces continuous timeline and correct net values."""
    daily = aggregate_daily_cashflow(sample_tx_df)
    assert len(daily) == 30
    assert "inflow" in daily.columns
    assert "outflow" in daily.columns
    assert "net_cashflow" in daily.columns
    # Daily net should be 10,000 - 7,000 = 3,000
    assert daily["net_cashflow"].iloc[0] == 3000.0
    assert daily["cumulative_net"].iloc[-1] == 90000.0


def test_forecast_horizons(sample_tx_df):
    """Verify forecast works across standard horizons (30, 60, 90 days)."""
    for h in [30, 60, 90]:
        res = forecast_cashflow(sample_tx_df, horizon_days=h, current_balance=50000.0)
        assert res["horizon_days"] == h
        assert len(res["forecast_daily"]) == h
        assert "lowest_balance" in res
        assert "lowest_balance_date" in res
        assert "summary_text" in res
        assert isinstance(res["shortage_flag"], (bool, type(True)))


def test_forecast_shortage_detection():
    """Verify forecaster flags negative balances when outflows exceed balance."""
    rows = [
        {"date": "2026-01-01", "description": "Minor Sale", "amount": 1000.0, "direction": "inflow", "source_file": "s.csv", "category": "Sales Revenue"},
        {"date": "2026-01-02", "description": "Huge Recurring Rent", "amount": 50000.0, "direction": "outflow", "source_file": "e.csv", "category": "Rent"},
    ]
    df = pd.DataFrame(rows)
    # Starting with only 10,000 balance
    res = forecast_cashflow(df, horizon_days=30, current_balance=10000.0, safety_buffer=20000.0)
    assert res["lowest_balance"] < 0.0
    assert res["shortage_flag"] is True
    assert res["safety_buffer_breach"] is True


def test_forecast_safety_buffer_healthy(sample_tx_df):
    """Verify forecaster flags safe status when balance stays above buffer."""
    res = forecast_cashflow(sample_tx_df, horizon_days=30, current_balance=200000.0, safety_buffer=50000.0)
    assert res["shortage_flag"] is False
    assert res["lowest_balance"] >= 50000.0
    assert res["safety_buffer_breach"] is False


def test_forecast_uses_bank_balance_metadata(sample_tx_df):
    sample_tx_df.attrs["current_balance"] = 0.0

    result = forecast_cashflow(sample_tx_df, horizon_days=1)

    assert result["current_balance"] == 0.0


def test_forecast_rejects_nonpositive_horizon(sample_tx_df):
    with pytest.raises(ValueError, match="horizon_days must be a positive integer"):
        forecast_cashflow(sample_tx_df, horizon_days=0)
