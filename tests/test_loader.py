"""Tests for normalizing financial files into the unified transaction schema."""

import pandas as pd

from src.forecasting.forecaster import forecast_cashflow
from src.ingestion.loader import load_file, load_files


def test_load_signed_amounts_and_expense_direction(tmp_path):
    expense_file = tmp_path / "expenses.csv"
    pd.DataFrame(
        [
            {"date": "2026-01-01", "description": "Supplier payment", "amount": 1250},
            {"date": "2026-01-02", "description": "Refund reversal", "amount": -300},
        ]
    ).to_csv(expense_file, index=False)

    transactions = load_file(expense_file)

    assert transactions["amount"].tolist() == [1250.0, 300.0]
    assert transactions["direction"].tolist() == ["outflow", "outflow"]
    assert transactions["source_file"].eq("expenses.csv").all()


def test_load_debit_only_file_and_preserve_latest_bank_balance(tmp_path):
    statement_file = tmp_path / "statement.csv"
    pd.DataFrame(
        [
            {"date": "2026-01-01", "description": "Opening transaction", "debit": 500, "balance": 9500},
            {"date": "2026-01-02", "description": "Signed reversal", "debit": -100, "balance": 9600},
        ]
    ).to_csv(statement_file, index=False)

    transactions = load_file(statement_file)
    combined = load_files([statement_file])
    forecast = forecast_cashflow(
        transactions,
        horizon_days=1,
        safety_buffer=0,
    )

    assert transactions["amount"].tolist() == [500.0, 100.0]
    assert transactions["direction"].tolist() == ["outflow", "inflow"]
    assert combined.attrs["current_balance"] == 9600.0
    assert forecast["current_balance"] == 9600.0
