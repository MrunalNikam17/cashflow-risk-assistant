"""Generate deterministic, date-relative sample finance data near the current date."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import SAMPLE_DATA_DIR


def generate_all_sample_data(output_dir: Path = SAMPLE_DATA_DIR) -> dict:
    """Generate deterministic sample CSV files relative to today."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    today = date.today()
    start = today - timedelta(days=180)

    sales_records = []
    expense_records = []
    invoice_records = []
    bank_records = []

    # Sales: daily POS with weekend lift, deterministic but relative to current date.
    for offset in range(180):
        current = start + timedelta(days=offset)
        weekday = current.weekday()
        base = 18000 + 800 * np.sin(offset / 10)
        if weekday in (5, 6):
            base *= 1.7
        base += rng.normal(0, 1200)
        amount = round(float(max(12000.0, base)), 2)
        sales_records.append(
            {
                "date": current.isoformat(),
                "description": "Retail POS sales",
                "amount": amount,
                "direction": "inflow",
                "source_file": "sales.csv",
                "category": "Sales Revenue",
            }
        )

    # Recurring expenses with a deterministic rent/payroll cadence.
    for month_delta in range(7):
        month_date = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        month_date = (month_date.replace(day=1) + timedelta(days=month_delta * 30))
        for pay_day, label, amount in [
            (1, "Rent", 42000.0),
            (5, "Salary", 68000.0),
            (12, "Utilities", 8000.0),
            (20, "Loan Repayment", 18000.0),
        ]:
            pay_date = month_date.replace(day=min(pay_day, 28))
            expense_records.append(
                {
                    "date": pay_date.isoformat(),
                    "description": label,
                    "amount": amount,
                    "direction": "outflow",
                    "source_file": "expenses.csv",
                    "category": label,
                }
            )

    # Attach a few purchase anomalies and some marketing spend.
    for offset in [14, 30, 62, 95, 124]:
        current = start + timedelta(days=offset)
        expense_records.append(
            {
                "date": current.isoformat(),
                "description": "Campaign spend",
                "amount": round(float(6000 + offset * 12), 2),
                "direction": "outflow",
                "source_file": "expenses.csv",
                "category": "Marketing",
            }
        )

    # Invoices: a mix of paid and pending items relative to today.
    for idx in range(18):
        due = today - timedelta(days=35 + idx * 2)
        amount = 12000 + idx * 1800 + (idx % 3) * 500
        invoice_records.append(
            {
                "date": due.isoformat(),
                "description": f"Invoice #{idx + 1001}",
                "amount": round(float(amount), 2),
                "direction": "inflow",
                "status": "paid" if idx % 3 == 0 else "pending",
                "payment_date": (due + timedelta(days=3)).isoformat() if idx % 3 == 0 else "",
                "source_file": "invoices.csv",
            }
        )

    # Bank statement ledger with a running balance near today.
    balance = 180000.0
    ledger_date = start
    for idx, sale in enumerate(sales_records[:70]):
        if idx % 6 == 0:
            balance -= 7200.0
        balance += sale["amount"]
        bank_records.append(
            {
                "date": sale["date"],
                "description": f"POS settlement {idx + 1}",
                "amount": round(float(sale["amount"]), 2),
                "direction": "inflow",
                "source_file": "bank_statement.csv",
                "balance": round(balance, 2),
            }
        )

    for idx, expense in enumerate(expense_records[:20]):
        balance -= expense["amount"]
        bank_records.append(
            {
                "date": expense["date"],
                "description": expense["description"],
                "amount": round(float(expense["amount"]), 2),
                "direction": "outflow",
                "source_file": "bank_statement.csv",
                "balance": round(balance, 2),
            }
        )

    sales_df = pd.DataFrame(sales_records)
    expense_df = pd.DataFrame(expense_records)
    invoice_df = pd.DataFrame(invoice_records)
    bank_df = pd.DataFrame(bank_records)

    sales_df.to_csv(output_dir / "sales.csv", index=False)
    expense_df.to_csv(output_dir / "expenses.csv", index=False)
    invoice_df.to_csv(output_dir / "invoices.csv", index=False)
    bank_df.to_csv(output_dir / "bank_statement.csv", index=False)

    return {
        "sales": output_dir / "sales.csv",
        "expenses": output_dir / "expenses.csv",
        "invoices": output_dir / "invoices.csv",
        "bank_statement": output_dir / "bank_statement.csv",
    }


if __name__ == "__main__":
    generate_all_sample_data()
