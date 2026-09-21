"""Unit tests for transaction categorizer."""

import pandas as pd
import pytest

from src.categorization.categorizer import (
    categorize_transactions,
    rule_based_fallback,
    clean_json_response,
)
from src.utils.config import CATEGORIES


def test_rule_based_fallback_categories():
    """Verify rule-based keywords map to standard categories."""
    assert rule_based_fallback("Monthly Retail Store Lease Rent") == "Rent"
    assert rule_based_fallback("Staff Salaries Disbursal for October") == "Salary"
    assert rule_based_fallback("Bulk Inventory Stock Supplier Payment") == "Inventory Purchase"
    assert rule_based_fallback("Commercial Electricity Meter Power Bill") == "Utilities"
    assert rule_based_fallback("POS Card Settlement Batch", direction="inflow") == "Sales Revenue"
    assert rule_based_fallback("Business Loan Monthly EMI") == "Loan Repayment"
    assert rule_based_fallback("Meta Ad Campaign for Store Promo") == "Marketing"
    assert rule_based_fallback("Quarterly GST Direct Challan Tax") == "Taxes"
    assert rule_based_fallback("Industrial Chiller Compressor Overhaul") == "Equipment"
    assert rule_based_fallback("Random Unclassified Expense 12345") == "Other"


def test_clean_json_response():
    """Verify markdown code blocks are properly stripped."""
    raw_markdown = """```json
[
  {"index": 0, "category": "Rent"}
]
```"""
    cleaned = clean_json_response(raw_markdown)
    assert cleaned.strip().startswith("[")
    assert cleaned.strip().endswith("]")


def test_categorize_transactions_offline():
    """Verify categorizer processes DataFrame and fills categories without crashing."""
    data = [
        {"date": "2026-01-01", "description": "Landlord Rent Payment", "amount": 40000.0, "direction": "outflow", "source_file": "test.csv", "category": None},
        {"date": "2026-01-05", "description": "Wages and Salary Payroll", "amount": 65000.0, "direction": "outflow", "source_file": "test.csv", "category": None},
        {"date": "2026-01-10", "description": "Customer Retail Sale", "amount": 8500.0, "direction": "inflow", "source_file": "test.csv", "category": None},
        {"date": "2026-01-15", "description": "Unknown Mystery Charge", "amount": 1200.0, "direction": "outflow", "source_file": "test.csv", "category": None},
    ]
    df = pd.DataFrame(data)
    df_cat = categorize_transactions(df, batch_size=2)

    assert len(df_cat) == 4
    assert df_cat.loc[0, "category"] == "Rent"
    assert df_cat.loc[1, "category"] == "Salary"
    assert df_cat.loc[2, "category"] == "Sales Revenue"
    assert df_cat.loc[3, "category"] in CATEGORIES
    # Ensure every single category belongs to the allowed list
    for cat in df_cat["category"]:
        assert cat in CATEGORIES


def test_categorize_preserves_existing_category():
    """Verify already-categorized rows are not overwritten."""
    data = [
        {"date": "2026-01-01", "description": "Custom Tooling", "amount": 5000.0, "direction": "outflow", "source_file": "test.csv", "category": "Equipment"},
    ]
    df = pd.DataFrame(data)
    df_cat = categorize_transactions(df)
    assert df_cat.loc[0, "category"] == "Equipment"


def test_categorize_empty_dataframe():
    """Verify empty DataFrame returns empty copy without errors."""
    empty_df = pd.DataFrame(columns=["date", "description", "amount", "direction", "source_file", "category"])
    result = categorize_transactions(empty_df)
    assert result.empty
