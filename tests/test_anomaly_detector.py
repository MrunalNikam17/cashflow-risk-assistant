"""Unit tests for anomaly detector."""

import pandas as pd
import pytest

from src.anomaly.anomaly_detector import detect_anomalies


def test_detect_anomalies_outlier():
    """Verify statistical anomaly detection flags massive spending deviation."""
    rows = []
    # 10 typical utility charges of ~7,500
    for i in range(10):
        rows.append({
            "date": f"2025-10-{10+i:02d}",
            "description": "Monthly utility bill",
            "amount": 7500.0 + (i * 50),
            "direction": "outflow",
            "source_file": "expenses.csv",
            "category": "Utilities",
        })
    # Injected massive utility anomaly
    rows.append({
        "date": "2026-01-14",
        "description": "Commercial power surge penalty bill",
        "amount": 45000.0,
        "direction": "outflow",
        "source_file": "expenses.csv",
        "category": "Utilities",
    })

    df = pd.DataFrame(rows)
    anomalies = detect_anomalies(df, z_threshold=2.0)

    assert len(anomalies) >= 1
    flagged = [a for a in anomalies if a["amount"] == 45000.0]
    assert len(flagged) == 1
    anom = flagged[0]
    assert anom["category"] == "Utilities"
    assert anom["amount"] == 45000.0
    assert anom["multiplier"] >= 5.0
    assert "reason" in anom
    assert "45,000" in anom["reason"]
    assert "Utilities" in anom["reason"]


def test_detect_anomalies_empty():
    """Verify empty DataFrame returns empty list without errors."""
    empty_df = pd.DataFrame(columns=["date", "description", "amount", "direction", "source_file", "category"])
    assert detect_anomalies(empty_df) == []


def test_detect_anomalies_low_sample_count():
    """Verify categories with <3 transactions do not cause division-by-zero or crashes."""
    rows = [
        {"date": "2026-01-01", "description": "One-off legal advisory", "amount": 15000.0, "direction": "outflow", "source_file": "e.csv", "category": "Other"},
        {"date": "2026-01-02", "description": "Second advisory", "amount": 16000.0, "direction": "outflow", "source_file": "e.csv", "category": "Other"},
    ]
    df = pd.DataFrame(rows)
    # With only 2 transactions, it should not crash
    anomalies = detect_anomalies(df, use_isolation_forest=False)
    assert isinstance(anomalies, list)
