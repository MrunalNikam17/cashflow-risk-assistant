"""Flags unusual transactions using statistical methods (Z-score, MAD, IsolationForest).

Computes spending deviations per category and generates human-readable explanations:
e.g. "This Rs. 45,000 'Utilities' charge is 6.0x your typical monthly utilities spend of Rs. 7,500".
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils.config import (
    DEFAULT_ZSCORE_THRESHOLD,
    DEFAULT_ISOLATION_CONTAMINATION,
    MIN_TRANSACTIONS_FOR_ZSCORE,
)

logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    z_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    use_isolation_forest: bool = True,
    contamination: float = DEFAULT_ISOLATION_CONTAMINATION,
) -> List[Dict[str, any]]:
    """Detects anomalous transactions across categories.

    Combines robust per-category statistical analysis (Z-score and median-relative multiplier)
    with multi-variate IsolationForest, producing clear, human-understandable explanations.

    Args:
        df: Normalized & categorized transactions DataFrame.
        z_threshold: Standard deviations or modified z-score threshold (default 2.0).
        use_isolation_forest: If True, also fits IsolationForest across multi-features.
        contamination: Proportion of outliers expected for IsolationForest.

    Returns:
        List of anomaly dictionaries sorted by anomaly severity.
    """
    if df.empty:
        return []

    df_clean = df.copy()
    df_clean["amount"] = pd.to_numeric(df_clean["amount"], errors="coerce").fillna(0.0)
    df_clean["date"] = pd.to_datetime(df_clean["date"])

    # Focus anomaly detection on expenses/outflows
    outflows = df_clean[df_clean["direction"] == "outflow"].copy()
    if outflows.empty:
        return []

    anomalies_map: Dict[int, Dict[str, any]] = {}

    # -------------------------------------------------------------
    # 1. Per-Category Statistical Analysis (Robust Z-Score & Multiplier)
    # -------------------------------------------------------------
    categories = outflows["category"].dropna().unique()

    for cat in categories:
        cat_mask = outflows["category"] == cat
        cat_indices = outflows[cat_mask].index
        amounts = outflows.loc[cat_mask, "amount"].values
        n_tx = len(amounts)

        if n_tx < MIN_TRANSACTIONS_FOR_ZSCORE:
            continue

        median_amt = float(np.median(amounts))
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts, ddof=1)) if n_tx > 1 else 0.0

        # Calculate Median Absolute Deviation (MAD) for robust outlier detection
        mad = float(np.median(np.abs(amounts - median_amt)))
        # Base typical spend reference: prefer median for categories with wild spikes
        typical_ref = median_amt if median_amt > 0 else (mean_amt if mean_amt > 0 else 1.0)

        for idx in cat_indices:
            amt = float(outflows.at[idx, "amount"])
            # Standard z-score
            z_std = (amt - mean_amt) / std_amt if std_amt > 0 else 0.0
            # Modified z-score using MAD
            mod_z = (0.6745 * (amt - median_amt) / mad) if mad > 0 else z_std
            effective_z = max(z_std, mod_z)

            multiplier = amt / typical_ref if typical_ref > 0 else 1.0

            # Flag if z exceeds threshold or if amount is significantly higher than typical (>= 2.5x)
            is_anomaly = (effective_z >= z_threshold and multiplier >= 1.5) or (multiplier >= 2.5 and amt > 10000.0)

            if is_anomaly:
                sev = "High" if (effective_z >= 3.5 or multiplier >= 4.0) else "Medium"
                reason = (
                    f"This Rs. {amt:,.2f} '{cat}' charge is {multiplier:.1f}x your typical "
                    f"{cat.lower()} transaction of Rs. {typical_ref:,.2f} (z-score: {effective_z:.1f})."
                )
                anomalies_map[idx] = {
                    "date": outflows.at[idx, "date"].strftime("%Y-%m-%d"),
                    "description": str(outflows.at[idx, "description"]),
                    "category": str(cat),
                    "amount": round(amt, 2),
                    "direction": "outflow",
                    "typical_mean": round(typical_ref, 2),
                    "typical_std": round(std_amt, 2),
                    "multiplier": round(multiplier, 1),
                    "z_score": round(float(effective_z), 2),
                    "severity": sev,
                    "detection_method": "Statistical Z-Score / MAD",
                    "reason": reason,
                    "source_file": str(outflows.at[idx, "source_file"]),
                }

    # -------------------------------------------------------------
    # 2. Multi-Variate Isolation Forest (Complementary)
    # -------------------------------------------------------------
    if use_isolation_forest and len(outflows) >= 15:
        try:
            feat_df = pd.DataFrame()
            feat_df["amount"] = outflows["amount"]
            feat_df["log_amount"] = np.log1p(outflows["amount"])
            feat_df["day_of_week"] = outflows["date"].dt.weekday
            feat_df["day_of_month"] = outflows["date"].dt.day

            cat_dummies = pd.get_dummies(outflows["category"], prefix="cat", drop_first=False)
            feat_combined = pd.concat([feat_df, cat_dummies], axis=1)

            iso = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
            )
            outlier_preds = iso.fit_predict(feat_combined)
            scores = iso.score_samples(feat_combined)

            for i, (orig_idx, row) in enumerate(outflows.iterrows()):
                if outlier_preds[i] == -1:
                    amt = float(row["amount"])
                    cat = str(row.get("category", "Other"))
                    cat_mean = float(outflows[outflows["category"] == cat]["amount"].median())
                    multiplier = (amt / cat_mean) if cat_mean > 0 else 1.0

                    if orig_idx not in anomalies_map and amt > cat_mean and multiplier >= 2.0:
                        reason = (
                            f"Multi-variate pattern outlier: Rs. {amt:,.2f} in '{cat}' "
                            f"({multiplier:.1f}x typical of Rs. {cat_mean:,.2f})."
                        )
                        anomalies_map[orig_idx] = {
                            "date": row["date"].strftime("%Y-%m-%d"),
                            "description": str(row["description"]),
                            "category": cat,
                            "amount": round(amt, 2),
                            "direction": "outflow",
                            "typical_mean": round(cat_mean, 2),
                            "typical_std": 0.0,
                            "multiplier": round(multiplier, 1),
                            "z_score": round(float(-scores[i] * 5), 2),
                            "severity": "Medium",
                            "detection_method": "IsolationForest",
                            "reason": reason,
                            "source_file": str(row.get("source_file", "")),
                        }
                    elif orig_idx in anomalies_map:
                        anomalies_map[orig_idx]["detection_method"] = "Ensemble (Z-Score + IsolationForest)"
        except Exception as e:
            logger.warning(f"IsolationForest anomaly detection failed: {e}")

    # Deduplicate entries with identical date, category, and amount (e.g. from bank statement + expense sheet)
    unique_anomalies = []
    seen = set()
    for a in anomalies_map.values():
        key = (a["date"], a["category"], a["amount"])
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    unique_anomalies.sort(key=lambda x: (x["z_score"], x["amount"]), reverse=True)
    return unique_anomalies
