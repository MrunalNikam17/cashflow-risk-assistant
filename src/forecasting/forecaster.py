"""Cash flow forecasting using explainable time-series decomposition.

Decomposes cash flow into:
  1. Base trend (linear / moving average)
  2. Weekly day-of-week seasonality (retail weekend vs weekday rhythm)
  3. Monthly recurring commitments (predictable payroll, rent, EMI anchors)

Projects cumulative cash balance for 30/60/90 days and detects upcoming
shortages or safety buffer breaches.
"""

import logging
from datetime import timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.config import DEFAULT_HORIZON_DAYS, DEFAULT_SAFETY_BUFFER

logger = logging.getLogger(__name__)


def aggregate_daily_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transactions into continuous daily inflows, outflows, and net cashflow.

    Args:
        df: Input DataFrame conforming to unified schema.

    Returns:
        DataFrame with continuous daily dates, inflow, outflow, and net_cashflow.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "inflow", "outflow", "net_cashflow", "cumulative_net"])

    df_clean = df.copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"]).dt.normalize()

    # Separate inflows and outflows
    inflows = (
        df_clean[df_clean["direction"] == "inflow"]
        .groupby("date")["amount"]
        .sum()
        .rename("inflow")
    )
    outflows = (
        df_clean[df_clean["direction"] == "outflow"]
        .groupby("date")["amount"]
        .sum()
        .rename("outflow")
    )

    # Align on full date range
    min_date = df_clean["date"].min()
    max_date = df_clean["date"].max()
    full_idx = pd.date_range(min_date, max_date, freq="D", name="date")

    daily = pd.DataFrame(index=full_idx).join(inflows).join(outflows).fillna(0.0)
    daily["net_cashflow"] = daily["inflow"] - daily["outflow"]
    daily["cumulative_net"] = daily["net_cashflow"].cumsum()
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def extract_explainable_components(
    daily_df: pd.DataFrame,
    df_categorized: Optional[pd.DataFrame] = None,
) -> Tuple[float, float, Dict[int, float], Dict[int, float], Dict[str, any]]:
    """Extracts trend slope, intercept, day-of-week seasonality, and monthly recurrence.

    Returns:
        (trend_slope, trend_intercept, weekday_effects, monthly_day_effects, factor_meta)
    """
    n_days = len(daily_df)
    if n_days < 7:
        # Fallback for very small dataset (<7 days)
        mean_net = float(daily_df["net_cashflow"].mean()) if n_days > 0 else 0.0
        factor_meta = {
            "trend_direction": "stable",
            "daily_trend_slope": 0.0,
            "weekend_vs_weekday_diff": 0.0,
            "significant_recurrence_days": [],
        }
        return 0.0, mean_net, {i: 0.0 for i in range(7)}, {}, factor_meta

    x = np.arange(n_days)
    y = daily_df["net_cashflow"].values

    # 1. Base Linear Trend
    p = np.polyfit(x, y, deg=1)
    slope = float(p[0])
    intercept = float(p[1])

    detrended = y - (slope * x + intercept)
    daily_copy = daily_df.copy()
    daily_copy["detrended"] = detrended
    daily_copy["weekday"] = daily_copy["date"].dt.weekday
    daily_copy["day_of_month"] = daily_copy["date"].dt.day

    # 2. Weekly Day-of-Week Seasonality (0=Mon ... 6=Sun)
    weekday_effects = daily_copy.groupby("weekday")["detrended"].mean().to_dict()

    # 3. Monthly Recurring Anchors (Days with large predictable outflows/inflows e.g. Rent on 1st, Payroll on 5th)
    # Filter for day-of-month spikes that deviate significantly from zero
    dom_effects = daily_copy.groupby("day_of_month")["detrended"].mean()
    std_residual = float(daily_copy["detrended"].std()) if len(daily_copy) > 1 else 1.0
    # Keep only days where the average residual exceeds 0.75 * std (meaning clear recurring event)
    significant_dom = dom_effects[dom_effects.abs() > (0.75 * std_residual)].to_dict()

    # Meta explanation notes
    trend_direction = "upward" if slope > 5 else ("downward" if slope < -5 else "stable")
    weekend_lift = (weekday_effects.get(5, 0.0) + weekday_effects.get(6, 0.0)) / 2 - (
        sum(weekday_effects.get(i, 0.0) for i in range(5)) / 5
    )

    factor_meta = {
        "trend_direction": trend_direction,
        "daily_trend_slope": round(slope, 2),
        "weekend_vs_weekday_diff": round(weekend_lift, 2),
        "significant_recurrence_days": list(significant_dom.keys()),
    }

    return slope, intercept, weekday_effects, significant_dom, factor_meta


def forecast_cashflow(
    df: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    current_balance: Optional[float] = None,
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
) -> Dict[str, any]:
    """Forecasts cash flow for the next N days using explainable additive decomposition.

    Args:
        df: Categorized transactions DataFrame.
        horizon_days: Projection window (30, 60, or 90 days).
        current_balance: Starting cash balance. If None, inferred from bank statements or cumulative net.
        safety_buffer: Minimum desired cash buffer threshold.

    Returns:
        Dictionary containing historical series, forecast series, lowest projected point,
        shortage alert, and narrative explanation.
    """
    if df.empty:
        raise ValueError("Cannot forecast on an empty transactions DataFrame.")
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")

    daily = aggregate_daily_cashflow(df)

    # Determine starting balance
    if current_balance is not None:
        start_bal = float(current_balance)
    elif df.attrs.get("current_balance") is not None:
        start_bal = float(df.attrs["current_balance"])
    else:
        cum_last = float(daily["cumulative_net"].iloc[-1]) if not daily.empty else 100000.0
        start_bal = max(cum_last, 50000.0)

    # Attach running historical balance to daily
    # Align the end of historical balance to start_bal
    hist_cum = daily["cumulative_net"].values
    offset = start_bal - hist_cum[-1] if len(hist_cum) > 0 else start_bal
    daily["historical_balance"] = daily["cumulative_net"] + offset

    # Time series decomposition
    slope, intercept, weekday_effects, dom_effects, factor_meta = extract_explainable_components(daily, df)

    last_date = daily["date"].max()
    n_hist = len(daily)

    # Estimate typical baseline daily inflow vs outflow ratio for projection
    mean_inflow = float(daily["inflow"].mean()) if daily["inflow"].mean() > 0 else 5000.0
    mean_outflow = float(daily["outflow"].mean()) if daily["outflow"].mean() > 0 else 4000.0

    future_rows = []
    running_balance = start_bal

    for step in range(1, horizon_days + 1):
        future_date = last_date + timedelta(days=step)
        x_step = n_hist + step

        # Component 1: Trend baseline
        base_net = slope * x_step + intercept

        # Component 2: Weekly seasonality
        w_effect = weekday_effects.get(future_date.weekday(), 0.0)

        # Component 3: Monthly recurring anchors
        m_effect = dom_effects.get(future_date.day, 0.0)

        projected_net = base_net + w_effect + m_effect

        # Deconstruct into projected inflow and outflow for realism
        if projected_net >= 0:
            proj_inflow = mean_inflow + (projected_net * 0.7)
            proj_outflow = max(0.0, proj_inflow - projected_net)
        else:
            proj_outflow = mean_outflow + (abs(projected_net) * 0.7)
            proj_inflow = max(0.0, proj_outflow + projected_net)

        running_balance += projected_net

        future_rows.append({
            "date": future_date,
            "projected_inflow": round(proj_inflow, 2),
            "projected_outflow": round(proj_outflow, 2),
            "projected_net": round(projected_net, 2),
            "projected_balance": round(running_balance, 2),
        })

    forecast_df = pd.DataFrame(future_rows)

    # Calculate key metrics
    lowest_row = forecast_df.loc[forecast_df["projected_balance"].idxmin()]
    lowest_balance = float(lowest_row["projected_balance"])
    lowest_date_str = pd.to_datetime(lowest_row["date"]).strftime("%Y-%m-%d")
    days_to_low = (pd.to_datetime(lowest_row["date"]) - last_date).days

    final_balance = float(forecast_df["projected_balance"].iloc[-1])
    total_proj_inflow = float(forecast_df["projected_inflow"].sum())
    total_proj_outflow = float(forecast_df["projected_outflow"].sum())
    total_proj_net = float(forecast_df["projected_net"].sum())

    shortage_flag = lowest_balance < 0.0
    safety_buffer_breach = lowest_balance < safety_buffer

    # Construct explainable summary text
    trend_text = f"a {factor_meta['trend_direction']} net cash flow trajectory"
    if factor_meta["weekend_vs_weekday_diff"] > 500:
        season_text = "with significant retail revenue surges on weekends"
    else:
        season_text = "with consistent weekly turnover"

    if shortage_flag:
        risk_alert = (
            f"CRITICAL CASH DEFICIT PROJECTED: Cash balance is projected to fall below zero "
            f"to Rs. {lowest_balance:,.2f} on {lowest_date_str} ({days_to_low} days ahead)."
        )
    elif safety_buffer_breach:
        risk_alert = (
            f"SAFETY BUFFER WARNING: Cash balance is projected to dip to a low of "
            f"Rs. {lowest_balance:,.2f} on {lowest_date_str} ({days_to_low} days ahead), "
            f"which is below your desired safety buffer of Rs. {safety_buffer:,.2f}."
        )
    else:
        risk_alert = (
            f"HEALTHY CASH FLOW: Lowest projected cash balance is Rs. {lowest_balance:,.2f} "
            f"on {lowest_date_str}, comfortably above your safety threshold of Rs. {safety_buffer:,.2f}."
        )

    summary_text = (
        f"Over the next {horizon_days} days, the business exhibits {trend_text} {season_text}. "
        f"Total projected cash inflow is Rs. {total_proj_inflow:,.2f} against Rs. {total_proj_outflow:,.2f} in outflows "
        f"(net change: Rs. {total_proj_net:+,.2f}). {risk_alert}"
    )

    return {
        "historical_daily": daily,
        "forecast_daily": forecast_df,
        "current_balance": round(start_bal, 2),
        "lowest_balance": round(lowest_balance, 2),
        "lowest_balance_date": lowest_date_str,
        "lowest_balance_days_ahead": days_to_low,
        "final_projected_balance": round(final_balance, 2),
        "shortage_flag": shortage_flag,
        "safety_buffer_breach": safety_buffer_breach,
        "safety_buffer": float(safety_buffer),
        "horizon_days": horizon_days,
        "total_projected_inflow": round(total_proj_inflow, 2),
        "total_projected_outflow": round(total_proj_outflow, 2),
        "total_projected_net": round(total_proj_net, 2),
        "explainable_factors": factor_meta,
        "summary_text": summary_text,
    }
