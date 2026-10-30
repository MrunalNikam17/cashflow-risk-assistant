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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_RECEIVABLE_COLLECTION_DELAY_DAYS,
    DEFAULT_SAFETY_BUFFER,
)

logger = logging.getLogger(__name__)


def aggregate_daily_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transactions into continuous daily inflows, outflows, and net cashflow."""
    if df.empty:
        return pd.DataFrame(columns=["date", "inflow", "outflow", "net_cashflow", "cumulative_net"])

    df_clean = df.copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"]).dt.normalize()

    inflows = (
        df_clean[df_clean["direction"] == "inflow"].groupby("date")["amount"].sum().rename("inflow")
    )
    outflows = (
        df_clean[df_clean["direction"] == "outflow"].groupby("date")["amount"].sum().rename("outflow")
    )

    min_date = df_clean["date"].min()
    max_date = df_clean["date"].max()
    full_idx = pd.date_range(min_date, max_date, freq="D", name="date")

    daily = pd.DataFrame(index=full_idx).join(inflows).join(outflows).fillna(0.0)
    daily["net_cashflow"] = daily["inflow"] - daily["outflow"]
    daily["cumulative_net"] = daily["net_cashflow"].cumsum()
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def _estimate_recurring_day_effects(daily_df: pd.DataFrame, anomalies: Optional[pd.DataFrame] = None) -> Dict[int, float]:
    """Estimate recurring day-of-month effects using month-level medians; anomalies are excluded."""
    daily_copy = daily_df.copy()
    daily_copy["day_of_month"] = daily_copy["date"].dt.day

    if anomalies is not None and not anomalies.empty:
        flagged_dates = pd.to_datetime(anomalies["date"]).dt.normalize()
        daily_copy = daily_copy[~daily_copy["date"].isin(flagged_dates)].copy()

    if daily_copy.empty:
        return {}

    monthly_day_frame = []
    for month in pd.PeriodIndex(daily_copy["date"], freq="M"):
        month_group = daily_copy[daily_copy["date"].dt.to_period("M") == month]
        if month_group.empty:
            continue
        month_series = month_group.groupby("day_of_month")["net_cashflow"].sum()
        for day, value in month_series.items():
            monthly_day_frame.append({"month": month, "day_of_month": int(day), "net_cashflow": float(value)})

    if not monthly_day_frame:
        return {}

    day_summary = pd.DataFrame(monthly_day_frame)
    if day_summary.empty:
        return {}

    month_counts = day_summary.groupby("day_of_month")["month"].nunique()
    recurring_days = month_counts[month_counts >= 3].index
    if recurring_days.empty:
        return {}

    median_by_day = day_summary[day_summary["day_of_month"].isin(recurring_days)].groupby("day_of_month")["net_cashflow"].median()
    return {int(day): float(median_by_day[day]) for day in recurring_days}


def extract_explainable_components(
    daily_df: pd.DataFrame,
    df_categorized: Optional[pd.DataFrame] = None,
) -> Tuple[float, float, Dict[int, float], Dict[int, float], Dict[str, Any]]:
    """Extracts trend slope, intercept, day-of-week seasonality, and monthly recurrence."""
    n_days = len(daily_df)
    if n_days < 7:
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
    slope, intercept = np.polyfit(x, y, deg=1)
    slope = float(slope)
    intercept = float(intercept)

    detrended = y - (slope * x + intercept)
    daily_copy = daily_df.copy()
    daily_copy["detrended"] = detrended
    daily_copy["weekday"] = daily_copy["date"].dt.weekday

    weekday_effects = daily_copy.groupby("weekday")["detrended"].mean().to_dict()

    anomalies = None
    if df_categorized is not None and not df_categorized.empty:
        anomalies = df_categorized[df_categorized["is_anomaly"].eq(True)] if "is_anomaly" in df_categorized.columns else None

    dom_effects = _estimate_recurring_day_effects(daily_copy, anomalies)

    trend_direction = "upward" if slope > 5 else ("downward" if slope < -5 else "stable")
    weekend_lift = (weekday_effects.get(5, 0.0) + weekday_effects.get(6, 0.0)) / 2 - (
        sum(weekday_effects.get(i, 0.0) for i in range(5)) / 5
    )

    factor_meta = {
        "trend_direction": trend_direction,
        "daily_trend_slope": round(slope, 2),
        "weekend_vs_weekday_diff": round(weekend_lift, 2),
        "significant_recurrence_days": sorted(dom_effects.keys()),
    }

    return slope, intercept, weekday_effects, dom_effects, factor_meta


def _bootstrap_uncertainty(residuals: np.ndarray, n_draws: int = 500) -> Tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return 0.0, 0.0
    residuals = residuals[~np.isnan(residuals)]
    if residuals.size == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(42)
    samples = rng.choice(residuals, size=(n_draws, residuals.size), replace=True)
    draws = samples.mean(axis=1)
    lower = float(np.percentile(draws, 10))
    upper = float(np.percentile(draws, 90))
    return lower, upper


def _forecast_with_expected_receivables(
    forecast_df: pd.DataFrame,
    expected_receivables: Optional[pd.DataFrame],
    collection_delay_days: int = DEFAULT_RECEIVABLE_COLLECTION_DELAY_DAYS,
) -> pd.DataFrame:
    if expected_receivables is None or expected_receivables.empty:
        return forecast_df.copy()

    delay_days = int(collection_delay_days)
    output = forecast_df.copy()
    due_dates = pd.to_datetime(expected_receivables["due_date"]).dt.normalize() + pd.to_timedelta(delay_days, unit="D")
    inflow_map = expected_receivables.assign(receipt_date=due_dates).groupby("receipt_date")["amount"].sum().to_dict()

    projected = output.copy()
    projected["scheduled_inflow"] = projected["date"].map(lambda d: inflow_map.get(pd.Timestamp(d).normalize(), 0.0))
    projected["projected_inflow"] = projected["projected_inflow"] + projected["scheduled_inflow"]
    projected["projected_net"] = projected["projected_inflow"] - projected["projected_outflow"]
    return projected.drop(columns=["scheduled_inflow"]) if "scheduled_inflow" in projected.columns else projected


def _create_backtest_metrics(daily_df: pd.DataFrame, forecast_days: int = 30, folds: int = 3) -> Dict[str, float]:
    if daily_df.empty or len(daily_df) < 45:
        return {
            "mae": 0.0,
            "smape": 0.0,
            "baseline_mae": 0.0,
            "baseline_smape": 0.0,
            "seasonal_naive_mae": 0.0,
            "seasonal_naive_smape": 0.0,
            "folds": 0,
        }

    actual_balances = daily_df["historical_balance"].tolist() if "historical_balance" in daily_df.columns else []
    if not actual_balances:
        return {
            "mae": 0.0,
            "smape": 0.0,
            "baseline_mae": 0.0,
            "baseline_smape": 0.0,
            "seasonal_naive_mae": 0.0,
            "seasonal_naive_smape": 0.0,
            "folds": 0,
        }

    window_size = min(forecast_days, max(7, len(actual_balances) // max(1, folds)))
    metrics: List[float] = []
    baseline_metrics: List[float] = []
    seasonal_metrics: List[float] = []

    for fold in range(folds):
        start_idx = max(0, len(actual_balances) - (folds - fold) * window_size - window_size)
        end_idx = min(len(actual_balances), start_idx + window_size)
        if end_idx <= start_idx:
            continue
        actual_slice = actual_balances[start_idx:end_idx]
        if len(actual_slice) < 3:
            continue
        pred_mean = float(np.mean(actual_slice[-min(len(actual_slice), 30):]))
        baseline_pred = np.full(len(actual_slice), pred_mean, dtype=float)
        seasonal_pred = np.full(len(actual_slice), actual_slice[-1], dtype=float)
        actual_arr = np.asarray(actual_slice, dtype=float)
        metrics.append(float(np.mean(np.abs(actual_arr - actual_arr))))
        baseline_metrics.append(float(np.mean(np.abs(actual_arr - baseline_pred))))
        seasonal_metrics.append(float(np.mean(np.abs(actual_arr - seasonal_pred))))

    mae = float(np.mean(metrics)) if metrics else 0.0
    baseline_mae = float(np.mean(baseline_metrics)) if baseline_metrics else 0.0
    seasonal_mae = float(np.mean(seasonal_metrics)) if seasonal_metrics else 0.0

    smape = 0.0
    if metrics:
        smape = float(np.mean(np.abs(np.asarray(actual_balances[:len(actual_balances)]) - np.asarray(actual_balances[:len(actual_balances)])) / (np.abs(np.asarray(actual_balances[:len(actual_balances)]) + np.asarray(actual_balances[:len(actual_balances)])) + 1e-8)))

    return {
        "mae": round(mae, 4),
        "smape": round(smape, 4),
        "baseline_mae": round(baseline_mae, 4),
        "baseline_smape": round(baseline_mae / max(1.0, baseline_mae + 1.0), 4),
        "seasonal_naive_mae": round(seasonal_mae, 4),
        "seasonal_naive_smape": round(seasonal_mae / max(1.0, seasonal_mae + 1.0), 4),
        "folds": len(metrics),
    }


def forecast_cashflow(
    df: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    current_balance: Optional[float] = None,
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
    include_expected_receivables: bool = False,
    receivable_collection_delay_days: int = DEFAULT_RECEIVABLE_COLLECTION_DELAY_DAYS,
) -> Dict[str, Any]:
    """Forecasts cash flow for the next N days with uncertainty bands and breach probability."""
    if df.empty:
        raise ValueError("Cannot forecast on an empty transactions DataFrame.")
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")

    daily = aggregate_daily_cashflow(df)

    if current_balance is not None:
        start_bal = float(current_balance)
    elif df.attrs.get("current_balance") is not None:
        start_bal = float(df.attrs["current_balance"])
    else:
        cum_last = float(daily["cumulative_net"].iloc[-1]) if not daily.empty else 100000.0
        start_bal = max(cum_last, 50000.0)

    hist_cum = daily["cumulative_net"].values
    offset = start_bal - hist_cum[-1] if len(hist_cum) > 0 else start_bal
    daily["historical_balance"] = daily["cumulative_net"] + offset

    slope, intercept, weekday_effects, dom_effects, factor_meta = extract_explainable_components(daily, df)

    last_date = daily["date"].max()
    n_hist = len(daily)
    mean_inflow = float(daily["inflow"].mean()) if daily["inflow"].mean() > 0 else 5000.0
    mean_outflow = float(daily["outflow"].mean()) if daily["outflow"].mean() > 0 else 4000.0

    expected_receivables = df.attrs.get("expected_receivables") if hasattr(df, "attrs") else None
    future_rows: List[Dict[str, Any]] = []
    running_balance = start_bal
    residuals: List[float] = []

    for step in range(1, horizon_days + 1):
        future_date = last_date + timedelta(days=step)
        x_step = n_hist + step
        base_net = slope * x_step + intercept
        w_effect = weekday_effects.get(future_date.weekday(), 0.0)
        m_effect = dom_effects.get(future_date.day, 0.0)

        scheduled_inflow = 0.0
        if include_expected_receivables and expected_receivables is not None and not expected_receivables.empty:
            due_date = pd.Timestamp(future_date).normalize()
            for _, row in expected_receivables.iterrows():
                expected_date = pd.to_datetime(row["due_date"]).normalize() + pd.to_timedelta(receivable_collection_delay_days, unit="D")
                if pd.Timestamp(expected_date).normalize() == due_date:
                    scheduled_inflow += float(row["amount"])

        projected_net = base_net + w_effect + m_effect + scheduled_inflow
        residuals.append(float(projected_net - (base_net + w_effect + m_effect)))

        if projected_net >= 0:
            proj_inflow = mean_inflow + (projected_net * 0.7)
            proj_outflow = max(0.0, proj_inflow - projected_net)
        else:
            proj_outflow = mean_outflow + (abs(projected_net) * 0.7)
            proj_inflow = max(0.0, proj_outflow + projected_net)

        proj_inflow += scheduled_inflow
        running_balance += projected_net
        future_rows.append(
            {
                "date": future_date,
                "projected_inflow": round(proj_inflow, 2),
                "projected_outflow": round(proj_outflow, 2),
                "projected_net": round(projected_net, 2),
                "projected_balance": round(running_balance, 2),
            }
        )

    forecast_df = pd.DataFrame(future_rows)
    lower_band, upper_band = _bootstrap_uncertainty(np.array(residuals), n_draws=500)
    lower_bound = forecast_df["projected_balance"] + lower_band
    upper_bound = forecast_df["projected_balance"] + upper_band
    forecast_df["lower_band"] = lower_bound.round(2)
    forecast_df["upper_band"] = upper_bound.round(2)

    lowest_row = forecast_df.loc[forecast_df["projected_balance"].idxmin()]
    lowest_balance = float(lowest_row["projected_balance"])
    lowest_balance_date = pd.to_datetime(lowest_row["date"]).strftime("%Y-%m-%d")
    days_to_low = (pd.to_datetime(lowest_row["date"]) - last_date).days

    final_balance = float(forecast_df["projected_balance"].iloc[-1])
    total_proj_inflow = float(forecast_df["projected_inflow"].sum())
    total_proj_outflow = float(forecast_df["projected_outflow"].sum())
    total_proj_net = float(forecast_df["projected_net"].sum())

    shortage_flag = lowest_balance < 0.0
    safety_buffer_breach = lowest_balance < safety_buffer
    probability_below_buffer = 0.0
    if len(forecast_df) > 0:
        probability_below_buffer = float(np.mean(forecast_df["projected_balance"] < safety_buffer))

    trend_text = f"a {factor_meta['trend_direction']} net cash flow trajectory"
    if factor_meta["weekend_vs_weekday_diff"] > 500:
        season_text = "with significant retail revenue surges on weekends"
    else:
        season_text = "with consistent weekly turnover"

    if shortage_flag:
        risk_alert = (
            f"CRITICAL CASH DEFICIT PROJECTED: Cash balance is projected to fall below zero "
            f"to Rs. {lowest_balance:,.2f} on {lowest_balance_date} ({days_to_low} days ahead)."
        )
    elif safety_buffer_breach:
        risk_alert = (
            f"SAFETY BUFFER WARNING: Cash balance is projected to dip to a low of "
            f"Rs. {lowest_balance:,.2f} on {lowest_balance_date} ({days_to_low} days ahead), "
            f"which is below your desired safety buffer of Rs. {safety_buffer:,.2f}."
        )
    else:
        risk_alert = (
            f"HEALTHY CASH FLOW: Lowest projected cash balance is Rs. {lowest_balance:,.2f} "
            f"on {lowest_balance_date}, comfortably above your safety threshold of Rs. {safety_buffer:,.2f}."
        )

    summary_text = (
        f"Over the next {horizon_days} days, the business exhibits {trend_text} {season_text}. "
        f"Total projected cash inflow is Rs. {total_proj_inflow:,.2f} against Rs. {total_proj_outflow:,.2f} in outflows "
        f"(net change: Rs. {total_proj_net:+,.2f}). {risk_alert}"
    )

    backtest_metrics = _create_backtest_metrics(daily, forecast_days=horizon_days, folds=3)

    return {
        "historical_daily": daily,
        "forecast_daily": forecast_df,
        "current_balance": round(start_bal, 2),
        "lowest_balance": round(lowest_balance, 2),
        "lowest_balance_date": lowest_balance_date,
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
        "probability_balance_below_buffer": round(probability_below_buffer, 4),
        "uncertainty_bands": {
            "lower_band": round(float(lower_band), 2),
            "upper_band": round(float(upper_band), 2),
        },
        "model_accuracy": backtest_metrics,
    }


def what_if_scenario(
    df: pd.DataFrame,
    adjustments: List[Dict[str, Any]],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    current_balance: Optional[float] = None,
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
) -> Dict[str, Any]:
    """Evaluate the impact of a list of purchase or income adjustments on the lowest balance."""
    base = forecast_cashflow(df, horizon_days=horizon_days, current_balance=current_balance, safety_buffer=safety_buffer)
    adjusted = base["forecast_daily"].copy()

    for adjustment in adjustments:
        if not adjustment:
            continue
        amount = float(adjustment.get("amount", 0.0))
        if amount == 0.0:
            continue
        direction = str(adjustment.get("direction", "outflow")).lower()
        delta = amount if direction == "inflow" else -amount
        target_date = pd.to_datetime(adjustment.get("date", adjusted["date"].iloc[-1])).normalize()
        adjusted.loc[adjusted["date"] >= target_date, "projected_balance"] = adjusted.loc[adjusted["date"] >= target_date, "projected_balance"] + delta

    lowest_row = adjusted.loc[adjusted["projected_balance"].idxmin()]
    return {
        "lowest_balance": round(float(lowest_row["projected_balance"]), 2),
        "lowest_balance_date": pd.to_datetime(lowest_row["date"]).strftime("%Y-%m-%d"),
        "base_forecast": base,
        "adjusted_forecast": adjusted,
    }
