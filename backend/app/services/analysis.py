from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.anomaly.anomaly_detector import detect_anomalies
from src.forecasting.forecaster import forecast_cashflow, what_if_scenario
from src.ingestion.loader import load_files
from src.insight.insight_agent import answer_affordability_question
from src.synthetic_data.generate_sample_data import generate_all_sample_data


def _as_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "current_balance": 0.0,
            "inflow_30d": 0.0,
            "outflow_30d": 0.0,
            "net_30d": 0.0,
            "burn_rate": 0.0,
            "runway_days": 0.0,
            "category_breakdown": {},
            "monthly_series": [],
            "risk_status": "SAFE",
        }

    balance = float(df.attrs.get("current_balance", 0.0))
    recent = df[df["date"] >= (df["date"].max() - pd.Timedelta(days=30))].copy()
    inflow = float(recent[recent["direction"] == "inflow"]["amount"].sum())
    outflow = float(recent[recent["direction"] == "outflow"]["amount"].sum())
    net = inflow - outflow
    burn_rate = outflow / 30.0 if outflow > 0 else 0.0
    runway_days = (balance / burn_rate) if burn_rate > 0 else 0.0
    breakdown = recent.groupby("category", dropna=False)["amount"].sum().sort_values(ascending=False).to_dict() if not recent.empty else {}
    month_series = []
    for month, month_df in df.assign(month=df["date"].dt.to_period("M").astype(str)).groupby("month"):
        month_inflow = float(month_df[month_df["direction"] == "inflow"]["amount"].sum())
        month_outflow = float(month_df[month_df["direction"] == "outflow"]["amount"].sum())
        month_series.append({
            "month": str(month),
            "inflow": month_inflow,
            "outflow": month_outflow,
            "net": month_inflow - month_outflow,
        })
    risk_status = "RISKY" if balance < 50000 else "SAFE"
    return {
        "current_balance": round(balance, 2),
        "inflow_30d": round(inflow, 2),
        "outflow_30d": round(outflow, 2),
        "net_30d": round(net, 2),
        "burn_rate": round(burn_rate, 2),
        "runway_days": round(runway_days, 2),
        "category_breakdown": {k: round(float(v), 2) for k, v in breakdown.items()},
        "monthly_series": month_series[-6:],
        "risk_status": risk_status,
    }


def build_forecast(df: pd.DataFrame, horizon: int = 30, safety_buffer: float = 50000.0, include_receivables: bool = False) -> Dict[str, Any]:
    forecast = forecast_cashflow(
        df,
        horizon_days=int(horizon),
        current_balance=float(df.attrs.get("current_balance", 0.0)),
        safety_buffer=float(safety_buffer),
        include_expected_receivables=bool(include_receivables),
    )
    return {
        "horizon_days": forecast["horizon_days"],
        "current_balance": forecast["current_balance"],
        "lowest_balance": forecast["lowest_balance"],
        "lowest_balance_date": forecast["lowest_balance_date"],
        "final_projected_balance": forecast["final_projected_balance"],
        "probability_balance_below_buffer": forecast["probability_balance_below_buffer"],
        "history": forecast["historical_daily"].to_dict(orient="records"),
        "forecast": forecast["forecast_daily"].to_dict(orient="records"),
        "explainable_factors": forecast["explainable_factors"],
        "model_accuracy": forecast["model_accuracy"],
        "summary_text": forecast["summary_text"],
    }


def build_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return detect_anomalies(df)


def build_ask_result(df: pd.DataFrame, question: str, amount: float | None = None, safety_buffer: float | None = None) -> Dict[str, Any]:
    resolved_buffer = float(safety_buffer or 50000.0)
    forecast = forecast_cashflow(
        df,
        horizon_days=90,
        current_balance=float(df.attrs.get("current_balance", 0.0)),
        safety_buffer=resolved_buffer,
        include_expected_receivables=True,
    )
    result = answer_affordability_question(
        question=question,
        amount=amount,
        current_balance=float(df.attrs.get("current_balance", 0.0)),
        safety_buffer=resolved_buffer,
        forecast_result=forecast,
        anomalies=build_anomalies(df),
    )
    return {
        "verdict": result["verdict"].upper(),
        "numbers": {
            "current_balance": float(result["current_balance"]),
            "purchase_amount": float(result["purchase_amount"]),
            "safety_buffer": float(result["safety_buffer"]),
            "post_purchase_min_balance": float(result["post_purchase_min_balance"]),
            "buffer_delta": float(result["buffer_delta"]),
        },
        "explanation": result["explanation"],
        "explanation_source": result["explanation_source"],
        "guardrail_passed": bool(result["guardrail_passed"]),
    }


def build_scenario(df: pd.DataFrame, adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = what_if_scenario(
        df,
        adjustments=adjustments,
        horizon_days=90,
        current_balance=float(df.attrs.get("current_balance", 0.0)),
        safety_buffer=50000.0,
    )
    return {
        "new_lowest_balance": result["lowest_balance"],
        "lowest_balance_date": result["lowest_balance_date"],
        "base_lowest_balance": 0.0,
        "adjusted_forecast": result["adjusted_forecast"].to_dict(orient="records"),
    }


def generate_sample_data() -> list[dict]:
    output_dir = Path(__file__).resolve().parents[2] / "data" / "sample"
    generate_all_sample_data(output_dir)
    data_files = [
        str(output_dir / "sales.csv"),
        str(output_dir / "expenses.csv"),
        str(output_dir / "invoices.csv"),
        str(output_dir / "bank_statement.csv"),
    ]
    df = load_files(data_files)
    return df.to_dict(orient="records")


def export_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def export_pdf(df: pd.DataFrame) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Cashflow & Risk Assistant Risk Report", styles["Title"]), Spacer(1, 12)]
    story.extend([
        Paragraph(f"Current balance: ₹{float(df.attrs.get('current_balance', 0.0)):,}", styles["BodyText"]),
        Paragraph(f"Rows: {len(df)}", styles["BodyText"]),
        Paragraph(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}", styles["BodyText"]),
    ])
    if not df.empty:
        rows = [["Date", "Description", "Amount", "Direction", "Category"]]
        for row in df.head(8).to_dict(orient="records"):
            rows.append([
                row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                row["description"],
                f"₹{float(row['amount']):,.2f}",
                row["direction"],
                row.get("category") or "-",
            ])
        table = Table(rows)
        table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, '#cccccc')]))
        story.extend([Spacer(1, 12), table])
    doc.build(story)
    return buffer.getvalue()
