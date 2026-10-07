"""Explainable insight agent for small business affordability questions.

Implements the "Compute-First, Explain-Second" principle:
1. Extracts purchase amount and evaluation window from the question.
2. Computes pre-purchase and post-purchase projected balances using forecaster output.
3. Determines deterministic verdict (safe / risky / not recommended).
4. Invokes OpenRouter LLM (or fallback template) ONLY to explain the computed figures,
   strictly forbidding the LLM from inventing or altering numbers.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.utils.config import DEFAULT_SAFETY_BUFFER
from src.utils.llm import call_claude

logger = logging.getLogger(__name__)


def parse_affordability_question(question: str) -> Tuple[Optional[float], int, str]:
    """Extracts purchase amount, evaluation window (days), and item description from text.

    Args:
        question: User's question string (e.g. "Can I afford to purchase ₹80,000 inventory next month?")

    Returns:
        (extracted_amount, window_days, item_name)
    """
    q_lower = question.lower()

    # 1. Extract Window in Days
    window_days = 30  # default
    if "next week" in q_lower or "7 days" in q_lower or "one week" in q_lower:
        window_days = 7
    elif "two weeks" in q_lower or "14 days" in q_lower or "fortnight" in q_lower:
        window_days = 14
    elif "next 60 days" in q_lower or "60 days" in q_lower or "two months" in q_lower or "2 months" in q_lower:
        window_days = 60
    elif "next quarter" in q_lower or "90 days" in q_lower or "3 months" in q_lower or "three months" in q_lower:
        window_days = 90
    elif "next month" in q_lower or "30 days" in q_lower or "1 month" in q_lower or "one month" in q_lower:
        window_days = 30

    # 2. Extract Amount
    # Handles: ₹80,000, Rs. 80,000, 80000, 80k, 1.5 lakh, 80000 rs
    extracted_amount = None

    # Check for lakh notation (e.g. 1.5 lakh, 2 lakhs)
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)s?", q_lower)
    if lakh_match:
        extracted_amount = float(lakh_match.group(1)) * 100000.0

    # Check for 'k' notation (e.g. 80k, 50k)
    if extracted_amount is None:
        k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", q_lower)
        if k_match:
            extracted_amount = float(k_match.group(1)) * 1000.0

    # Standard currency pattern (e.g. ₹80,000, Rs 80,000, 80,000)
    if extracted_amount is None:
        curr_match = re.search(r"(?:[₹\$]|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}(?:\.\d+)?)", question, re.IGNORECASE)
        if curr_match:
            clean_str = curr_match.group(1).replace(",", "")
            try:
                extracted_amount = float(clean_str)
            except ValueError:
                pass

    # 3. Extract Item Name
    item_name = "requested purchase"
    item_patterns = [
        r"(?:purchase|buy|spend on|order|invest in|afford)\s+(?:of\s+)?(?:[₹\$]|rs\.?|inr)?\s*[\d,k\.]+\s*(?:of\s+)?([a-zA-Z\s]{3,30}?)(?:\s+(?:next|in|for|this|\?|$))",
        r"(?:hire|employ)\s+(?:a\s+)?([a-zA-Z\s]{3,30}?)(?:\s+(?:for|next|in|\?|$))",
    ]
    for pat in item_patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if candidate and len(candidate) > 2 and candidate not in ["next", "month", "days", "quarter"]:
                item_name = candidate
                break

    return extracted_amount, window_days, item_name


def format_template_explanation(
    question: str,
    verdict: str,
    item_name: str,
    purchase_amount: float,
    window_days: int,
    current_balance: float,
    pre_end_bal: float,
    pre_min_bal: float,
    pre_min_date: str,
    post_end_bal: float,
    post_min_bal: float,
    safety_buffer: float,
    buffer_delta: float,
    anomalies: List[Dict[str, any]],
) -> str:
    """Generates a deterministic natural-language explanation when LLM is offline."""
    if verdict == "safe":
        status_line = (
            f"**Yes, you can comfortably afford this purchase.** Your business cash flow is projected to "
            f"remain well-buffered throughout the next {window_days} days."
        )
        buffer_assessment = (
            f"Even after spending Rs. {purchase_amount:,.2f}, your lowest projected cash point will be "
            f"Rs. {post_min_bal:,.2f} on {pre_min_date}, which maintains a cushion of Rs. {buffer_delta:,.2f} "
            f"above your safety threshold of Rs. {safety_buffer:,.2f}."
        )
        recommendation = "You have adequate working capital headroom to proceed with this purchase without risking upcoming commitments."

    elif verdict == "risky":
        status_line = (
            f"**Proceed with caution: this purchase is RISKY.** While your cash balance remains positive, "
            f"it will dip into your required safety buffer."
        )
        buffer_assessment = (
            f"After spending Rs. {purchase_amount:,.2f}, your cash balance is projected to reach an ending balance of "
            f"Rs. {post_end_bal:,.2f}, but will dip to a low point of Rs. {post_min_bal:,.2f} on {pre_min_date}. "
            f"This is Rs. {abs(buffer_delta):,.2f} below your desired safety reserve of Rs. {safety_buffer:,.2f}."
        )
        recommendation = (
            "If you must proceed, consider staggering vendor payment terms, securing early invoice collections, "
            "or splitting the purchase into two tranches."
        )

    else:  # not recommended
        status_line = (
            f"**NOT RECOMMENDED: This purchase will trigger a cash deficit.** "
            f"Your operations cannot absorb this outflow without overdraft or external funding."
        )
        buffer_assessment = (
            f"Deducting Rs. {purchase_amount:,.2f} causes your projected lowest balance to fall into a negative "
            f"balance of Rs. {post_min_bal:,.2f} on {pre_min_date} (pre-purchase low was Rs. {pre_min_bal:,.2f}). "
            f"Your projected ending balance at day {window_days} would be Rs. {post_end_bal:,.2f}."
        )
        recommendation = "Hold off on this purchase until cash collections improve, or arrange bridge trade credit."

    anomaly_note = ""
    if anomalies:
        top_a = anomalies[0]
        anomaly_note = f"\n\n*Note on Recent Volatility:* Recent unusual expenses (such as the Rs. {top_a['amount']:,.2f} '{top_a['category']}' charge on {top_a['date']}) have placed additional strain on working capital."

    return (
        f"### Verdict: {verdict.upper()}\n\n"
        f"{status_line}\n\n"
        f"**Financial Breakdown:**\n"
        f"- **Current Available Balance:** Rs. {current_balance:,.2f}\n"
        f"- **Purchase Cost:** Rs. {purchase_amount:,.2f} ({item_name})\n"
        f"- **Projected Baseline Balance (without purchase):** Rs. {pre_end_bal:,.2f} (lowest: Rs. {pre_min_bal:,.2f} on {pre_min_date})\n"
        f"- **Projected Balance (after purchase):** Rs. {post_end_bal:,.2f} (lowest: Rs. {post_min_bal:,.2f})\n"
        f"- **Safety Buffer Delta:** Rs. {buffer_delta:+,.2f} (vs Rs. {safety_buffer:,.2f} safety target)\n\n"
        f"{buffer_assessment}\n\n"
        f"**Advisor Action:** {recommendation}{anomaly_note}"
    )


def answer_affordability_question(
    question: str,
    amount: Optional[float] = None,
    forecast_result: Optional[Dict[str, any]] = None,
    current_balance: Optional[float] = None,
    anomalies: Optional[List[Dict[str, any]]] = None,
    category_summary: Optional[Dict[str, float]] = None,
    safety_buffer: Optional[float] = None,
) -> Dict[str, any]:
    """Evaluates whether a small business can afford an expenditure using compute-first logic.

    Contract:
      1. Determines evaluation time window from question.
      2. Extracts amount if not explicitly passed.
      3. Computes pre-purchase and post-purchase balances and minimums using forecast.
      4. Decides strict verdict: 'safe', 'risky', or 'not recommended'.
      5. Calls LLM (or fallback template) ONLY to phrase an explanation of these exact numbers.
    """
    if not question or not str(question).strip():
        raise ValueError("Question cannot be empty.")

    # 1. Parse Question parameters
    parsed_amt, window_days, item_name = parse_affordability_question(question)
    purchase_amount = float(amount) if amount is not None else (parsed_amt or 0.0)

    if purchase_amount <= 0:
        raise ValueError(
            "Could not determine a valid purchase amount from the question. "
            "Please specify an amount like '₹80,000' or pass amount explicitly."
        )
    if forecast_result is None or "forecast_daily" not in forecast_result:
        raise ValueError("A cash flow forecast is required to evaluate affordability.")
    if forecast_result["forecast_daily"].empty:
        raise ValueError("The cash flow forecast has no projected dates to evaluate.")
    if len(forecast_result["forecast_daily"]) < window_days:
        raise ValueError(
            f"The forecast must cover the full {window_days}-day affordability window."
        )

    # Resolve safety buffer
    resolved_buffer = float(safety_buffer) if safety_buffer is not None else (
        forecast_result.get("safety_buffer", DEFAULT_SAFETY_BUFFER) if forecast_result else DEFAULT_SAFETY_BUFFER
    )

    # Resolve current balance
    resolved_current_balance = float(current_balance) if current_balance is not None else (
        forecast_result.get("current_balance", 100000.0) if forecast_result else 100000.0
    )

    # 2. Extract Forecast Timeline Slice
    fdf = forecast_result["forecast_daily"].copy()
    # Slice only after proving the supplied forecast covers the requested window.
    slice_df = fdf.iloc[:window_days]
    pre_end_bal = float(slice_df["projected_balance"].iloc[-1])
    pre_min_row = slice_df.loc[slice_df["projected_balance"].idxmin()]
    pre_min_bal = float(pre_min_row["projected_balance"])
    pre_min_date = pd.to_datetime(pre_min_row["date"]).strftime("%Y-%m-%d")

    # 3. Compute Post-Purchase Balance Numbers
    post_end_bal = pre_end_bal - purchase_amount
    post_min_bal = pre_min_bal - purchase_amount
    buffer_delta = post_min_bal - resolved_buffer

    # 4. Deterministic Verdict Assignment
    #   safe: post_min_bal >= safety_buffer
    #   risky: 0 <= post_min_bal < safety_buffer
    #   not recommended: post_min_bal < 0
    if post_min_bal >= resolved_buffer:
        verdict = "safe"
    elif post_min_bal >= 0:
        verdict = "risky"
    else:
        verdict = "not recommended"

    anomalies_list = anomalies or []

    # 5. Call LLM to phrase natural-language explanation using ONLY computed numbers
    llm_explanation = None
    system_prompt = (
        "You are an explainable small business financial advisor. You are provided with "
        "pre-computed cash flow numbers and an already-decided verdict. "
        "Your role is ONLY to explain this decision to the business owner in clear, actionable terms. "
        "CRITICAL RULE: You MUST use ONLY the exact numbers provided below. "
        "Do NOT invent, recalculate, round differently, or introduce any financial numbers of your own."
    )

    anomaly_context = ""
    if anomalies_list:
        top_anom = anomalies_list[:2]
        anomaly_context = "\nRecent Notable Anomalies:\n" + "\n".join(
            f"- {a['date']}: Rs. {a['amount']:,.2f} in '{a['category']}' ({a['reason']})" for a in top_anom
        )

    user_message = f"""User Question: "{question}"

COMPUTED FINANCIAL AUDIT NUMBERS:
- Evaluated Window: Next {window_days} days
- Purchase Item: {item_name}
- Requested Outflow Amount: Rs. {purchase_amount:,.2f}
- Current Starting Cash Balance: Rs. {resolved_current_balance:,.2f}
- Baseline Projected Ending Balance (before purchase): Rs. {pre_end_bal:,.2f}
- Baseline Projected Lowest Cash Point (before purchase): Rs. {pre_min_bal:,.2f} on {pre_min_date}
- Post-Purchase Projected Ending Balance: Rs. {post_end_bal:,.2f}
- Post-Purchase Projected Lowest Cash Point: Rs. {post_min_bal:,.2f}
- Minimum Safety Buffer Requirement: Rs. {resolved_buffer:,.2f}
- Safety Cushion Delta: Rs. {buffer_delta:+,.2f}
- DETERMINISTIC VERDICT: {verdict.upper()}
{anomaly_context}

INSTRUCTIONS FOR YOUR RESPONSE:
1. State the verdict clearly ({verdict.upper()}).
2. Explain the numeric impact clearly, comparing the pre-purchase low and post-purchase low to the safety buffer.
3. Provide 2-3 specific, pragmatic advisor tips based on this exact verdict.
4. Stick STRICTLY to the exact numbers above.
"""

    try:
        raw_llm = call_claude(system_prompt=system_prompt, user_message=user_message, max_tokens=800)
        if raw_llm and raw_llm.strip() and "Unable to interpret query" not in raw_llm:
            llm_explanation = raw_llm.strip()
    except Exception as e:
        logger.warning(f"Insight LLM explanation call failed: {e}")

    # If LLM didn't respond (offline / no API key / failure), use template explanation
    if not llm_explanation:
        llm_explanation = format_template_explanation(
            question=question,
            verdict=verdict,
            item_name=item_name,
            purchase_amount=purchase_amount,
            window_days=window_days,
            current_balance=resolved_current_balance,
            pre_end_bal=pre_end_bal,
            pre_min_bal=pre_min_bal,
            pre_min_date=pre_min_date,
            post_end_bal=post_end_bal,
            post_min_bal=post_min_bal,
            safety_buffer=resolved_buffer,
            buffer_delta=buffer_delta,
            anomalies=anomalies_list,
        )

    return {
        "question": question,
        "item_name": item_name,
        "purchase_amount": purchase_amount,
        "window_days": window_days,
        "current_balance": resolved_current_balance,
        "pre_purchase_end_balance": round(pre_end_bal, 2),
        "pre_purchase_min_balance": round(pre_min_bal, 2),
        "pre_purchase_min_date": pre_min_date,
        "post_purchase_end_balance": round(post_end_bal, 2),
        "post_purchase_min_balance": round(post_min_bal, 2),
        "safety_buffer": resolved_buffer,
        "buffer_delta": round(buffer_delta, 2),
        "verdict": verdict,  # "safe", "risky", or "not recommended"
        "explanation": llm_explanation,
    }
