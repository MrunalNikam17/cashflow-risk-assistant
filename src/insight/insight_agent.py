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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import DEFAULT_SAFETY_BUFFER
from src.utils.llm import call_llm_with_fallback

logger = logging.getLogger(__name__)


def _extract_window_days(question: str) -> int:
    q_lower = question.lower()
    if re.search(r"\b(?:next\s+)?(?:week|7\s+days|one\s+week)\b", q_lower):
        return 7
    if re.search(r"\b(?:two\s+weeks|14\s+days|fortnight)\b", q_lower):
        return 14
    if re.search(r"\b(?:next\s+)?(?:month|30\s+days|1\s+month|one\s+month)\b", q_lower):
        return 30
    if re.search(r"\b(?:next\s+)?(?:60\s+days|2\s+months|two\s+months)\b", q_lower):
        return 60
    if re.search(r"\b(?:next\s+)?(?:quarter|90\s+days|3\s+months|three\s+months)\b", q_lower):
        return 90
    if re.search(r"\b(?:6\s+weeks|six\s+weeks|42\s+days)\b", q_lower):
        return 42
    if re.search(r"\b(?:45\s+days|forty-five\s+days)\b", q_lower):
        return 45
    if re.search(r"\b\d+\s+days\b|\b\d+\s+weeks\b|\b\d+\s+months\b", q_lower):
        raw_match = re.search(r"(\d+)\s*(days?|weeks?|months?)", q_lower)
        if raw_match:
            value = int(raw_match.group(1))
            unit = raw_match.group(2)
            if unit.startswith("day"):
                return max(1, value)
            if unit.startswith("week"):
                return max(1, value * 7)
            if unit.startswith("month"):
                return max(1, value * 30)
    return 30


def _extract_amount(question: str) -> Optional[float]:
    q_lower = question.lower()
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|lakhs|lacs)s?", q_lower)
    if lakh_match:
        return float(lakh_match.group(1)) * 100000.0
    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)s?", q_lower)
    if crore_match:
        return float(crore_match.group(1)) * 10000000.0
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", q_lower)
    if k_match:
        return float(k_match.group(1)) * 1000.0
    curr_match = re.search(
        r"(?:[₹\$]|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}(?:\.\d+)?)",
        question,
        re.IGNORECASE,
    )
    if curr_match:
        try:
            return float(curr_match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def parse_affordability_question(question: str) -> Tuple[Optional[float], int, str]:
    """Extracts purchase amount, evaluation window (days), and item description from text."""
    window_days = _extract_window_days(question)
    extracted_amount = _extract_amount(question)
    item_name = "requested purchase"
    item_patterns = [
        r"(?:purchase|buy|spend on|order|invest in|afford)\s+(?:of\s+)?(?:[₹\$]|rs\.?|inr)?\s*[\d,k\.]+\s*(?:of\s+)?([a-zA-Z\s]{3,30}?)(?:\s+(?:next|in|for|this|\?|$))",
        r"(?:hire|employ)\s+(?:a\s+)?([a-zA-Z\s]{3,30}?)(?:\s+(?:for|next|in|\?|$))",
    ]
    for pat in item_patterns:
        match = re.search(pat, question, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate and len(candidate) > 2 and candidate not in ["next", "month", "days", "quarter"]:
                item_name = candidate
                break
    return extracted_amount, window_days, item_name


def _parse_purchase_day(question: str) -> int:
    q_lower = question.lower()
    if re.search(r"\b(?:on|by|at|paid\s+on)\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+day)?\b", q_lower):
        match = re.search(r"\b(?:on|by|at|paid\s+on)\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+day)?\b", q_lower)
        if match:
            return max(1, min(31, int(match.group(1))))
    if re.search(r"\bday\s+(\d{1,2})\b", q_lower):
        match = re.search(r"\bday\s+(\d{1,2})\b", q_lower)
        if match:
            return max(1, min(31, int(match.group(1))))
    return 0


def _parse_recurring_costs(question: str) -> List[Dict[str, Any]]:
    matches = []
    for pattern in [
        r"([a-zA-Z\s]+?)\s+at\s+(?:₹|rs\.?|inr)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:k|thousand)?\s*(?:per\s+)?(day|week|month)\s+for\s+(\d+)\s*(day|days|week|weeks|month|months)",
        r"(?:hire|add)\s+[a-zA-Z\s]+?\s+(?:₹|rs\.?|inr)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:k|thousand)?\s*(?:per\s+)?(day|week|month)\s+for\s+(\d+)\s*(day|days|week|weeks|month|months)",
    ]:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            amount = float(match.group(2).replace(",", ""))
            unit = match.group(3).lower()
            duration = match.group(4).lower()
            multiplier = 1
            if unit.startswith("day"):
                multiplier = 1
            elif unit.startswith("week"):
                multiplier = 7
            elif unit.startswith("month"):
                multiplier = 30
            count = max(1, int(match.group(5))) if match.groups() >= 5 else 1
            matches.append({
                "label": (match.group(1) or "recurring cost").strip(),
                "amount": amount * (1000.0 if "k" in match.group(0).lower() else 1.0),
                "days": multiplier * count,
                "day": _parse_purchase_day(question),
            })
    return matches


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
    anomalies: List[Dict[str, Any]],
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

    else:
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


def _extract_numbers(text: str) -> List[float]:
    text = text or ""
    values: List[float] = []
    for token in re.findall(
        r"(?:₹|rs\.?|inr|\$)?\s*-?\d[\d,]*(?:\.\d+)?(?:\s*(?:k|lakhs?|lac|crores?|cr))?",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = token.replace("₹", "").replace("$", "").replace("RS", "").replace("INR", "").strip()
        suffix = ""
        if re.search(r"(?:k|lakhs?|lac|crores?|cr)$", cleaned, re.IGNORECASE):
            suffix = re.search(r"(?:k|lakhs?|lac|crores?|cr)$", cleaned, re.IGNORECASE).group(0).lower()
            cleaned = re.sub(r"(?:k|lakhs?|lac|crores?|cr)$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace(",", "")
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if suffix.startswith("k"):
            number *= 1000.0
        elif suffix.startswith("l"):
            number *= 100000.0
        elif suffix.startswith("c"):
            number *= 10000000.0
        values.append(number)
    return values


def _guard_llm_response(response: str, allowed_numbers: List[float], tolerance: float = 0.02) -> bool:
    allowed = [float(x) for x in allowed_numbers]
    found = _extract_numbers(response)
    for value in found:
        if not any(abs(value - candidate) <= max(abs(candidate) * tolerance, 1.0) for candidate in allowed):
            return False
    return True


def answer_affordability_question(
    question: str,
    amount: Optional[float] = None,
    forecast_result: Optional[Dict[str, Any]] = None,
    current_balance: Optional[float] = None,
    anomalies: Optional[List[Dict[str, Any]]] = None,
    category_summary: Optional[Dict[str, float]] = None,
    safety_buffer: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluates whether a small business can afford an expenditure using compute-first logic."""
    if not question or not str(question).strip():
        raise ValueError("Question cannot be empty.")

    parsed_amt, window_days, item_name = parse_affordability_question(question)
    purchase_amount = float(amount) if amount is not None else (parsed_amt or 0.0)

    if purchase_amount <= 0:
        raise ValueError(
            "Could not determine a valid purchase amount from the question. "
            "Please specify an amount like '₹80,000' or pass amount explicitly."
        )
    if forecast_result is None or "forecast_daily" not in forecast_result:
        raise ValueError("A cash flow forecast is required to evaluate affordability.")

    fdf = forecast_result["forecast_daily"].copy()
    if fdf.empty:
        raise ValueError("The cash flow forecast has no projected dates to evaluate.")

    if len(fdf) < window_days:
        horizon_days = forecast_result.get("horizon_days")
        if horizon_days is None or int(horizon_days) < window_days:
            raise ValueError(f"The forecast must cover the full {window_days}-day affordability window.")
        last_date = pd.to_datetime(fdf["date"].iloc[-1])
        last_balance = float(fdf["projected_balance"].iloc[-1])
        pad_rows = []
        while len(fdf) < window_days:
            last_date += pd.Timedelta(days=1)
            pad_rows.append({"date": last_date, "projected_balance": last_balance})
            fdf = pd.concat([fdf, pd.DataFrame(pad_rows)], ignore_index=True)

    slice_df = fdf.iloc[:window_days].copy()
    if slice_df.empty:
        raise ValueError(f"The forecast must cover the full {window_days}-day affordability window.")

    resolved_buffer = float(safety_buffer) if safety_buffer is not None else (
        forecast_result.get("safety_buffer", DEFAULT_SAFETY_BUFFER) if forecast_result else DEFAULT_SAFETY_BUFFER
    )
    resolved_current_balance = float(current_balance) if current_balance is not None else (
        forecast_result.get("current_balance", 100000.0) if forecast_result else 100000.0
    )

    pre_end_bal = float(slice_df["projected_balance"].iloc[-1])
    pre_min_row = slice_df.loc[slice_df["projected_balance"].idxmin()]
    pre_min_bal = float(pre_min_row["projected_balance"])
    pre_min_date = pd.to_datetime(pre_min_row["date"]).strftime("%Y-%m-%d")

    purchase_day = _parse_purchase_day(question)
    if purchase_day:
        purchase_index = max(0, min(purchase_day - 1, len(slice_df) - 1))
        post_balances = slice_df["projected_balance"].copy()
        post_balances.iloc[purchase_index:] = post_balances.iloc[purchase_index:] - purchase_amount
        post_end_bal = float(post_balances.iloc[-1])
        post_min_row = post_balances.idxmin()
        post_min_bal = float(post_balances.iloc[post_balances.idxmin()])
        post_min_date = pd.to_datetime(slice_df.iloc[post_balances.idxmin()]["date"]).strftime("%Y-%m-%d")
    else:
        post_end_bal = pre_end_bal - purchase_amount
        post_min_bal = pre_min_bal - purchase_amount
        post_min_date = pre_min_date

    buffer_delta = post_min_bal - resolved_buffer

    if post_min_bal >= resolved_buffer:
        verdict = "safe"
    elif post_min_bal >= 0:
        verdict = "risky"
    else:
        verdict = "not recommended"

    anomalies_list = anomalies or []
    llm_explanation = None
    guardrail_passed = True
    explanation_source = "template"
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
3. Provide 2-3 specific pragmatic advisor tips based on this exact verdict.
4. Stick STRICTLY to the exact numbers above.
"""

    allowed_numbers = [
        float(resolved_current_balance),
        float(purchase_amount),
        float(pre_end_bal),
        float(pre_min_bal),
        float(post_end_bal),
        float(post_min_bal),
        float(resolved_buffer),
        float(buffer_delta),
    ]

    try:
        raw_llm = call_llm_with_fallback(system_prompt=system_prompt, user_message=user_message, max_tokens=800)
        if raw_llm and raw_llm.strip() and "Unable to interpret query" not in raw_llm:
            if _guard_llm_response(raw_llm, allowed_numbers):
                llm_explanation = raw_llm.strip()
                explanation_source = "llm"
            else:
                guardrail_passed = False
                logger.warning("LLM response contained unauthorized numeric values; using deterministic template.")
    except Exception as exc:
        logger.warning("Insight LLM explanation call failed: %s", exc)

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
        "verdict": verdict,
        "explanation": llm_explanation,
        "explanation_source": explanation_source,
        "guardrail_passed": guardrail_passed,
        "purchase_day": purchase_day,
    }
