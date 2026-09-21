"""Classifies transactions into standardized business categories.

Uses OpenRouter LLM (via src/utils/llm.py) with batched few-shot prompts,
validated against allowed categories, with deterministic rule-based fallback
when offline or API key is absent.
"""

import json
import logging
import re
from typing import List, Dict, Optional
import pandas as pd

from src.utils.config import CATEGORIES, DEFAULT_BATCH_SIZE
from src.utils.llm import call_llm

logger = logging.getLogger(__name__)

# Case-insensitive category lookup map for fast validation
CATEGORY_LOOKUP = {c.lower(): c for c in CATEGORIES}

# Rule-based fallback keywords for offline use or LLM fallback
KEYWORD_RULES = [
    ("Salary", [r"salary", r"salaries", r"payroll", r"wages", r"stipend", r"hrops", r"staff", r"employee"]),
    ("Rent", [r"rent", r"lease", r"landlord", r"storefront", r"tenancy", r"premises"]),
    ("Loan Repayment", [r"loan", r"emi", r"installment", r"repayment", r"mortgage", r"debt"]),
    ("Taxes", [r"gst", r"tax", r"challan", r"tds", r"vat", r"customs", r"treasury"]),
    ("Utilities", [r"electricity", r"power", r"utility", r"water", r"internet", r"broadband", r"meter charge", r"telecom"]),
    ("Inventory Purchase", [r"inventory", r"stock", r"wholesale", r"raw material", r"restock", r"supplies wholesale"]),
    ("Marketing", [r"marketing", r"advertising", r"campaign", r"ads", r"flyer", r"meta ads", r"google ads", r"promotional"]),
    ("Equipment", [r"equipment", r"repair", r"compressor", r"hvac", r"maintenance", r"chiller", r"hardware", r"machinery"]),
    ("Sales Revenue", [r"sales", r"sale", r"pos", r"settlement", r"merchant", r"counter upi", r"client inward", r"remittance", r"invoice #"]),
]


def rule_based_fallback(description: str, direction: str = "outflow") -> str:
    """Deterministic keyword-based classification used when LLM is offline or unavailable."""
    desc_lower = str(description).lower()

    if direction == "inflow" and any(k in desc_lower for k in ["sale", "pos", "settlement", "merchant", "invoice", "remittance", "inward"]):
        return "Sales Revenue"

    for cat_name, patterns in KEYWORD_RULES:
        for pattern in patterns:
            if re.search(r"\b" + re.escape(pattern) + r"\b", desc_lower) or pattern in desc_lower:
                return cat_name

    if direction == "inflow":
        return "Sales Revenue"

    return "Other"


def clean_json_response(text: str) -> str:
    """Strips markdown code blocks from LLM output to extract pure JSON."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def build_categorization_prompt(items: List[Dict[str, any]]) -> str:
    """Builds a batched few-shot prompt for the OpenRouter LLM."""
    categories_str = ", ".join(f'"{c}"' for c in CATEGORIES)
    items_json = json.dumps(items, indent=2)

    prompt = f"""You are a financial transaction classification expert for small business accounting.

Allowed Categories:
[{categories_str}]

Classify each transaction description below into EXACTLY ONE of the allowed categories above.
If none fit clearly, use "Other".

Few-shot examples:
- "Monthly Storefront Retail Lease Rent" -> "Rent"
- "Staff Salaries & Wages Payroll Disbursal" -> "Salary"
- "Bulk Wholesale Stock Restock & Raw Material Order" -> "Inventory Purchase"
- "Commercial Electricity & Water Utility Bill" -> "Utilities"
- "Daily Point-of-Sale Card Settlements & Merchant Batch Inflow" -> "Sales Revenue"
- "Equipment Term Loan Monthly EMI Installment" -> "Loan Repayment"
- "Digital Ad Campaign & Promotional Local Print Flyers" -> "Marketing"
- "Quarterly GST Direct Challan Tax Payment Filing" -> "Taxes"
- "Emergency Industrial Chiller Compressor Replacement" -> "Equipment"

Transactions to classify:
{items_json}

INSTRUCTIONS:
Output ONLY a valid JSON array of objects with keys "index" and "category".
No markdown explanations, no text outside the JSON array.
Example response:
[
  {{"index": 0, "category": "Rent"}},
  {{"index": 1, "category": "Salary"}}
]
"""
    return prompt


def classify_batch_llm(batch: List[Dict[str, any]]) -> Dict[int, str]:
    """Classifies a batch of transactions using OpenRouter LLM with fallback."""
    results = {}
    if not batch:
        return results

    prompt = build_categorization_prompt(batch)
    raw_response = call_llm(prompt, max_tokens=1500)

    if raw_response and raw_response.strip():
        cleaned = clean_json_response(raw_response)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "index" in item and "category" in item:
                        idx = int(item["index"])
                        cat = str(item["category"]).strip()
                        # Validate category
                        matched_cat = CATEGORY_LOOKUP.get(cat.lower())
                        if matched_cat:
                            results[idx] = matched_cat
                        else:
                            logger.warning(f"LLM returned invalid category '{cat}' for index {idx}. Defaulting to 'Other'.")
                            results[idx] = "Other"
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}. Raw response: {raw_response[:200]}")

    # For any items in batch that LLM missed or if LLM failed, fallback to rules
    for item in batch:
        idx = item["index"]
        if idx not in results:
            fallback_cat = rule_based_fallback(item.get("description", ""), item.get("direction", "outflow"))
            results[idx] = fallback_cat

    return results


def categorize_transactions(
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Categorizes transactions in a DataFrame.

    Preserves existing valid categories if already set. Batches uncategorized
    transactions through OpenRouter LLM, falling back to rule-based classification
    when needed.

    Args:
        df: Input DataFrame conforming to unified transaction schema.
        batch_size: Number of transactions per LLM call.
        use_cache: If True, caches categorization by (description, direction).

    Returns:
        DataFrame with the 'category' column fully populated.
    """
    if df.empty:
        return df.copy()

    df_out = df.copy()

    # Identify rows needing categorization
    needs_cat = df_out["category"].isna() | (df_out["category"] == "") | (~df_out["category"].isin(CATEGORIES))
    indices_to_cat = df_out[needs_cat].index.tolist()

    if not indices_to_cat:
        return df_out

    cache: Dict[str, str] = {}

    # Gather unique descriptions to avoid duplicate LLM queries
    items_to_query = []
    index_to_cache_key = {}

    for idx in indices_to_cat:
        desc = str(df_out.at[idx, "description"]).strip()
        direction = str(df_out.at[idx, "direction"]).strip()
        cache_key = f"{direction}:{desc.lower()}"
        index_to_cache_key[idx] = cache_key

        if use_cache and cache_key in cache:
            continue

        # Check if already in items_to_query
        if not any(item["cache_key"] == cache_key for item in items_to_query):
            items_to_query.append({
                "index": len(items_to_query),
                "description": desc,
                "direction": direction,
                "cache_key": cache_key,
            })

    # Process items_to_query in batches
    for i in range(0, len(items_to_query), batch_size):
        batch = items_to_query[i : i + batch_size]
        batch_payload = [{"index": b["index"], "description": b["description"], "direction": b["direction"]} for b in batch]
        batch_results = classify_batch_llm(batch_payload)

        for b in batch:
            assigned_cat = batch_results.get(b["index"], "Other")
            cache[b["cache_key"]] = assigned_cat

    # Assign categories to df_out
    for idx in indices_to_cat:
        key = index_to_cache_key[idx]
        assigned = cache.get(key) or rule_based_fallback(
            str(df_out.at[idx, "description"]),
            str(df_out.at[idx, "direction"])
        )
        df_out.at[idx, "category"] = assigned

    return df_out
