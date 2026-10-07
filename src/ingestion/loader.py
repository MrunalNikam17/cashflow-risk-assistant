"""Parses uploaded CSV/Excel files into a unified transaction schema.

Normalizes diverse financial file formats (invoices, expenses, sales, bank statements)
into a standard DataFrame schema:
    [date, description, amount, direction, source_file, category]
where amount is strictly positive and direction is either 'inflow' or 'outflow'.
"""

import logging
import re
from pathlib import Path
from typing import List, Union, Optional
import pandas as pd
import numpy as np

from src.utils.config import UNIFIED_COLUMNS, CATEGORIES

logger = logging.getLogger(__name__)


def clean_amount(val) -> Optional[float]:
    """Cleans currency strings, symbols, and commas into a float."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    # Strip symbols like ₹, $, Rs., commas, whitespace
    cleaned = re.sub(r"[^\d.-]", "", str(val).strip())
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds the first column in df whose normalized name matches candidates."""
    norm_cols = {re.sub(r"[\s_-]+", "", c.lower()): c for c in df.columns}
    for candidate in candidates:
        norm_candidate = re.sub(r"[\s_-]+", "", candidate.lower())
        if norm_candidate in norm_cols:
            return norm_cols[norm_candidate]
    return None


def load_file(file_path_or_buffer: Union[str, Path, object], filename: Optional[str] = None) -> pd.DataFrame:
    """Loads and normalizes a single CSV or Excel file.

    Args:
        file_path_or_buffer: Path to file or file-like object (e.g. from Streamlit file_uploader).
        filename: Optional filename override when passing a buffer.

    Returns:
        DataFrame strictly conforming to UNIFIED_COLUMNS.
    """
    if isinstance(file_path_or_buffer, (str, Path)):
        path = Path(file_path_or_buffer)
        fname = path.name
        suffix = path.suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            raw_df = pd.read_excel(path)
        else:
            raw_df = pd.read_csv(path)
    else:
        fname = filename or getattr(file_path_or_buffer, "name", "uploaded_file.csv")
        suffix = Path(fname).suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            raw_df = pd.read_excel(file_path_or_buffer)
        else:
            raw_df = pd.read_csv(file_path_or_buffer)

    if raw_df.empty:
        logger.warning(f"File {fname} is empty.")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # Detect Date column
    date_col = find_column(
        raw_df,
        ["date", "transactiondate", "txdate", "bookingdate", "invoicedate", "paymentdate", "timestamp", "entrydate"]
    )
    if not date_col:
        raise ValueError(f"Could not identify a date column in '{fname}'. Columns found: {list(raw_df.columns)}")

    # Detect Description column
    desc_col = find_column(
        raw_df,
        ["description", "transactiondetails", "details", "narration", "particulars", "memo", "notes", "vendor", "clientname", "customer"]
    )

    # Detect Category column if present
    cat_col = find_column(raw_df, ["category", "expensecategory", "classification"])

    # Detect debit/credit dual columns (typical in bank statements)
    debit_col = find_column(raw_df, ["debit", "dr", "withdrawal", "paidout", "expenseamount"])
    credit_col = find_column(raw_df, ["credit", "cr", "deposit", "paidin", "incomeamount"])
    balance_col = find_column(raw_df, ["balance", "runningbalance", "closingbalance", "availablebalance"])

    # Detect single Amount column
    amt_col = find_column(raw_df, ["amount", "netsalesamount", "value", "total", "netamount", "invoiceamount"])

    # Detect explicit direction / type column
    dir_col = find_column(raw_df, ["direction", "type", "drcr", "transactiontype", "flow"])

    # Status column (for invoices e.g. PAID vs PENDING)
    status_col = find_column(raw_df, ["status", "paymentstatus"])

    records = []
    latest_balance = None

    for idx, row in raw_df.iterrows():
        # Date parsing
        raw_date = row[date_col]
        pay_date_col = find_column(raw_df, ["paymentdate", "paiddate"])
        if pay_date_col and pd.notna(row.get(pay_date_col)):
            val_pay = str(row.get(pay_date_col)).strip()
            if val_pay and val_pay.lower() not in ["nan", "none", "nat", ""]:
                raw_date = val_pay

        parsed_date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed_date):
            logger.warning(f"Row {idx} in {fname} skipped: invalid date '{raw_date}'")
            continue

        if balance_col and pd.notna(row.get(balance_col)):
            parsed_balance = clean_amount(row.get(balance_col))
            if parsed_balance is not None and (
                latest_balance is None or parsed_date >= latest_balance[0]
            ):
                latest_balance = (parsed_date, parsed_balance)

        # Description
        description = ""
        if desc_col and pd.notna(row.get(desc_col)):
            description = str(row[desc_col]).strip()
        elif cat_col and pd.notna(row.get(cat_col)):
            description = f"{row[cat_col]} Transaction"
        else:
            description = f"Transaction from {fname}"

        # Category
        category = None
        if cat_col and pd.notna(row.get(cat_col)):
            raw_cat = str(row[cat_col]).strip()
            # Normalize to known categories if close match
            for known_cat in CATEGORIES:
                if raw_cat.lower() == known_cat.lower():
                    category = known_cat
                    break
            if not category:
                category = raw_cat

        # Determine Amount and Direction
        # Scenario A: Dual Debit/Credit columns (e.g. bank statement)
        if debit_col or credit_col:
            raw_debit = row.get(debit_col) if debit_col else None
            raw_credit = row.get(credit_col) if credit_col else None
            dr_val = clean_amount(raw_debit) if debit_col else 0.0
            cr_val = clean_amount(raw_credit) if credit_col else 0.0
            if debit_col and pd.notna(raw_debit) and str(raw_debit).strip() and dr_val is None:
                logger.warning(f"Row {idx} in {fname} skipped malformed debit '{raw_debit}'.")
            if credit_col and pd.notna(raw_credit) and str(raw_credit).strip() and cr_val is None:
                logger.warning(f"Row {idx} in {fname} skipped malformed credit '{raw_credit}'.")
            dr_val = dr_val or 0.0
            cr_val = cr_val or 0.0

            if dr_val != 0:
                records.append({
                    "date": parsed_date,
                    "description": description,
                    "amount": round(abs(dr_val), 2),
                    "direction": "outflow" if dr_val > 0 else "inflow",
                    "source_file": fname,
                    "category": category,
                })
            if cr_val != 0:
                records.append({
                    "date": parsed_date,
                    "description": description,
                    "amount": round(abs(cr_val), 2),
                    "direction": "inflow" if cr_val > 0 else "outflow",
                    "source_file": fname,
                    "category": category or "Sales Revenue",
                })
            continue

        # Scenario B: Single Amount column
        if amt_col:
            raw_amt = clean_amount(row.get(amt_col))
            if raw_amt is None:
                logger.warning(f"Row {idx} in {fname} skipped: unparseable amount '{row.get(amt_col)}'")
                continue

            # Check status for invoices (if PENDING, we still record but note it; if PAID, inflow)
            direction = None
            if dir_col and pd.notna(row.get(dir_col)):
                val_dir = str(row[dir_col]).strip().lower()
                if val_dir in ["inflow", "credit", "cr", "deposit", "sale", "income", "received"]:
                    direction = "inflow"
                elif val_dir in ["outflow", "debit", "dr", "withdrawal", "expense", "paid"]:
                    direction = "outflow"

            # If no explicit direction column, infer from sign or filename
            if not direction:
                if raw_amt < 0:
                    direction = "outflow"
                else:
                    lower_fname = fname.lower()
                    if any(k in lower_fname for k in ["expense", "bill", "payout", "cost", "vendor"]):
                        direction = "outflow"
                    elif any(k in lower_fname for k in ["sale", "pos", "revenue", "invoice", "income", "credit"]):
                        direction = "inflow"
                    else:
                        direction = "inflow"

            # Set default category for obvious sales files if empty
            if not category:
                if direction == "inflow" and any(k in fname.lower() for k in ["sale", "pos", "invoice"]):
                    category = "Sales Revenue"

            records.append({
                "date": parsed_date,
                "description": description,
                "amount": round(abs(raw_amt), 2),
                "direction": direction,
                "source_file": fname,
                "category": category,
            })

    if not records:
        logger.warning(f"No valid transaction records extracted from {fname}.")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df_result = pd.DataFrame(records)[UNIFIED_COLUMNS]
    df_result["date"] = pd.to_datetime(df_result["date"])
    df_result = df_result.sort_values("date").reset_index(drop=True)
    if latest_balance is not None:
        df_result.attrs["current_balance"] = float(latest_balance[1])
    return df_result


def load_files(file_paths_or_buffers: List[Union[str, Path, object]]) -> pd.DataFrame:
    """Loads and concatenates multiple CSV/Excel files into a single unified DataFrame."""
    dfs = []
    for item in file_paths_or_buffers:
        try:
            df = load_file(item)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Failed loading file {item}: {e}", exc_info=True)

    if not dfs:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    balance_sources = [df.attrs["current_balance"] for df in dfs if "current_balance" in df.attrs]
    if balance_sources:
        combined.attrs["current_balance"] = float(balance_sources[-1])
    return combined
