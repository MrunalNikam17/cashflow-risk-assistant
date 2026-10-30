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

from src.utils.config import UNIFIED_COLUMNS, CATEGORIES

logger = logging.getLogger(__name__)


def clean_amount(val) -> Optional[float]:
    """Cleans currency strings, symbols, and commas into a float."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
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


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _deduplicate_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df.copy(), 0
    normalized = df.copy()
    normalized["_description_key"] = normalized["description"].fillna("").map(_normalize_text)
    normalized["_description_key"] = normalized["_description_key"].str.replace(r"[^a-z0-9 ]+", " ", regex=True)
    normalized["_date_key"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    normalized["_dup_key"] = (
        normalized["_date_key"]
        + "|"
        + normalized["amount"].round(2).astype(str)
        + "|"
        + normalized["direction"]
        + "|"
        + normalized["_description_key"]
    )
    deduped = normalized.drop_duplicates(subset=["_dup_key"], keep="first")
    dropped = int(len(normalized) - len(deduped))
    deduped = deduped.drop(columns=["_date_key", "_description_key", "_dup_key"])
    return deduped.reset_index(drop=True), dropped


def load_file(file_path_or_buffer: Union[str, Path, object], filename: Optional[str] = None) -> pd.DataFrame:
    """Loads and normalizes a single CSV or Excel file.

    Raises:
        ValueError: For empty files, missing required columns, or malformed dates.
    """
    if isinstance(file_path_or_buffer, (str, Path)):
        path = Path(file_path_or_buffer)
        fname = path.name
        suffix = path.suffix.lower()
        try:
            if suffix in [".xlsx", ".xls"]:
                raw_df = pd.read_excel(path)
            else:
                raw_df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - file read errors are surfaced clearly
            raise ValueError(f"Could not read '{fname}': {exc}") from exc
    else:
        fname = filename or getattr(file_path_or_buffer, "name", "uploaded_file.csv")
        suffix = Path(fname).suffix.lower()
        try:
            if suffix in [".xlsx", ".xls"]:
                raw_df = pd.read_excel(file_path_or_buffer)
            else:
                raw_df = pd.read_csv(file_path_or_buffer)
        except Exception as exc:
            raise ValueError(f"Could not read '{fname}': {exc}") from exc

    if raw_df is None or raw_df.empty:
        raise ValueError(f"File '{fname}' is empty.")

    date_col = find_column(
        raw_df,
        ["date", "transactiondate", "txdate", "bookingdate", "invoicedate", "paymentdate", "timestamp", "entrydate"],
    )
    if not date_col:
        raise ValueError(f"File '{fname}' is missing a required date column. Available columns: {list(raw_df.columns)}")

    desc_col = find_column(
        raw_df,
        ["description", "transactiondetails", "details", "narration", "particulars", "memo", "notes", "vendor", "clientname", "customer"],
    )
    cat_col = find_column(raw_df, ["category", "expensecategory", "classification"])
    debit_col = find_column(raw_df, ["debit", "dr", "withdrawal", "paidout", "expenseamount"])
    credit_col = find_column(raw_df, ["credit", "cr", "deposit", "paidin", "incomeamount"])
    balance_col = find_column(raw_df, ["balance", "runningbalance", "closingbalance", "availablebalance"])
    amt_col = find_column(raw_df, ["amount", "netsalesamount", "value", "total", "netamount", "invoiceamount"])
    dir_col = find_column(raw_df, ["direction", "type", "drcr", "transactiontype", "flow"])
    status_col = find_column(raw_df, ["status", "paymentstatus"])
    pay_date_col = find_column(raw_df, ["paymentdate", "paiddate"])

    if not (amt_col or debit_col or credit_col):
        raise ValueError(
            f"File '{fname}' is missing a required money column. Expected an 'amount' column or debit/credit columns."
        )

    records = []
    expected_receivables = []
    latest_balance = None

    for idx, row in raw_df.iterrows():
        raw_date = row.get(date_col)
        if pd.isna(raw_date) or str(raw_date).strip() in ["", "nan", "none", "nat"]:
            raise ValueError(f"Bad date value in '{fname}' row {idx + 2}: date is missing.")

        effective_date = raw_date
        if pay_date_col and pd.notna(row.get(pay_date_col)):
            val_pay = str(row.get(pay_date_col)).strip()
            if val_pay and val_pay.lower() not in ["nan", "none", "nat", ""]:
                effective_date = val_pay

        parsed_date = pd.to_datetime(effective_date, errors="coerce")
        if pd.isna(parsed_date):
            raise ValueError(f"Bad date value in '{fname}' row {idx + 2}: '{effective_date}' is not parseable.")

        if balance_col and pd.notna(row.get(balance_col)):
            parsed_balance = clean_amount(row.get(balance_col))
            if parsed_balance is not None and (latest_balance is None or parsed_date >= latest_balance[0]):
                latest_balance = (parsed_date, parsed_balance)

        description = ""
        if desc_col and pd.notna(row.get(desc_col)):
            description = str(row[desc_col]).strip()
        elif cat_col and pd.notna(row.get(cat_col)):
            description = f"{row[cat_col]} Transaction"
        else:
            description = f"Transaction from {fname}"

        category = None
        if cat_col and pd.notna(row.get(cat_col)):
            raw_cat = str(row[cat_col]).strip()
            for known_cat in CATEGORIES:
                if raw_cat.lower() == known_cat.lower():
                    category = known_cat
                    break
            if not category:
                category = raw_cat

        if debit_col or credit_col:
            raw_debit = row.get(debit_col) if debit_col else None
            raw_credit = row.get(credit_col) if credit_col else None
            dr_val = clean_amount(raw_debit) if debit_col else 0.0
            cr_val = clean_amount(raw_credit) if credit_col else 0.0
            if debit_col and pd.notna(raw_debit) and str(raw_debit).strip() and dr_val is None:
                raise ValueError(f"Malformed debit value in '{fname}' row {idx + 2}: '{raw_debit}'")
            if credit_col and pd.notna(raw_credit) and str(raw_credit).strip() and cr_val is None:
                raise ValueError(f"Malformed credit value in '{fname}' row {idx + 2}: '{raw_credit}'")
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

        if amt_col:
            raw_amt = clean_amount(row.get(amt_col))
            if raw_amt is None:
                raise ValueError(f"Malformed amount value in '{fname}' row {idx + 2}: '{row.get(amt_col)}'")

            status_value = str(row.get(status_col, "")).strip().lower() if status_col and pd.notna(row.get(status_col)) else ""
            paid_statuses = {"paid", "paid in full", "settled", "completed", "received", "collected"}
            pending_statuses = {"pending", "overdue", "open", "outstanding", "unpaid", "in arrears"}

            if status_col and status_value:
                if status_value in paid_statuses:
                    payment_date = parsed_date
                    if pay_date_col and pd.notna(row.get(pay_date_col)):
                        candidate = pd.to_datetime(row.get(pay_date_col), errors="coerce")
                        if not pd.isna(candidate):
                            payment_date = candidate
                    records.append({
                        "date": payment_date,
                        "description": description,
                        "amount": round(abs(raw_amt), 2),
                        "direction": "inflow",
                        "source_file": fname,
                        "category": category or "Sales Revenue",
                    })
                    continue
                if status_value in pending_statuses:
                    expected_receivables.append({
                        "due_date": pd.to_datetime(parsed_date).normalize(),
                        "amount": round(abs(raw_amt), 2),
                        "description": description,
                        "status": status_value,
                        "source_file": fname,
                    })
                    continue

            direction = None
            if dir_col and pd.notna(row.get(dir_col)):
                val_dir = str(row[dir_col]).strip().lower()
                if val_dir in ["inflow", "credit", "cr", "deposit", "sale", "income", "received"]:
                    direction = "inflow"
                elif val_dir in ["outflow", "debit", "dr", "withdrawal", "expense", "paid"]:
                    direction = "outflow"

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

            if not category and direction == "inflow" and any(k in fname.lower() for k in ["sale", "pos", "invoice"]):
                category = "Sales Revenue"

            records.append({
                "date": parsed_date,
                "description": description,
                "amount": round(abs(raw_amt), 2),
                "direction": direction,
                "source_file": fname,
                "category": category,
            })

    if not records and not expected_receivables:
        raise ValueError(f"No valid transaction records were extracted from '{fname}'.")

    df_result = pd.DataFrame(records)[UNIFIED_COLUMNS] if records else pd.DataFrame(columns=UNIFIED_COLUMNS)
    if not df_result.empty:
        df_result["date"] = pd.to_datetime(df_result["date"])
        df_result = df_result.sort_values("date").reset_index(drop=True)
    if latest_balance is not None:
        df_result.attrs["current_balance"] = float(latest_balance[1])
    if expected_receivables:
        expected_df = pd.DataFrame(expected_receivables)
        expected_df["due_date"] = pd.to_datetime(expected_df["due_date"]).dt.normalize()
        df_result.attrs["expected_receivables"] = expected_df
    return df_result


def load_files(file_paths_or_buffers: List[Union[str, Path, object]]) -> pd.DataFrame:
    """Loads and concatenates multiple CSV/Excel files into a single unified DataFrame."""
    if not file_paths_or_buffers:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    dfs = []
    expected_frames = []
    for item in file_paths_or_buffers:
        try:
            df = load_file(item)
            if not df.empty:
                dfs.append(df)
            if "expected_receivables" in df.attrs:
                expected_frames.append(df.attrs["expected_receivables"])
        except Exception as e:
            logger.error(f"Failed loading file {item}: {e}", exc_info=True)

    if not dfs:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    combined = pd.concat(dfs, ignore_index=True)
    combined, duplicates_dropped = _deduplicate_transactions(combined)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.attrs["duplicates_dropped"] = duplicates_dropped

    if expected_frames:
        expected_df = pd.concat(expected_frames, ignore_index=True)
        expected_df = expected_df.drop_duplicates(subset=["due_date", "amount", "description"], keep="first")
        combined.attrs["expected_receivables"] = expected_df

    balance_sources = [df.attrs["current_balance"] for df in dfs if "current_balance" in df.attrs]
    if balance_sources:
        combined.attrs["current_balance"] = float(balance_sources[-1])
    return combined
