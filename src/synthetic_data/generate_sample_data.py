"""Generates realistic synthetic financial CSV files for a small business.

Creates 4 realistic CSVs over a 6-month period (Oct 2025 - Mar 2026):
  1. invoices.csv      (B2B client invoices, paid & pending)
  2. expenses.csv      (Vendor bills, rent, salary, utilities, anomalies)
  3. sales.csv         (Daily retail POS transactions with weekly seasonality)
  4. bank_statement.csv(Comprehensive checking account ledger with running balance)

Includes realistic small business patterns:
  - Recurring monthly rent (₹40,000 on 1st) and payroll (₹65,000 on 5th)
  - Weekly retail sales peaks (higher on weekends)
  - Deliberate anomalies:
      * ₹45,000 surge utility bill (vs normal ₹7,500)
      * ₹85,000 emergency chiller repair
  - Tight cash crunch period in late February
"""

import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import SAMPLE_DATA_DIR


def generate_all_sample_data(output_dir: Path = SAMPLE_DATA_DIR) -> dict:
    """Generates 4 realistic CSV datasets and saves them to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    start_date = date(2025, 10, 1)
    end_date = date(2026, 3, 31)
    total_days = (end_date - start_date).days + 1

    # -------------------------------------------------------------
    # 1. EXPENSES (Vendor payments, recurring bills, anomalies)
    # -------------------------------------------------------------
    expense_records = []
    exp_id = 1001

    # Generate recurring monthly items
    current = start_date
    while current <= end_date:
        # Rent on 1st
        rent_date = date(current.year, current.month, 1)
        if start_date <= rent_date <= end_date:
            expense_records.append({
                "expense_id": f"EXP-{exp_id}",
                "date": rent_date.isoformat(),
                "vendor": "Apex Commercial Properties",
                "category": "Rent",
                "description": f"Monthly Storefront Retail Lease Rent - {rent_date.strftime('%B %Y')}",
                "amount": 40000.0,
                "payment_method": "Bank Transfer",
            })
            exp_id += 1

        # Salary on 5th
        sal_date = date(current.year, current.month, 5)
        if start_date <= sal_date <= end_date:
            expense_records.append({
                "expense_id": f"EXP-{exp_id}",
                "date": sal_date.isoformat(),
                "vendor": "HROps Payroll Disbursal",
                "category": "Salary",
                "description": f"Staff Salaries & Wages Payroll Disbursal ({sal_date.strftime('%B %Y')})",
                "amount": 65000.0,
                "payment_method": "NEFT",
            })
            exp_id += 1

        # Loan EMI on 20th
        emi_date = date(current.year, current.month, 20)
        if start_date <= emi_date <= end_date:
            expense_records.append({
                "expense_id": f"EXP-{exp_id}",
                "date": emi_date.isoformat(),
                "vendor": "HDFC Business Loan Division",
                "category": "Loan Repayment",
                "description": "Equipment Term Loan Monthly EMI Installment",
                "amount": 18500.0,
                "payment_method": "Auto-Debit",
            })
            exp_id += 1

        # Regular Utilities on ~12th-14th
        util_date = date(current.year, current.month, 13)
        if start_date <= util_date <= end_date:
            # Inject utility anomaly in January 2026!
            if current.year == 2026 and current.month == 1:
                util_amt = 45000.0
                util_desc = "State Electricity Board Commercial Power Surge & Penalty Meter Charge"
            else:
                util_amt = round(random.uniform(6800.0, 8400.0), 2)
                util_desc = f"Commercial Electricity & Water Utility Bill ({util_date.strftime('%B %Y')})"

            expense_records.append({
                "expense_id": f"EXP-{exp_id}",
                "date": util_date.isoformat(),
                "vendor": "State Electricity & Municipal Utility",
                "category": "Utilities",
                "description": util_desc,
                "amount": util_amt,
                "payment_method": "Net Banking",
            })
            exp_id += 1

        # Move to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    # Periodic Inventory, Marketing, Maintenance
    current_day = start_date
    while current_day <= end_date:
        # Marketing ads every ~10 days
        if current_day.day in [8, 18, 28]:
            mkt_amt = round(random.uniform(3500.0, 6500.0), 2)
            expense_records.append({
                "expense_id": f"EXP-{exp_id}",
                "date": current_day.isoformat(),
                "vendor": "Meta Ads & Local Print Media",
                "category": "Marketing",
                "description": "Digital Ad Campaign & Promotional Local Print Flyers",
                "amount": mkt_amt,
                "payment_method": "Credit Card",
            })
            exp_id += 1

        # Inventory restocking twice a month (approx 10th and 24th)
        if current_day.day in [10, 24]:
            inv_amt = round(random.uniform(28000.0, 48000.0), 2)
            expense_records.append({
                "expense_id": f"EXP-{exp_id}",
                "date": current_day.isoformat(),
                "vendor": "Global Supplies Wholesale Ltd",
                "category": "Inventory Purchase",
                "description": "Bulk Wholesale Stock Restock & Raw Material Order",
                "amount": inv_amt,
                "payment_method": "Bank Transfer",
            })
            exp_id += 1

        # Periodic Routine maintenance
        if current_day.day == 16 and current_day.month in [10, 12, 2]:
            maint_amt = round(random.uniform(3200.0, 5500.0), 2)
            expense_records.append({
                "expense_id": f"EXP-{exp_id}",
                "date": current_day.isoformat(),
                "vendor": "QuickFix Commercial Facility Care",
                "category": "Equipment",
                "description": "Routine Store HVAC Maintenance & Filter Servicing",
                "amount": maint_amt,
                "payment_method": "UPI",
            })
            exp_id += 1

        current_day += timedelta(days=1)

    # Injected Major Equipment Anomaly in late February 2026!
    expense_records.append({
        "expense_id": f"EXP-{exp_id}",
        "date": "2026-02-22",
        "vendor": "CoolTech Industrial Refrigeration Corp",
        "category": "Equipment",
        "description": "Emergency Industrial Chiller Compressor Complete Overhaul & Repair",
        "amount": 85000.0,
        "payment_method": "Bank Transfer",
    })
    exp_id += 1

    # Quarterly GST Taxes (Oct 20, Jan 20)
    for tax_date_str in ["2025-10-20", "2026-01-20"]:
        expense_records.append({
            "expense_id": f"EXP-{exp_id}",
            "date": tax_date_str,
            "vendor": "GST State Tax Treasury",
            "category": "Taxes",
            "description": "Quarterly GST Direct Challan Tax Payment Filing",
            "amount": 24500.0,
            "payment_method": "Government Portal NEFT",
        })
        exp_id += 1

    df_expenses = pd.DataFrame(expense_records).sort_values("date").reset_index(drop=True)
    expenses_path = output_dir / "expenses.csv"
    df_expenses.to_csv(expenses_path, index=False)

    # -------------------------------------------------------------
    # 2. SALES (Daily POS transactions with weekend & seasonal lift)
    # -------------------------------------------------------------
    sales_records = []
    tx_counter = 5001

    for day_idx in range(total_days):
        day_date = start_date + timedelta(days=day_idx)
        weekday = day_date.weekday()  # 0=Monday, 6=Sunday

        # Baseline: Mon-Thu lower, Fri moderate, Sat-Sun peak
        if weekday in [5, 6]:  # Weekend
            base = random.uniform(9500.0, 16000.0)
        elif weekday == 4:     # Friday
            base = random.uniform(7000.0, 11500.0)
        else:                  # Mon-Thu
            base = random.uniform(4200.0, 7800.0)

        # Seasonal lift during festive Nov-Dec (+25%) and post-holiday dip in Feb (-15%)
        if day_date.month in [11, 12]:
            base *= 1.25
        elif day_date.month == 2:
            base *= 0.85

        # Split daily sales into 2-3 batches (POS Card/UPI vs Cash/Online)
        card_portion = round(base * 0.65, 2)
        upi_cash_portion = round(base * 0.35, 2)

        sales_records.append({
            "transaction_id": f"TXN-{tx_counter}",
            "date": day_date.isoformat(),
            "payment_channel": "POS Terminal / Credit Card",
            "description": "Daily Point-of-Sale Card Settlements & Merchant Batch Inflow",
            "amount": card_portion,
            "direction": "inflow",
        })
        tx_counter += 1

        sales_records.append({
            "transaction_id": f"TXN-{tx_counter}",
            "date": day_date.isoformat(),
            "payment_channel": "UPI / QR / Cash Deposit",
            "description": "Counter UPI QR & Daily Retail Cash Sales Register Settlement",
            "amount": upi_cash_portion,
            "direction": "inflow",
        })
        tx_counter += 1

    df_sales = pd.DataFrame(sales_records).sort_values("date").reset_index(drop=True)
    sales_path = output_dir / "sales.csv"
    df_sales.to_csv(sales_path, index=False)

    # -------------------------------------------------------------
    # 3. INVOICES (B2B Client Invoices: Paid & Pending)
    # -------------------------------------------------------------
    clients = [
        "Metropolis Corporate Gifts LLP",
        "Zenith Hospitality Group",
        "Aura Wellness Centers",
        "Pinnacle Tech Campus Cafeteria",
        "Silverline Event Organizers",
    ]

    invoice_records = []
    inv_num = 201

    current = start_date
    while current <= end_date:
        # 2-3 B2B orders per month
        num_invoices = random.randint(2, 3)
        for _ in range(num_invoices):
            inv_day = random.randint(2, 26)
            try:
                inv_date = date(current.year, current.month, inv_day)
            except ValueError:
                inv_date = date(current.year, current.month, 25)

            if inv_date > end_date:
                continue

            due_date = inv_date + timedelta(days=25)
            inv_amount = round(random.uniform(18000.0, 42000.0), 2)
            client = random.choice(clients)

            # Invoices due before mid March are paid; later ones might be pending
            if due_date <= date(2026, 3, 10):
                status = "PAID"
                payment_date = (due_date - timedelta(days=random.randint(0, 5))).isoformat()
            else:
                status = "PENDING"
                payment_date = ""

            invoice_records.append({
                "invoice_number": f"INV-{inv_num}",
                "invoice_date": inv_date.isoformat(),
                "due_date": due_date.isoformat(),
                "client_name": client,
                "amount": inv_amount,
                "status": status,
                "payment_date": payment_date,
                "description": f"B2B Bulk Corporate Order Invoice #{inv_num} - {client}",
            })
            inv_num += 1

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    df_invoices = pd.DataFrame(invoice_records).sort_values("invoice_date").reset_index(drop=True)
    invoices_path = output_dir / "invoices.csv"
    df_invoices.to_csv(invoices_path, index=False)

    # -------------------------------------------------------------
    # 4. BANK STATEMENT (Comprehensive Ledger with Running Balance)
    # -------------------------------------------------------------
    # We combine paid items, sales deposits, and expenses into a chronological ledger
    bank_events = []

    # Add expenses as Debits (Outflow)
    for exp in expense_records:
        bank_events.append({
            "date": exp["date"],
            "description": f"{exp['category']}: {exp['description']} (Ref: {exp['vendor']})",
            "debit": exp["amount"],
            "credit": 0.0,
            "type": "debit",
        })

    # Add daily sales deposits as Credits (Inflow) - grouped daily for realistic statement
    daily_sales = df_sales.groupby("date")["amount"].sum().reset_index()
    for _, row in daily_sales.iterrows():
        bank_events.append({
            "date": row["date"],
            "description": "Merchant Settlement Credit - Daily POS & Store Card Inflow",
            "debit": 0.0,
            "credit": row["amount"],
            "type": "credit",
        })

    # Add paid B2B invoices as Credits (Inflow)
    paid_invs = df_invoices[df_invoices["status"] == "PAID"]
    for _, inv in paid_invs.iterrows():
        if inv["payment_date"]:
            bank_events.append({
                "date": inv["payment_date"],
                "description": f"NEFT Client Inward Remittance - {inv['client_name']} (Inv #{inv['invoice_number']})",
                "debit": 0.0,
                "credit": inv["amount"],
                "type": "credit",
            })

    # Sort all bank events by date
    bank_events.sort(key=lambda x: x["date"])

    # Compute running balance starting at ₹125,000
    opening_balance = 125000.0
    current_balance = opening_balance
    statement_rows = []

    ref_seq = 80001
    for ev in bank_events:
        current_balance += ev["credit"] - ev["debit"]
        statement_rows.append({
            "date": ev["date"],
            "transaction_details": ev["description"],
            "reference_no": f"TXNREF{ref_seq}",
            "debit": round(ev["debit"], 2) if ev["debit"] > 0 else 0.0,
            "credit": round(ev["credit"], 2) if ev["credit"] > 0 else 0.0,
            "balance": round(current_balance, 2),
        })
        ref_seq += 1

    df_bank = pd.DataFrame(statement_rows)
    bank_path = output_dir / "bank_statement.csv"
    df_bank.to_csv(bank_path, index=False)

    summary = {
        "expenses": len(df_expenses),
        "sales": len(df_sales),
        "invoices": len(df_invoices),
        "bank_statement_rows": len(df_bank),
        "starting_balance": opening_balance,
        "final_balance": round(current_balance, 2),
        "min_balance": round(df_bank["balance"].min(), 2),
        "min_balance_date": df_bank.loc[df_bank["balance"].idxmin(), "date"],
        "paths": {
            "expenses": str(expenses_path),
            "sales": str(sales_path),
            "invoices": str(invoices_path),
            "bank_statement": str(bank_path),
        }
    }
    return summary


if __name__ == "__main__":
    print("Generating synthetic small business datasets...")
    res = generate_all_sample_data()
    print("\n--- Generation Summary ---")
    print(f"Expenses records:       {res['expenses']}")
    print(f"Sales records:          {res['sales']}")
    print(f"Invoices records:       {res['invoices']}")
    print(f"Bank statement entries: {res['bank_statement_rows']}")
    print(f"Opening Bank Balance:   Rs. {res['starting_balance']:,.2f}")
    print(f"Closing Bank Balance:   Rs. {res['final_balance']:,.2f}")
    print(f"Lowest Cash Point:      Rs. {res['min_balance']:,.2f} on {res['min_balance_date']}")
    print("Files successfully generated in:", SAMPLE_DATA_DIR)
