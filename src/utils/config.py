"""Central configuration settings for Cashflow & Risk Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
UPLOAD_DATA_DIR = DATA_DIR / "uploads"

# Ensure data directories exist
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Unified transaction schema
UNIFIED_COLUMNS = [
    "date",
    "description",
    "amount",
    "direction",
    "source_file",
    "category",
]

# Standard allowed transaction categories
CATEGORIES = [
    "Rent",
    "Salary",
    "Inventory Purchase",
    "Utilities",
    "Sales Revenue",
    "Loan Repayment",
    "Marketing",
    "Taxes",
    "Equipment",
    "Other",
]

# Forecasting settings
DEFAULT_HORIZONS = [30, 60, 90]
DEFAULT_HORIZON_DAYS = 30
DEFAULT_SAFETY_BUFFER = 50000.0  # ₹50,000 minimum safety cushion
DEFAULT_RECEIVABLE_COLLECTION_DELAY_DAYS = 7

# Anomaly detection settings
DEFAULT_ZSCORE_THRESHOLD = 2.5
DEFAULT_ISOLATION_CONTAMINATION = 0.05
MIN_TRANSACTIONS_FOR_ZSCORE = 3

# LLM & OpenRouter settings
_DEFAULT_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "cohere/north-mini-code:free",
    "liquid/lfm-2.5-2.6b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

raw_models = os.getenv("OPENROUTER_MODELS", "")
if raw_models.strip():
    WORKING_MODELS = [m.strip() for m in raw_models.replace(";", ",").split(",") if m.strip()]
else:
    WORKING_MODELS = _DEFAULT_MODELS

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_BATCH_SIZE = 20
LLM_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "20"))
