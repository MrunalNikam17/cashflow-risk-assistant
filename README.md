# 💼 Cashflow & Risk Assistant
### *AI-Powered Financial Advisor for Small Businesses Grounded in Deterministic Math*

> **One-Line Pitch:** A transparent, explainable financial advisor that transforms raw transaction data into proactive cash flow forecasts, detects spending anomalies, and provides mathematical, hallucination-free affordability recommendations.

---

## 🛑 Problem Statement

Small business owners frequently face critical financial questions:
- *"Can I afford to purchase ₹80,000 of inventory next month?"*
- *"Will our current runway survive upcoming payroll and supplier commitments?"*
- *"Are there hidden spending leaks or unusual supplier charges eroding our margins?"*

Traditional accounting tools present static historical charts but fail to project forward-looking risk. Conversely, asking general-purpose Large Language Models (LLMs) financial questions is dangerous: LLMs are prone to hallucinating numbers, fabricating arithmetic, and giving vague or confident yet catastrophic financial advice without computing true working capital constraints.

**Cashflow & Risk Assistant** solves this by enforcing a strict **Compute-First, Explain-Second** architecture:
1. **Python computes the ground truth first** (daily cash flow velocity, trend, weekly seasonality, recurring monthly obligations, minimum projected balances, and safety buffer thresholds).
2. **The LLM is invoked only to articulate** a clear, empathetic, natural-language explanation of the already-computed numbers. The LLM is strictly prohibited from inventing, altering, or recalculating any figures.

---

## 🏗️ Architecture & Pipeline

```
+-------------------------------------------------------------------------------------------------+
|                                    1. MULTI-SOURCE INGESTION                                    |
|   Invoices (CSV/XLSX)  •  Vendor Expenses  •  POS Retail Sales  •  Bank Statements (Debit/Credit) |
+-------------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-------------------------------------------------------------------------------------------------+
|                                   2. SCHEMA NORMALIZATION                                       |
|      Unified schema: [date, description, amount, direction (inflow/outflow), source_file]       |
+-------------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-------------------------------------------------------------------------------------------------+
|                             3. BATCHED OPENROUTER CATEGORIZATION                                |
|    Batched few-shot LLM (free-tier fallback) -> [Rent, Salary, Inventory, Utilities, Sales, ...]  |
|    Deterministic keyword rules fallback when offline                                            |
+-------------------------------------------------------------------------------------------------+
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
+-----------------------------------------+   +--------------------------------------------------+
|      4. STATISTICAL ANOMALY DETECTOR    |   |           5. EXPLAINABLE CASH FORECASTER         |
|  - Per-category Z-Score & MAD analysis  |   |  - Additive decomposition:                       |
|  - Multi-variate IsolationForest        |   |    Trend + Weekly Seasonality + Monthly Anchors  |
|  - Generates human-readable findings    |   |  - 30 / 60 / 90-day forward cumulative balance   |
|    (e.g. "6.0x typical utility spend")  |   |  - Identifies lowest balance point & date        |
+-----------------------------------------+   |  - Flags cash deficits & safety buffer breaches  |
                     │                        +--------------------------------------------------+
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                ▼
+-------------------------------------------------------------------------------------------------+
|                       6. GROUNDED AFFORDABILITY ENGINE (insight_agent.py)                       |
|                                                                                                 |
|   Step A: Extract purchase amount (e.g. ₹80,000) & evaluation window (e.g. 30 days)            |
|   Step B: Slice projected timeline -> compute pre-purchase low and post-purchase low            |
|   Step C: Determine strict deterministic verdict:                                               |
|           • SAFE: Post-purchase min >= Safety Buffer                                            |
|           • RISKY: 0 <= Post-purchase min < Safety Buffer                                       |
|           • NOT RECOMMENDED: Post-purchase min < 0 (Cash Deficit)                               |
|   Step D: Invoke OpenRouter LLM with computed numbers -> synthesize grounded explanation        |
|   Step E: Template fallback ensures 100% uptime even if offline                                 |
+-------------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-------------------------------------------------------------------------------------------------+
|                                 7. STREAMLIT INTERACTIVE DASHBOARD                              |
|          Tab 1: Overview & Ledger  •  Tab 2: Forecast Chart  •  Tab 3: Anomalies                |
|                             Tab 4: Interactive Affordability Advisor                            |
+-------------------------------------------------------------------------------------------------+
```

---

## 🎯 Why "Compute-First, Then Explain" Matters

In financial applications, **a hallucinated number is worse than no answer at all**. If an LLM estimates that an owner has ₹1,20,000 in cash when their true projected minimum balance is negative ₹15,000 due to recurring payroll, following that advice could trigger bankruptcy or bounced vendor checks.

### How `src/insight/insight_agent.py` Prevents Financial Hallucinations:

1. **Deterministic Execution:** The decision logic does not ask the LLM *"Can the user afford this?"*. Python code calculates:
   $$B'_{min} = B_{min} - \text{Purchase Amount}$$
   $$\Delta_{buffer} = B'_{min} - \text{Safety Buffer}$$
2. **Fixed Verdict Contract:** Python assigns the verdict (`SAFE`, `RISKY`, or `NOT RECOMMENDED`) before any LLM call occurs.
3. **Constrained System Persona:** The LLM's prompt explicitly injects the exact pre-computed figures and instructs:
   > *"You are provided with pre-computed cash flow numbers and an already-decided verdict. Your role is ONLY to explain this decision in clear terms. CRITICAL RULE: You MUST use ONLY the exact numbers provided. Do NOT invent, recalculate, round differently, or introduce any financial figures of your own."*
4. **Deterministic Template Fallback:** If OpenRouter is unreachable, rate-limited, or no API key is supplied, a structured template generates a comprehensive mathematical audit report without dropping a single computation.

---

## 📈 Explainable Forecasting vs. Black-Box Models

Many forecasting systems apply complex deep learning (e.g., LSTMs, Transformers) or opaque proprietary models. For small business advisory, **explainability is paramount**:

- **Auditable Drivers:** Business owners need to know *why* a dip happens. Our forecaster explicitly isolates:
  - **Base Growth / Burn Trend:** Linear trend rate ($\Delta ₹/\text{day}$) representing operational trajectory.
  - **Weekly Day-of-Week Seasonality:** Captures weekend retail spikes versus weekday lulls.
  - **Monthly Recurring Anchors:** Identifies predictable monthly spikes (Rent on the 1st, Payroll on the 5th, Loan EMI on the 20th).
- **Transparency Over Black Boxes:** Instead of opaque neural weights, the projected balance on any future date $t$ is directly verifiable:
  $$\text{Net}(t) = \text{Trend}(t) + \text{Seasonality}_{\text{weekday}}(t) + \text{Recurrence}_{\text{day\_of\_month}}(t)$$
- **Shortage Pinpointing:** Computes the exact projected lowest balance point and its calendar date, alerting owners to impending liquidity crunches weeks in advance.

---

## 🗂️ Project Structure

```
cashflow-risk-assistant/
├── data/
│   ├── sample/                     # Pre-generated realistic synthetic CSV datasets
│   │   ├── invoices.csv            # Client orders (paid & pending)
│   │   ├── expenses.csv            # Vendor expenses & recurring commitments
│   │   ├── sales.csv               # Daily POS & card sales register
│   │   └── bank_statement.csv      # Checking account ledger with running balance
│   └── uploads/                    # Directory for runtime user uploads
├── src/
│   ├── ingestion/
│   │   └── loader.py               # Multi-format CSV/Excel parser & schema normalizer
│   ├── categorization/
│   │   └── categorizer.py          # Batched LLM classification with keyword fallback
│   ├── forecasting/
│   │   └── forecaster.py           # Explainable trend + seasonality time-series engine
│   ├── anomaly/
│   │   └── anomaly_detector.py     # Statistical Z-Score, MAD & IsolationForest audits
│   ├── insight/
│   │   └── insight_agent.py        # Compute-first affordability reasoning engine
│   ├── utils/
│   │   ├── llm.py                   # OpenRouter multi-model fallback caller
│   │   └── config.py                # Central thresholds, categories, and settings
│   └── synthetic_data/
│       └── generate_sample_data.py # Realistic 6-month retail dataset generator
├── dashboard/
│   └── app.py                      # 4-tab Streamlit dashboard with custom modern theme
├── tests/
│   ├── test_categorizer.py         # Batching & category validity tests
│   ├── test_forecaster.py          # Projections, horizons, and shortage tests
│   ├── test_anomaly_detector.py    # Z-score, MAD, and explanation tests
│   └── test_insight_agent.py       # Compute-first math & verdict tests
├── requirements.txt                # Production dependencies
├── .env.example                    # Template for OpenRouter credentials
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project documentation
```

---

## 🧰 Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Core Runtime** | Python 3.10+ | Standard ecosystem for financial data engineering & ML |
| **Data Ingestion** | Pandas, OpenPyXL | High-performance ingestion of tabular CSV and Excel spreadsheets |
| **Numerical Math** | NumPy | Vectorized time-series calculations and statistical operations |
| **Forecasting** | Statsmodels, NumPy | Explainable additive time-series decomposition (trend + seasonality) |
| **Anomaly Detection** | Scikit-learn, SciPy | Robust Z-Score, Median Absolute Deviation (MAD), IsolationForest |
| **Dashboard UI** | Streamlit, Plotly | Interactive web dashboard with responsive charts & modern styling |
| **LLM Reasoning** | OpenRouter Free Tier | Multi-model fallback across top free LLMs (zero local GPU required) |
| **Testing** | Pytest | Comprehensive automated unit and integration test suite |

---

## 🚀 Setup & Execution Guide

### 1. Clone & Setup Environment

```bash
git clone https://github.com/MrunalNikam17/cashflow-risk-assistant.git
cd cashflow-risk-assistant

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

The application includes full offline fallback support for ingestion, forecasting, anomaly detection, and affordability math. To enable OpenRouter LLM reasoning:

```bash
cp .env.example .env
```
Edit `.env` and set your key from [openrouter.ai/keys](https://openrouter.ai/keys):
```ini
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Generate Realistic Sample Data

Generate 6 months of synthetic retail data (invoices, expenses, sales, bank statement) featuring recurring rent, payroll, weekend sales peaks, and deliberate anomalies:

```bash
python -m src.synthetic_data.generate_sample_data
```

### 4. Run Automated Tests

Run the full pytest suite (23 tests covering all modules):

```bash
pytest -v tests/
```

### 5. Launch Interactive Dashboard

Start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`. Click **"🚀 Try with sample data"** in the sidebar to test all 4 tabs instantly. The dashboard reads the consolidated synthetic bank ledger so the same transactions in the sales, invoice, and expense detail files are not double-counted.

---

## 🛡️ OpenRouter Multi-Model Fallback Pattern

Replicating the proven architecture from [DataMind AI](https://github.com/MrunalNikam17/datamind-ai), `src/utils/llm.py` cascades through multiple top-performing free-tier models on OpenRouter:

1. `nvidia/nemotron-3-super-120b-a12b:free`
2. `cohere/north-mini-code:free`
3. `liquid/lfm-2.5-2.6b:free`
4. `google/gemma-4-31b-it:free`
5. `google/gemma-4-26b-a4b-it:free`
6. `poolside/laguna-s-2.1:free`
7. `nvidia/nemotron-3.5-lightning:free`
8. `nvidia/nemotron-3-ultra-550b-a55b:free`

If any model encounters rate limits (`HTTP 429`), timeouts, or service errors, the caller automatically falls back to the next available model in real time. If all models are unavailable or no API key is provided, the system falls back to deterministic rule sets and template synthesis without crashing.

---

## 📄 License
MIT License. Built for small business financial resilience.
