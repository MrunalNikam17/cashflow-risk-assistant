# Architecture

```mermaid
flowchart LR
A[Transactions / CSV uploads] --> B[Normalization + validation]
B --> C[Forecast engine]
B --> D[Anomaly detector]
C --> E[Risk verdict]
D --> E
E --> F[Explainable advisor]
F --> G[Streamlit / future API / UI]
```

The current repo focuses on the core deterministic engine:
- ingestion and validation
- cash-flow forecasting and uncertainty bands
- affordability guardrails
- deterministic fallback explanations
- sample-data generation

The backend/frontend deployment layers remain future work and are intentionally documented as TODO in the progress file.
