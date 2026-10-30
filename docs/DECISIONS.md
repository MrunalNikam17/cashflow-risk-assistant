# Decisions

- Kept the Python compute-first engine as the primary source of truth and offline fallback.
- Retained the Streamlit dashboard as a secondary demo path rather than deleting it, because the repo still contains a working Python analytics foundation and the full user-facing app layer is not present yet.
- Chose deterministic date-relative synthetic data generation so the sample files remain reproducible without internet access.
- Kept LLM calls optional and fail-safe: when no API key is present, the system stays functional and uses deterministic explanations.
- Deferred the FastAPI backend and React/Vite frontend to a later scaffold because the current repo snapshot only includes the engine and tests.
