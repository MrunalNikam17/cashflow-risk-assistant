# API Notes

This repository snapshot currently centers on the Python engine rather than a deployed REST API. The core functions are exposed through modules such as:

- src.ingestion.loader
- src.forecasting.forecaster
- src.insight.insight_agent
- src.utils.llm

The app remains offline-safe: if no OpenRouter key is provided, deterministic fallbacks keep the system workable without crashing.
