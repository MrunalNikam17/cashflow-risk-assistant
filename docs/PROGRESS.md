# Progress

## Resume check
- Repo status: working tree clean after the resumed patch set; engine fixes landed and tests pass.
- Baseline suite: 23 passed (pytest -q).

## Checklist

- Phase 0 / repo resume: DONE
- Phase 1: Engine correctness in src/: PARTIAL
  - loader.py: DONE for status-aware invoice handling, deduplication, validation.
  - forecaster.py: DONE for recurring anchor robustness, uncertainty bands, backtest structure, scenario analysis.
  - insight_agent.py: DONE for generic window parsing, guardrail, model fallback alias.
  - synthetic data: DONE for deterministic relative generation.
- Phase 2: Backend API: TODO
  - Full FastAPI service is not yet present in this repo revision.
- Phase 3: Frontend: TODO
  - Frontend app is not yet present in this repo revision.
- Phase 4: Deployment and DevOps: TODO
- Phase 5: Documentation and portfolio polish: PARTIAL
  - README already exists; new project docs were added in docs/.
  - Full live-demo, screenshots, and deployment guides remain to be completed.

## Needs manual action
- Add a real OpenRouter key in the target environment before using the LLM path.
- Add backend and frontend code and deployment secrets when the stack is scaffolded.
- Connect Render/Vercel and set ALLOWED_ORIGINS / VITE_API_BASE_URL in production.
- Publish final screenshots and live URL into the README.

## Current status
The repo currently contains the robust Python engine and test suite, but the full backend/frontend deployment layers described in the original brief are still future work for this codebase.
