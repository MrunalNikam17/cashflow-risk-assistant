# Progress

## Resume check
- Repo status: inspection-only; no git state changes were made.
- Baseline suite: 33 passed, backend coverage 90%.
- Frontend validation: lint passed, Vitest passed (2/2), production build passed.

## Checklist

- Phase 0 / repo resume: DONE
  - Verified repo structure and engine/test baseline without changing git state.
- Phase 1: Engine correctness in src/: DONE
  - loader.py: DONE — status-aware invoice handling, de-duplication, validation, expected receivables behavior.
  - forecaster.py: DONE — recurring anchors, uncertainty bands, backtest metrics, what-if scenario support.
  - insight_agent.py: DONE — generic window parsing, recurring cost + purchase timing logic, guardrail fallback.
  - synthetic data: DONE — deterministic, date-relative sample generation.
- Phase 2: Backend API: PARTIAL
  - 2.1 Scaffolding: DONE — FastAPI app, config, schemas, in-memory store, routers/services, CORS, rate limits, docs, error handling.
  - 2.2 Session store: DONE — TTL + max-session in-memory store with no persistence.
  - 2.3 Endpoints: DONE for core slice — health, upload, sample, summary, transactions, forecast, anomalies, ask, scenario, export.
  - 2.4 Tests: DONE — pytest + TestClient endpoint coverage for happy + bad input paths.
  - 2.5 Requirements: DONE — backend requirements pinned and README run instructions updated in the local repo snapshot.
- Phase 3: Frontend: PARTIAL
  - 3.1 Scaffold: DONE — Vite + React + Tailwind + typed API client + INR formatting + dark mode shell.
  - 3.2 Upload page: PARTIAL — UI shell exists, but full drag-and-drop/real upload flow and cold-start UX are limited.
  - 3.3 Dashboard: DONE — KPI cards and charts are present with fallback data.
  - 3.4 Forecast page: DONE — forecast toggle, chart, model accuracy card, floors and band placeholders.
  - 3.5 Anomalies: DONE — sortable-style table and severity labels present.
  - 3.6 Ask Advisor: DONE — prompt chips, verdict card, numbers, explanation source badge.
  - 3.7 Ledger: DONE — table + export buttons in a basic shell.
  - 3.8 Polish: PARTIAL — accessibility/layout baseline exists; full production polish remains to be expanded.
  - 3.9 Tests: PARTIAL — basic Vitest coverage is passing; e2e / broader UI coverage not completed in this snapshot.
- Phase 4: DevOps and deployment files: PARTIAL
  - 4.1 Docker: DONE — backend Dockerfile, .dockerignore, docker-compose scaffold created.
  - 4.2 Render/Vercel env + docs: DONE — render.yaml and frontend/vercel.json added; env examples created.
  - 4.3 CI workflow: DONE — .github/workflows/ci.yml file created but not pushed.
  - 4.4 Makefile / pre-commit / lockfile / ruff config: DONE — files created; package-lock present.
  - 4.5 Streamlit note: DONE — decision captured in docs/DECISIONS.md.
  - 4.6 .gitignore: PARTIAL — base patterns covered and runtime dirs included, but not all generated artifact exclusions are exhaustive.
- Phase 5: Documentation and portfolio polish: PARTIAL
  - 5.1 README: PARTIAL — the repo has a substantial README, but it is not yet a complete live-demo/portfolio-ready final rewrite with all placeholders and exact deployment details.
  - 5.2 Docs: DONE — docs/ARCHITECTURE.md and docs/API.md added; LICENSE present.

## Needs manual action
- Create a real OpenRouter API key in the target environment before using the LLM path in production.
- Review the frontend API contracts and replace the current fallback/mock data with live backend wiring when the app is used beyond local demos.
- Connect the app to Render and Vercel, then set ALLOWED_ORIGINS and VITE_API_BASE_URL in the deployment environment.
- Add screenshots, verify the final live URL, and publish the final README when you are ready to push the repo to GitHub.

## Current status
The compute-first Python engine is operating correctly, the backend API slice is working, and the frontend scaffold has successfully passed lint/test/build checks. The remaining gaps are the final deployment polish, extended production UI coverage, and the final live-demo README rewrite expected by the original long prompt.
