.PHONY: backend-test backend-run frontend-test frontend-build frontend-dev docker-up lint format

backend-test:
	pytest -q

backend-run:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

frontend-test:
	cd frontend && npm run test

frontend-build:
	cd frontend && npm run build

frontend-dev:
	cd frontend && npm run dev

lint:
	pytest -q
	cd frontend && npm run lint

build: frontend-build

docker-up:
	docker compose up --build

format:
	python -m ruff check . --fix
	cd frontend && npm run lint -- --fix
