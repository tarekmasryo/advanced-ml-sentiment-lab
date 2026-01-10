.PHONY: help install dev run lint lint-fix format test check precommit

PY ?= python
APP_FILE ?= app.py

help:
	@echo "Targets:"
	@echo "  install      Install runtime deps (requirements.txt if present)"
	@echo "  dev          Install dev deps (requirements-dev.txt if present)"
	@echo "  run          Run app (Streamlit by default)"
	@echo "  lint         Run ruff lint"
	@echo "  lint-fix     Run ruff lint with fixes"
	@echo "  format       Format code with ruff"
	@echo "  check        Lint + format check + tests"
	@echo "  test         Run pytest"
	@echo "  precommit    Install pre-commit hooks"

install:
	$(PY) -m pip install -U pip
ifneq ($(wildcard requirements.txt),)
	$(PY) -m pip install -r requirements.txt
endif
	@$(PY) -m pip check || true

dev:
	$(PY) -m pip install -U pip
ifneq ($(wildcard requirements-dev.txt),)
	$(PY) -m pip install -r requirements-dev.txt
endif
	@$(PY) -m pip check || true

run:
	$(PY) -m streamlit run $(APP_FILE)

lint:
	$(PY) -m ruff check .

lint-fix:
	$(PY) -m ruff check . --fix --exit-non-zero-on-fix

format:
	$(PY) -m ruff format .

test:
	$(PY) -m pytest -q

check: lint
	$(PY) -m ruff format --check .
	$(PY) -m pytest -q

precommit:
	$(PY) -m pip install -U pre-commit
	pre-commit install
