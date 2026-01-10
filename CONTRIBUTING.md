# Contributing

This is a portfolio project. Issues and suggestions are welcome.

## Ways to contribute
- Report bugs (include steps to reproduce and logs).
- Suggest improvements (UX, performance, reliability).
- Propose small PRs (docs, tests, refactors).
- Security issues: please avoid posting sensitive details publicly.

## Dev setup

From the repo root (inside a virtual environment):

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m streamlit run app.py
