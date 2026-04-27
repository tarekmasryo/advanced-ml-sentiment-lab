# Contributing

This repository is maintained as a focused ML application. Small, targeted contributions are preferred.

## Useful contributions

- Bug reports with reproduction steps.
- Test improvements.
- Documentation fixes.
- Small refactors that preserve behavior.
- UX improvements that reduce friction without adding visual noise.

## Development

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m streamlit run app.py
```

For a quick local check, run:

```powershell
python scripts/doctor.py --check
```
