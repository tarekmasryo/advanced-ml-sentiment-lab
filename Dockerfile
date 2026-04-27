# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11-slim

FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN useradd -m -u 10001 appuser

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -r requirements.txt


FROM base AS test

COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -r requirements-dev.txt

COPY . .
RUN python -m ruff check . && \
    python -m ruff format --check . && \
    python -m pytest -q


FROM base AS runtime

COPY --chown=appuser:appuser . /app

USER appuser

ENV HOST=0.0.0.0 \
    PORT=8501 \
    APP_FILE=app.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import os, socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', int(os.getenv('PORT', '8501')))); s.close()"

CMD ["sh", "-lc", "python -m streamlit run ${APP_FILE} --server.address=${HOST} --server.port=${PORT} --server.headless=true"]
