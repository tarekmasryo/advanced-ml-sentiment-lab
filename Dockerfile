# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11-slim

# ---------- Base (deps only, cached) ----------
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Create non-root user early (stable UID helps on some hosts)
RUN useradd -m -u 10001 appuser

# Install deps (cache-friendly)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -U pip && \
    python -m pip install -r requirements.txt


# ---------- Test stage ----------
FROM base AS test

COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -r requirements-dev.txt

COPY . .
RUN python -m ruff check . && \
    python -m ruff format --check . && \
    python -m pytest -q


# ---------- Runtime stage ----------
FROM base AS runtime

# Copy app as non-root owner (avoid chown -R)
COPY --chown=appuser:appuser . /app

USER appuser

# Defaults (overrideable)
ENV HOST=0.0.0.0 \
    PORT=8501 \
    APP_FILE=app.py

EXPOSE 8501

# Optional healthcheck (keeps it generic enough)
# HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
#   CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', int('${PORT}'))); s.close()"

CMD ["sh", "-lc", "python -m streamlit run ${APP_FILE} --server.address=${HOST} --server.port=${PORT} --server.headless=true"]
