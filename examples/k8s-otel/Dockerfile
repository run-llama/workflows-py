# syntax=docker/dockerfile:1
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /src

# 1) Copy only dependency metadata — cached until lock/pyproject files change
COPY pyproject.toml uv.lock ./
COPY packages/llama-index-workflows/pyproject.toml packages/llama-index-workflows/pyproject.toml
COPY packages/llama-index-utils-workflow/pyproject.toml packages/llama-index-utils-workflow/pyproject.toml
COPY packages/llama-agents-server/pyproject.toml packages/llama-agents-server/pyproject.toml
COPY packages/llama-agents-client/pyproject.toml packages/llama-agents-client/pyproject.toml
COPY packages/llama-agents-dbos/pyproject.toml packages/llama-agents-dbos/pyproject.toml
COPY packages/llama-agents-integration-tests/pyproject.toml packages/llama-agents-integration-tests/pyproject.toml
COPY examples/k8s-otel/pyproject.toml examples/k8s-otel/pyproject.toml

# 2) Install all dependencies (cached unless lock/pyproject files change)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable --package k8s-otel-example --no-install-workspace

# 3) Copy source and install workspace packages only (fast — deps already present)
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable --package k8s-otel-example

ENV PATH="/src/.venv/bin:$PATH"

ENTRYPOINT ["python", "examples/k8s-otel/app.py"]
