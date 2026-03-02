# syntax=docker/dockerfile:1
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /src

# 1) Copy only dependency metadata — cached until lock/pyproject files change
COPY workflows-py/pyproject.toml workflows-py/uv.lock ./
COPY workflows-py/packages/llama-index-workflows/pyproject.toml packages/llama-index-workflows/pyproject.toml
COPY workflows-py/packages/llama-index-utils-workflow/pyproject.toml packages/llama-index-utils-workflow/pyproject.toml
COPY workflows-py/packages/llama-agents-server/pyproject.toml packages/llama-agents-server/pyproject.toml
COPY workflows-py/packages/llama-agents-client/pyproject.toml packages/llama-agents-client/pyproject.toml
COPY workflows-py/packages/llama-agents-dbos/pyproject.toml packages/llama-agents-dbos/pyproject.toml
COPY workflows-py/packages/llama-agents-integration-tests/pyproject.toml packages/llama-agents-integration-tests/pyproject.toml
COPY workflows-py/examples/k8s-otel/pyproject.toml examples/k8s-otel/pyproject.toml
# Local llama_index packages for development
COPY llama_index/llama-index-instrumentation \
     /llama_index/llama-index-instrumentation
COPY llama_index/llama-index-integrations/observability/llama-index-observability-otel \
     /llama_index/llama-index-integrations/observability/llama-index-observability-otel

# 2) Install all dependencies (cached unless lock/pyproject files change)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable --package k8s-otel-example --no-install-workspace

# 3) Copy source and install workspace packages only (fast — deps already present)
COPY workflows-py/ .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable --package k8s-otel-example

ENV PATH="/src/.venv/bin:$PATH"

ENTRYPOINT ["python", "examples/k8s-otel/app.py"]
