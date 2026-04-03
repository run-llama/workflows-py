from __future__ import annotations

import json
import logging
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_agents.appserver.configure_logging import (
    _is_health_request,
    add_log_middleware,
    setup_logging,
)


@pytest.fixture()
def isolated_logging() -> Generator[None, None, None]:
    """Minimal isolation for logging and structlog state.

    - Snapshot root logger level/handlers/filters
    - Restore all after test
    """
    # Root logger snapshot
    root_logger = logging.getLogger()
    prev_root_level = root_logger.level
    prev_root_handlers = list(root_logger.handlers)
    prev_root_filters = list(root_logger.filters)

    try:
        yield
    finally:
        # Restore root
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        for f in list(root.filters):
            root.removeFilter(f)
        for h in prev_root_handlers:
            root.addHandler(h)
        for f in prev_root_filters:
            root.addFilter(f)
        root.setLevel(prev_root_level)


def test_setup_logging_json_filters_by_level_and_renders_json(
    isolated_logging: None,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    setup_logging(level="WARNING")

    logger = logging.getLogger("test")
    logger.info("info message", extra={"foo": 1})
    logger.warning("warn message", extra={"bar": 2})

    out, err = capfd.readouterr()
    lines = [line for line in out.splitlines() if line.strip()]

    # Only the WARNING should appear
    assert not any("info message" in line for line in lines)
    warn_lines = [line for line in lines if "warn message" in line]
    assert len(warn_lines) >= 1

    # Validate JSON content shape
    record = json.loads(warn_lines[-1])
    # Basic keys we expect from our processors
    assert "timestamp" in record
    assert record.get("event") == "warn message"
    # level casing depends on structlog; accept either
    level_val = record.get("level")
    assert level_val in {"warning", "WARN", "WARNING", "warn"}


def test_setup_logging_console_renders_human_readable(
    isolated_logging: None,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default format is console; ensure explicitly
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    setup_logging(level="INFO")

    logger = logging.getLogger("console-test")
    logger.info("hello world", extra={"answer": 42})

    out, err = capfd.readouterr()
    # Should not be JSON
    assert "hello world" in out
    assert not out.strip().startswith("{")


# ---------------------------------------------------------------------------
# _is_health_request unit tests
# ---------------------------------------------------------------------------


def _fake_request(path: str) -> MagicMock:
    req = MagicMock()
    req.url.path = path
    return req


@pytest.mark.parametrize(
    "path",
    ["/health", "/health/", "/healthz", "/livez", "/readyz", "/metrics"],
)
def test_is_health_request_matches(path: str) -> None:
    assert _is_health_request(_fake_request(path)) is True


@pytest.mark.parametrize(
    "path",
    ["/", "/deployments/myapp/workflows", "/health/extra"],
)
def test_is_health_request_rejects(path: str) -> None:
    assert _is_health_request(_fake_request(path)) is False


# ---------------------------------------------------------------------------
# access log middleware integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_app() -> FastAPI:
    """Minimal FastAPI app with access log middleware and a couple of routes."""
    test_app = FastAPI()

    @test_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @test_app.get("/other")
    def other() -> dict[str, str]:
        return {"data": "hello"}

    add_log_middleware(test_app)
    return test_app


def test_access_log_suppressed_for_health(
    isolated_logging: None,
    log_app: FastAPI,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    setup_logging(level="INFO")

    with TestClient(log_app) as client:
        client.get("/health")
        client.get("/other")

    out, _ = capfd.readouterr()
    access_lines = [line for line in out.splitlines() if "[app.access]" in line]
    assert any("/other" in line for line in access_lines)
    assert not any("/health" in line for line in access_lines)


def test_access_log_emitted_for_normal_routes(
    isolated_logging: None,
    log_app: FastAPI,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    setup_logging(level="INFO")

    with TestClient(log_app) as client:
        client.get("/other")

    out, _ = capfd.readouterr()
    assert "GET /other" in out
