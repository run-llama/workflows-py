import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_agents.appserver.routers.status import health_router


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(health_router)
    return TestClient(app)


@pytest.mark.parametrize("path", ["/health", "/healthz", "/livez", "/readyz"])
def test_health_endpoints(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.json() == {"status": "Healthy"}
