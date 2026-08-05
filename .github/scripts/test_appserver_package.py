# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from fastapi.testclient import TestClient
from llama_agents.appserver.app import app


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json() == {"status": "Healthy"}

        workflows = client.get("/deployments/compatibility-test/workflows")
        assert workflows.status_code == 200, workflows.text
        assert workflows.json() == {"workflows": []}

        metrics = client.get("/metrics")
        assert metrics.status_code == 200, metrics.text
        assert b'apiserver_state{apiserver_state="running"} 1.0' in metrics.content
        assert b'handler="/deployments/{deployment_name}/workflows"' in metrics.content


if __name__ == "__main__":
    main()
