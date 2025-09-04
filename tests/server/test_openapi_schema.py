# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from workflows import Workflow

from workflows.server import WorkflowServer


def test_openapi_schema_includes_all_routes(simple_test_workflow: Workflow) -> None:
    server = WorkflowServer()
    server.add_workflow("test", simple_test_workflow)
    schema = server.openapi_schema()

    assert "paths" in schema
    paths = schema["paths"]

    expected = {
        "/workflows": {"get"},
        "/workflows/{name}/run": {"post"},
        "/workflows/{name}/run-nowait": {"post"},
        "/results/{handler_id}": {"get"},
        "/events/{handler_id}": {"get"},
        "/health": {"get"},
    }

    # Validate each expected path and method is present
    for path, methods in expected.items():
        assert path in paths, f"Missing path in schema: {path}"
        present_methods = {m.lower() for m in paths[path].keys()}
        for method in methods:
            assert method in present_methods, (
                f"Missing method for {path}: expected {method}, found {present_methods}"
            )
