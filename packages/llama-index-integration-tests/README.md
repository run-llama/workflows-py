# LlamaIndex Integration Tests

This package contains integration tests to validate that llama-index-core workflow abstractions continue to work correctly with the workflows package.

## Purpose

The llama-index-core package has several workflow abstractions that subclass `Workflow` from this package:

- `BaseWorkflowAgent` - Base class for workflow-based agents
- `FunctionAgent` - Function calling agent
- `ReActAgent` - ReAct (Reasoning + Acting) agent
- `AgentWorkflow` - Multi-agent orchestration workflow
- `CodeActAgent` - Agent that can execute Python code

These tests ensure that updates to the workflows package don't break these abstractions.

## Running Tests

```bash
uv run --directory packages/llama-index-integration-tests pytest
```

## Note

This package is not published. It is only used for testing purposes during development.
