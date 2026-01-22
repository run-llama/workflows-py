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

## Test Coverage

### Agent Tests
- `test_function_agent.py` - FunctionAgent execution, tools, max iterations, early stopping
- `test_react_agent.py` - ReActAgent execution, retry on parse errors, memory
- `test_multi_agent_workflow.py` - Multi-agent orchestration, handoff, event streaming

### Workflow Feature Tests
- `test_workflow_features.py` - Core workflow features used with agents:
  - `wait_for_event` inside tool functions (human-in-the-loop pattern)
  - Context state management via `ctx.store`
  - Pickle serialization for workflow pause/resume
  - Multiple sequential tool calls with shared state

## Running Tests

```bash
uv run --directory packages/llama-index-integration-tests pytest
```

## Note

This package is not published. It is only used for testing purposes during development.
