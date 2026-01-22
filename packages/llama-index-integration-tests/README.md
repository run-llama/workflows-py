# LlamaIndex Integration Tests

This package contains integration tests to validate that llama-index-core workflow abstractions work correctly with the workflows package.

## Purpose

The llama-index-core package uses several workflow features:
- `ctx.store.get/set` for state management
- `ctx.wait_for_event` for human-in-the-loop patterns
- Event streaming via `stream_events()`
- Context serialization for pause/resume

These tests ensure updates to the workflows package don't break these integration points.

## Test Organization

Tests are organized by **workflow feature**, not agent type. Each test is parameterized to run against both `FunctionAgent` and `ReActAgent`.

| File | Coverage |
|------|----------|
| `test_context_store.py` | `ctx.store.get/set`, state persistence, tool access to state |
| `test_event_streaming.py` | `stream_events()`, event types, tool call events |
| `test_human_in_the_loop.py` | `wait_for_event`, context serialization, pause/resume |
| `test_error_handling.py` | Max iterations, early stopping, tool errors |

## Running Tests

```bash
uv run --directory packages/llama-index-integration-tests pytest
```

## Note

This package is not published. It exists only for integration testing during development.
