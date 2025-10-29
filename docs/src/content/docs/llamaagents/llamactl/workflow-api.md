---
title: Serving your Workflows
sidebar:
  order: 5
---
:::caution
Cloud deployments of LlamaAgents is still in alpha. You can try it out locally, or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs)
:::
LlamaAgents runs your LlamaIndex workflows locally and in the cloud. Author your workflows, add minimal configuration, and `llamactl` wraps them in an application server that exposes them as HTTP APIs.

## Learn the basics (LlamaIndex Workflows)

LlamaAgents is built on top of LlamaIndex workflows. If you're new to workflows, start here: [LlamaIndex Workflows](/python/llamaagents/workflows).

## Author a workflow (quick example)

```python
# src/app/workflows.py
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

class QuestionFlow(Workflow):
    @step
    async def generate(self, ev: StartEvent) -> StopEvent:
        question = ev.question
        return StopEvent(result=f"Answer to {question}")

qa_workflow = QuestionFlow(timeout=120)
```

## Configure workflows for LlamaAgents to serve

The app server reads workflows configured in your `pyproject.toml` and makes them available under their configured names.

Define workflow instances in your code, then reference them in your config.

```toml
# pyproject.toml
[project]
name = "app"
# ...
[tool.llamadeploy.workflows]
answer-question = "app.workflows:qa_workflow"
```

## How serving works (local and cloud)

- `llamactl serve` discovers your config. See [`llamactl serve`](/python/cloud/llamaagents/llamactl-reference/commands-serve).
- The app server loads your workflows.
- HTTP routes are exposed under `/deployments/{name}`. In development, `{name}` defaults to your Python project name and is configurable. On deploy, you can set a new name; a short random suffix may be appended to ensure uniqueness.
- Workflow instances are registered under the specified name. For example, `POST /deployments/app/workflows/answer-question/run` runs the workflow above.
- If you configure a UI, it runs alongside your API (proxied in dev, static in preview). For details, see [UI build and dev integration](/python/cloud/llamaagents/ui-build).

During development, the API is available at `http://localhost:4501`. After you deploy to LlamaCloud, it is available at `https://api.cloud.llamaindex.ai`.

### Authorization

During local development, the API is unprotected. After deployment, your API uses the same authorization as LlamaCloud. Create an API token in the same project as the agent to make requests. For example:

```bash
curl 'https://api.cloud.llamaindex.ai/deployments/app-xyz123/workflows/answer-question/run' \
  -H 'Authorization: Bearer llx-xxx' \
  -H 'Content-Type: application/json' \
  --data '{"start_event": {"question": "What is the capital of France?"}}'
```

## Workflow HTTP API

When using a `WorkflowServer`, the app server exposes your workflows as an API. View the OpenAPI reference at `/deployments/<name>/docs`.

This API allows you to:
- Retrieve details about registered workflows
- Trigger runs of your workflows
- Stream published events from your workflows, and retrieve final results from them
- Send events to in-progress workflows (for example, HITL scenarios).

During development, visit `http://localhost:4501/debugger` to test and observe your workflows in a UI.
