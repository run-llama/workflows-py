# LlamaAgents

[![Unit Testing](https://github.com/run-llama/workflows/actions/workflows/test.yml/badge.svg)](https://github.com/run-llama/workflows/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/run-llama/workflows/badge.svg?branch=main)](https://coveralls.io/github/run-llama/workflows?branch=main)
[![GitHub contributors](https://img.shields.io/github/contributors/run-llama/workflows)](https://github.com/run-llama/llama-index-workflows/graphs/contributors)


[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-index-workflows)](https://pypi.org/project/llama-index-workflows/)
[![Discord](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![Twitter](https://img.shields.io/twitter/follow/llama_index)](https://x.com/llama_index)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)

An open-source framework for building and shipping document-centric agents in Python.

Real document workflows are messy. You're stitching together OCR, LLMs, structured extraction, classification, custom validation, and human review into pipelines that have to run reliably in production. The steps are slow, the payloads are heavy, and a lot of the work is in-process Python: embedding models, image analysis, vision calls, custom heuristics that don't want to be a microservice. Standing up a real durable orchestration layer for that kind of workload is a project on its own, so most teams end up shoving the pipeline into a side process that nobody else wants to integrate with.

LlamaAgents is built around that reality. The core is [**Agent Workflows**](./packages/llama-index-workflows/), an event-driven orchestration library where steps are async Python functions that emit and consume events. Branch, loop, parallelize, persist state, recover from failures, all in plain Python with no DSL.

Then there are a few ways to actually run it.

## Use it as a library

The simplest path. `pip install llama-index-workflows`, define your workflow, and `await workflow.run(...)` from anywhere: async scripts, FastAPI, notebooks. No server, no infra, just a Python library you call.

```python
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent

class HelloWorkflow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=f"Hello, {ev.name}")
```

## Mount it inside an app you already have

[**`llama-agents-server`**](./packages/llama-agents-server/) wraps any workflow as a REST API with streaming, persistence, and human-in-the-loop support. Drop it into an existing Starlette/FastAPI app, or run it standalone. [**`llama-agents-client`**](./packages/llama-agents-client/) is the matching async client for calling workflows from other services.

```python
from llama_agents.server import WorkflowServer

server = WorkflowServer()
server.add_workflow("greet", HelloWorkflow())
```

## Or ship it as a deployable agent

[**`llamactl`**](./packages/llamactl/) is the CLI for building and deploying agent apps end-to-end. Init from a starter, develop locally with hot reload, then deploy to LlamaParse, AWS Bedrock AgentCore, or your own infra. Agents can be headless workflow services, MCP servers, or full-stack apps with a UI, whatever your agent needs to be.

```bash
uv tool install llamactl
llamactl init
llamactl serve
llamactl deployments create
```

## Works with LlamaParse

The heavy document primitives (OCR, structured extraction, classification, splitting) are what [LlamaParse](https://cloud.llamaindex.ai) is for. Plug them into your workflow as steps, let LlamaParse handle the document understanding, and keep your agent code focused on orchestration, business logic, and review.
