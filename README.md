# LlamaAgents

[![Unit Testing](https://github.com/run-llama/workflows/actions/workflows/test.yml/badge.svg)](https://github.com/run-llama/workflows/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/run-llama/workflows/badge.svg?branch=main)](https://coveralls.io/github/run-llama/workflows?branch=main)
[![GitHub contributors](https://img.shields.io/github/contributors/run-llama/workflows)](https://github.com/run-llama/llama-index-workflows/graphs/contributors)


[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-index-workflows)](https://pypi.org/project/llama-index-workflows/)
[![Discord](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![Twitter](https://img.shields.io/twitter/follow/llama_index)](https://x.com/llama_index)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)

**Build production document agents in Python.**

Document AI workflows are messy. You're chaining OCR, LLMs, classification, extraction, custom validation, and human review into pipelines that need to run reliably at scale. They're slow, deal with heavy payloads, and need to be observable so you can tighten the quality loop over time.

LlamaAgents is an open-source agent framework and app server for building these. Define your logic as durable, event-driven Agent Workflows in Python, then run them however fits your stack — as a library, mounted in an existing server, or as a deployable agent app.

## Agent Workflows — the core

Everything in this repo is built on [**Agent Workflows**](./packages/llama-index-workflows/), a battle-tested workflow engine extracted from `llama_index`. Steps are async functions that emit and consume events. Branch, loop, run in parallel, persist state, and recover from failures — all in plain Python.

```python
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent

class HelloWorkflow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=f"Hello, {ev.name}")
```

```bash
pip install llama-index-workflows
```

## Three ways to run them

### As a library

Drop Agent Workflows directly into your existing Python app — async scripts, FastAPI, notebooks, anywhere. `pip install llama-index-workflows` and call `await workflow.run(...)`.

### As an HTTP server you mount in your app

[**`llama-agents-server`**](./packages/llama-agents-server/) wraps any workflow as a REST API with streaming, persistence, and human-in-the-loop support. Mount it inside your existing Starlette/FastAPI app, or run it standalone. Pair with [**`llama-agents-client`**](./packages/llama-agents-client/) to call workflows from other services.

```python
from llama_agents.server import WorkflowServer

server = WorkflowServer()
server.add_workflow("greet", HelloWorkflow())
```

### As a deployable agent app

[**`llamactl`**](./packages/llamactl/) is the CLI for building and deploying agent apps. Initialize from a starter, develop locally with hot reload, then deploy to LlamaParse, AWS Bedrock AgentCore, or self-host. Apps can be headless workflow services, MCP servers, or include a UI — whatever your agent needs to be.

```bash
uv tool install llamactl
llamactl init
llamactl serve
llamactl deployments create
```

## Works with LlamaParse

LlamaAgents pairs naturally with [LlamaParse](https://cloud.llamaindex.ai) for production-grade document parsing, extraction, and classification. Plug LlamaParse primitives into your workflows as steps and let LlamaParse handle the heavy document processing — OCR, structured extraction, classification, splitting — while your agent handles orchestration, business logic, and human-in-the-loop review.
