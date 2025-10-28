---
title: Overview
sidebar:
  order: 1
---

## LlamaAgents at a Glance

LlamaAgents helps you build, serve and deploy small, workflow‑driven agentic apps using LlamaIndex, locally and on LlamaCloud. Define LlamaIndex [workflows](/python/llamaagents/workflows), run them as durable APIs that can pause for input, optionally add a UI, and deploy to LlamaCloud in seconds.

LlamaAgents is for developers and teams building automation, internal tools, and app‑like agent experiences without heavy infrastructure work.

Build and ship small, focused agentic apps—fast. Start from either our templated LlamaIndex workflow apps, or from workflows you've already prototyped, iterate locally, and deploy to LlamaCloud right from your terminal in seconds.

- Write [LlamaIndex Python workflows](/python/llamaagents/workflows/), and serve them as an API. For example, make a request to process incoming files, analyze them, and return the result or forward them on to another system.
- Workflow runs are durable, and can wait indefinitely for human or other external inputs before proceeding.
- Optionally add a UI for user-driven applications. Make custom chat applications, data extraction and review applications.
- Deploy your app in seconds to LlamaCloud. Call it as an API with your API key, or visit it secured with your LlamaCloud login.

LlamaAgents is built on top of [LlamaIndex Workflows](/python/llamaagents/workflows/). Use the [`llamactl`](/llamaagents/llamactl/getting-started) command line interface (CLI) to scaffold projects (`llamactl init`), run them locally (`llamactl serve`), and deploy to LlamaCloud (`llamactl deployments create`).

In addition to LlamaAgents, LlamaIndex publishes additional SDKs to facilitate rapid development:
- Our `llama-cloud-services` JS and Python SDKs offer a simple way to persist ad hoc Agent Data. [Read more here](/python/llamaagents/llamactl/agent-data-overview).
- Our `@llamaindex/ui` React library offers off-the-shelf components and hooks to facilitate developing workflow-driven UIs.
