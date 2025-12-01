---
title: Overview
sidebar:
  order: 1
---

## LlamaAgents at a Glance


LlamaAgents helps you build, serve and deploy workflow‑driven, document-centric agentic apps fast.

Agentic applications are difficult to develop: LLMs are slow and expensive. Pipelines are complex, non-deterministic. There's usually a need to keep a human in the loop to monitor and approve decisions or results. On top of this, the eco-system undergoes constant churn,
and changes in best practices.

LlamaAgents is a suite of tools to help you easily handle these challenges, while getting out of your way for the rest. At their core is our [Workflows](/python/llamaagents/workflows/) library, which is a powerful, event-driven framework for building agentic applications. It makes defining complex workflows easy, and durability simple and opt-in without the complicated infrastructure.

With Cloud LlamaAgents, you can easily develop and deploy your workflows as durable APIs and UIs on our cloud. **Click-to-deploy from predefined applications directly in [LlamaCloud](https://cloud.llamaindex.ai/).** Later, fork and customize with git, or develop and deploy from your own git repository with `llamactl`.

Llama Agents are applications composed of the following components:

**[Workflows](/python/llamaagents/workflows/)** - Define complex pipelines easily with our event-driven workflows. They can be as deterministic or dynamic as needed. They easily support loops, branching, and parallel execution. Pause for input from a human or other external system. You can easily embed them in your application as a simple async process, serve as an API, or embed them in your pre-existing python API.

**[`llamactl` cli](/python/llamaagents/llamactl/getting-started/)** - Develop and deploy Python workflow powered full-stack applications quickly with the `llamactl` CLI. Initialize from our templates with `llamactl init`, or develop from scratch. Just configure your `pyproject.toml` with your workflows, and serve them with `llamactl serve`. Manage your deployments to LlamaCloud with `llamactl deployments`, and then view them in [LlamaCloud](https://cloud.llamaindex.ai/).

**`llama-cloud-services`** - You can easily integrate with the rest of LlamaCloud's document focused primitives via `llama-cloud-services`, and use [agent data](/python/llamaagents/llamactl/agent-data-overview/) to persist your application state without managing the infrastructure.

Develop workflow powered react frontends, using our [@llamaindex/ui](https://developers.llamaindex.ai/python/llamaagents/llamactl/ui-hooks/) library. Deploy alongside your backend workflows with llamactl, or embed in your existing react application.

Use your deployed workflows as an API. You can call them as a REST API, or use our python [Workflows client](https://developers.llamaindex.ai/python/llamaagents/workflows/deployment/#using-workflowclient-to-interact-with-servers) to easily integrate with your server.
