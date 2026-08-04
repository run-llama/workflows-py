---
title: Overview
sidebar:
  order: 1
---

## LlamaAgents at a Glance

LlamaAgents is the most advanced way to build **agent workflows**. Author and run **multi-step document agents** with our open-source [Agent Workflows](/python/llamaagents/workflows/), then deploy them to LlamaCloud or your own infrastructure.

Stitch together Parse, Extract, Split, Classify, and custom operations into [Workflows](/python/llamaagents/workflows/) that perform knowledge tasks on your documents. When you need full control, it's real Python underneath: fork and extend without a rewrite. Agent Workflows give you event-driven orchestration with branching, parallelism, [human-in-the-loop](/python/llamaagents/workflows/human-in-the-loop/) review, durability, and [observability](/python/llamaagents/workflows/observability/).

### Get Started

- **Build locally**: Use the [`llamactl` CLI](/python/llamaagents/llamactl/getting-started/) to create projects from [starter templates](/python/llamaagents/llamactl-reference/commands-init/), develop and serve workflows on your machine, then deploy to LlamaCloud or self-host. You can also use [Agent Workflows](/python/llamaagents/workflows/) directly in your own Python applications—run them as async processes or [mount them as endpoints](/python/llamaagents/workflows/deployment/) in your existing server.

- **Deploy a starter template**: [Choose a template in LlamaCloud](/python/llamaagents/cloud/click-to-deploy/) and deploy it without using the command line.

- **Go deeper**: Combine local development with cloud services. Use [Agent Workflows](/python/llamaagents/workflows/) for orchestration and [WorkflowClient](/python/llamaagents/workflows/deployment/#using-workflowclient-to-interact-with-servers) to call deployed workflows via REST or the typed Python client.

### Components

**[`llamactl` CLI](/python/llamaagents/llamactl/getting-started/)**: Development and deployment for local workflow apps. Initialize from [starter templates](/python/llamaagents/llamactl-reference/commands-init/), serve locally, and deploy to LlamaCloud or export for self-hosting.

**[Agent Workflows](/python/llamaagents/workflows/)**: The event-driven orchestration framework at the core. Use it as an async library in your own code, or let `llamactl` serve it. Built-in durability and [observability](/python/llamaagents/workflows/observability/).

**[`llama-cloud-services`](/python/cloud/)**: LlamaCloud document primitives (Parse, Extract, Classify), [Agent Data](/python/llamaagents/cloud/agent-data-overview/) for structured storage, and vector indexes. `llamactl` handles authentication when deploying to the cloud.

**[@llamaindex/ui](/python/llamaagents/llamactl/ui-hooks/)**: React hooks for workflow-powered frontends. Deploy alongside your backend with `llamactl`.

**[Workflows Client](/python/llamaagents/workflows/deployment/#using-workflowclient-to-interact-with-servers)**: Call deployed workflows via REST API or typed Python client.
