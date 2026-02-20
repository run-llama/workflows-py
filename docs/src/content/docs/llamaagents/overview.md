---
title: Overview
sidebar:
  order: 1
---

## LlamaAgents at a Glance

LlamaAgents is the most advanced way to build **agent workflows**. Author and run **multi-step document agents** from scratch locally using our open-source [Agent Workflows](/python/llamaagents/workflows/), or build and deploy them in the cloud with our vibe-coding [**Agent Builder**](/python/llamaagents/cloud/builder/) in [LlamaCloud](https://cloud.llamaindex.ai/)—without wiring up infrastructure, persistence, or deployment yourself.

Stitch together Parse, Extract, Split, Classify, and custom operations into [Workflows](/python/llamaagents/workflows/) that perform knowledge tasks on your documents. When you need full control, it's real Python underneath: fork and extend without a rewrite. Agent Workflows give you event-driven orchestration with branching, parallelism, [human-in-the-loop](/python/llamaagents/workflows/human-in-the-loop/) review, durability, and [observability](/python/llamaagents/workflows/observability/).

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1.5rem;">
  <iframe
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/0Zhf5z2Onjs" title="LlamaAgents overview" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></>
</div>

### Get Started

- **Build locally**: Use the [`llamactl` CLI](/python/llamaagents/llamactl/getting-started/) to create projects from [starter templates](/python/llamaagents/llamactl-reference/commands-init/), develop and serve workflows on your machine, then deploy to LlamaCloud or self-host. You can also use [Agent Workflows](/python/llamaagents/workflows/) directly in your own Python applications—run them as async processes or [mount them as endpoints](/python/llamaagents/workflows/deployment/) in your existing server.

- **Build in the cloud**: Use [**Agent Builder**](/python/llamaagents/cloud/builder/) in [LlamaCloud](https://cloud.llamaindex.ai/) (Agents → Builder) to describe your workflow in plain language; an AI coding agent generates a complete, deployable workflow. The code is yours—customize it in GitHub or run it on your own infrastructure. For a one-click path, [click-to-deploy a starter template](/python/llamaagents/llamactl/click-to-deploy/) like SEC Insights or Invoice Matching.

- **Go deeper**: Combine local development with cloud services. Use [Agent Workflows](/python/llamaagents/workflows/) for orchestration and [WorkflowClient](/python/llamaagents/workflows/deployment/#using-workflowclient-to-interact-with-servers) to call deployed workflows via REST or the typed Python client.

### Components

**[`llamactl` CLI](/python/llamaagents/llamactl/getting-started/)**: Development and deployment for local workflow apps. Initialize from [starter templates](/python/llamaagents/llamactl-reference/commands-init/), serve locally, and deploy to LlamaCloud or export for self-hosting.

**[Agent Workflows](/python/llamaagents/workflows/)**: The event-driven orchestration framework at the core. Use it as an async library in your own code, or let `llamactl` serve it. Built-in durability and [observability](/python/llamaagents/workflows/observability/).

**[Agent Builder](/python/llamaagents/cloud/builder/)**: In [LlamaCloud](https://cloud.llamaindex.ai/) → **Agents** → **Builder**. Natural-language, vibe-coding interface to create document workflows; the agent generates real Python you can deploy or take to GitHub.

**[`llama-cloud-services`](/python/cloud/)**: LlamaCloud document primitives (Parse, Extract, Classify), [Agent Data](/python/llamaagents/cloud/agent-data-overview/) for structured storage, and vector indexes. `llamactl` handles authentication when deploying to the cloud.

**[@llamaindex/ui](/python/llamaagents/llamactl/ui-hooks/)**: React hooks for workflow-powered frontends. Deploy alongside your backend with `llamactl`.

**[Workflows Client](/python/llamaagents/workflows/deployment/#using-workflowclient-to-interact-with-servers)**: Call deployed workflows via REST API or typed Python client.
