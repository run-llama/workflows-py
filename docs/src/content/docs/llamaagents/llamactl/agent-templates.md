---
title: Agent Templates
sidebar:
  order: 2
---

We provide a set of agent templates, both as full-stack apps with a UI, and as headless servers. These templates aim to cover a number of most common use-cases, and are intended as a starting point for your own custom agents.

You can pull these templates via our [llamactl CLI](/python/llamaagents/llamactl/getting-started/), which will create a local repository for your agent project, equipped with all the source-code and UI elements.

> This list of agents is not final, expect changes and updates along the way.

### Available LlamaAgent Templates

There are two groups of templates, with and without UI elements. You can see the most up-to-date list of templates by [initializing an agent project with `llamactl`](/python/llamaagents/llamactl/getting-started/#initialize-a-project).

Below is a full list of templates available via the `llamactl` CLI. We also provide [Click to Deploy templates from within LlamaCloud](python/llamaagents/llamactl/click-to-deploy/).

| **Template With UI**  | |
| --- | --- |
| **Template Name** | **Description** |
| Basic UI  | A minimal starting point for building an event-driven, async-first agent with UI using LlamaAgent workflows and LlamaDeploy. Includes a Vite app that calls the local agent workflow server via API requests. |
| Showcase | A showcase application demonstrating different agent workflow patterns and capabilities with a full-stack setup. Includes agent workflow sources and a Vite UI app that communicates with the workflow server via API requests.  |
| Document Q&A | A document question-answering application built with LlamaAgent workflows and LlamaCloud services like the LlamaCloud Index. Allows users to upload documents and ask questions about their content.  |
| Extraction Agent with Review UI | A data extraction and ingestion application with a review UI for validating and editing extracted data. Uses LlamaExtract agents to extract structured data from documents, stores results in Agent Data, and provides a dynamic UI that generates editing interfaces based on the extraction schema. |
| Invoice Extraction & Reconciliation | Extracts structured data from invoices and reconciles it against contract documents using LlamaExtract and LlamaCloud Index. Helps finance and operations teams validate that incoming invoices comply with agreed contract terms by automatically detecting mismatches in payment terms, totals, and other key fields. |
| **Template with Headless Workflows  (No UI)**  | |
| **Template Name** | **Description** |
| Basic Workflow | A minimal starting point for building an event-driven, async-first agent using LlamaAgent workflows. Provides a simple hello-world example that can be extended with custom steps and logic.  |
| Document Parser | A document parsing agent template for processing and extracting information from complex documents using LlamaAgent workflows and LlamaParse.  |
| Human in the Loop | A LlamaAgent workflow template that incorporates human-in-the-loop interactions, allowing agents to pause and wait for human input or approval at specific steps. |
| Invoice Extraction | An agent template for extracting structured data from invoices using LlamaIndex workflows and extraction agents. |
| RAG | A Retrieval-Augmented Generation (RAG) agent template for building question-answering systems that retrieve relevant context from indexed documents before generating responses.  |
| Web Scraping | A web scraping agent workflow template for extracting data from web pages using LlamaIndex workflows and web scraping capabilities.  |

### Coding Agent Support Files

Each template downloaded/cloned with `llamactl` will also come with coding agent files like `CLAUDE.md`, `GEMINI.md` and `AGENTS.md`, designed to help developing and customizing LlamaAgents with the assistance of coding agents simpler. These files contain all the relevant context and information that coding agents such as Cursor, Claude Code etc might need.
