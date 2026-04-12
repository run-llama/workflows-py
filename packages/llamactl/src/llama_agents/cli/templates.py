# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GithubTemplateRepo:
    url: str


@dataclass
class TemplateOption:
    id: str
    name: str
    description: str
    source: GithubTemplateRepo
    llama_cloud: bool


UI_TEMPLATES = [
    TemplateOption(
        id="basic-ui",
        name="Basic UI",
        description="A basic starter workflow with a React Vite UI",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-basic-ui"
        ),
        llama_cloud=False,
    ),
    TemplateOption(
        id="showcase",
        name="Showcase",
        description="A collection of workflow and UI patterns to build LlamaDeploy apps",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-showcase"
        ),
        llama_cloud=False,
    ),
    TemplateOption(
        id="document-qa",
        name="Document Question & Answer",
        description="Upload documents and run question answering through a React UI",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-document-qa"
        ),
        llama_cloud=True,
    ),
    TemplateOption(
        id="extraction-review",
        name="Extraction Agent with Review UI",
        description="Extract data from documents using a custom schema and Llama Cloud. Includes a UI to review and correct the results",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-data-extraction"
        ),
        llama_cloud=True,
    ),
    TemplateOption(
        id="classify-extract-sec",
        name="SEC Insights",
        description="Upload SEC filings, classifying them to the appropriate type and extracting key information",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-classify-extract-sec"
        ),
        llama_cloud=True,
    ),
    TemplateOption(
        id="extract-reconcile-invoice",
        name="Invoice Extraction & Reconciliation",
        description="Extract and reconcile invoice data against contracts",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-extract-reconcile-invoice"
        ),
        llama_cloud=True,
    ),
]

HEADLESS_TEMPLATES = [
    TemplateOption(
        id="basic",
        name="Basic Workflow",
        description="A base example that showcases usage patterns for workflows",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-basic"
        ),
        llama_cloud=False,
    ),
    TemplateOption(
        id="document_parsing",
        name="Document Parser",
        description="A workflow that, using LlamaParse, parses unstructured documents and returns their raw text content",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-document-parsing"
        ),
        llama_cloud=True,
    ),
    TemplateOption(
        id="human_in_the_loop",
        name="Human in the Loop",
        description="A workflow showcasing how to use human in the loop with LlamaIndex workflows",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-human-in-the-loop"
        ),
        llama_cloud=False,
    ),
    TemplateOption(
        id="invoice_extraction",
        name="Invoice Extraction",
        description="A workflow that, given an invoice, extracts several key details using LlamaExtract",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-invoice-extraction"
        ),
        llama_cloud=True,
    ),
    TemplateOption(
        id="rag",
        name="RAG",
        description="A workflow that embeds, indexes and queries your documents on the fly, providing you with a simple RAG pipeline",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-rag"
        ),
        llama_cloud=False,
    ),
    TemplateOption(
        id="web_scraping",
        name="Web Scraping",
        description="A workflow that, given several urls, scrapes and summarizes their content using Google's Gemini API",
        source=GithubTemplateRepo(
            url="https://github.com/run-llama/template-workflow-web-scraping"
        ),
        llama_cloud=False,
    ),
]

ALL_TEMPLATES = UI_TEMPLATES + HEADLESS_TEMPLATES
