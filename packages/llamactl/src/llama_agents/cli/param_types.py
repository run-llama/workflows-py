# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import asyncio
from typing import Any

import click
from click.shell_completion import CompletionItem
from llama_agents.cli.completion_cache import (
    _env_hash,
    read_cache,
)

# Template IDs and descriptions, extracted from init.py for static completion.
TEMPLATES: list[dict[str, str]] = [
    {"id": "basic-ui", "help": "A basic starter workflow with a React Vite UI"},
    {"id": "showcase", "help": "Workflow and UI patterns collection"},
    {"id": "document-qa", "help": "Upload documents and run question answering"},
    {"id": "extraction-review", "help": "Extract data from documents with review UI"},
    {"id": "classify-extract-sec", "help": "Classify and extract SEC filings"},
    {"id": "extract-reconcile-invoice", "help": "Extract and reconcile invoice data"},
    {"id": "basic", "help": "A base example showcasing workflow usage patterns"},
    {"id": "document_parsing", "help": "Parse documents with LlamaParse"},
    {"id": "human_in_the_loop", "help": "Human in the loop workflow patterns"},
    {"id": "invoice_extraction", "help": "Extract invoice details with LlamaExtract"},
    {"id": "rag", "help": "Simple RAG pipeline"},
    {"id": "web_scraping", "help": "Scrape and summarize web content"},
]


def _safe_complete(fn: Any) -> list[CompletionItem]:
    """Wrap a completion function so exceptions return empty rather than crashing the shell."""
    try:
        return fn()
    except Exception:
        return []


def _fetch_deployments() -> list[dict[str, str]]:
    """Fetch deployment list from API. Called by refresh_cache."""
    from llama_agents.cli.client import get_project_client

    client = get_project_client()
    deployments = asyncio.run(client.list_deployments())
    return [{"id": d.id, "help": f"{d.display_name} — {d.status}"} for d in deployments]


def _fetch_projects() -> list[dict[str, str]]:
    """Fetch project list from API. Called by refresh_cache."""
    from llama_agents.cli.client import get_control_plane_client

    client = get_control_plane_client()
    projects = asyncio.run(client.list_projects())
    return [
        {
            "id": p.project_id,
            "help": f"{p.project_name} ({p.deployment_count} deployments)",
        }
        for p in projects
    ]


def _fetch_deployment_history(deployment_id: str) -> list[dict[str, str]]:
    """Fetch deployment history for git SHA completion."""
    from llama_agents.cli.client import get_project_client

    client = get_project_client()

    async def _fetch() -> Any:
        return await client.get_deployment_history(deployment_id)

    history = asyncio.run(_fetch())
    return [
        {"id": item.git_sha, "help": item.released_at.isoformat()}
        for item in history.history
    ]


class DeploymentType(click.ParamType):
    name = "deployment"

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        def _complete() -> list[CompletionItem]:
            eh = _env_hash()
            items = read_cache("deployments", eh)
            lower = incomplete.lower()
            return [
                CompletionItem(d["id"], help=d.get("help", ""))
                for d in items
                if lower in d["id"].lower() or lower in d.get("help", "").lower()
            ]

        return _safe_complete(_complete)


class ProfileType(click.ParamType):
    name = "profile"

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        def _complete() -> list[CompletionItem]:
            from llama_agents.cli.config.env_service import service

            auth_svc = service.current_auth_service()
            profiles = auth_svc.list_profiles()
            lower = incomplete.lower()
            return [
                CompletionItem(p.name, help=p.api_url)
                for p in profiles
                if lower in p.name.lower()
            ]

        return _safe_complete(_complete)


class ProjectType(click.ParamType):
    name = "project"

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        def _complete() -> list[CompletionItem]:
            eh = _env_hash()
            items = read_cache("projects", eh)
            lower = incomplete.lower()
            return [
                CompletionItem(p["id"], help=p.get("help", ""))
                for p in items
                if lower in p["id"].lower() or lower in p.get("help", "").lower()
            ]

        return _safe_complete(_complete)


class EnvironmentType(click.ParamType):
    name = "environment"

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        def _complete() -> list[CompletionItem]:
            from llama_agents.cli.config.env_service import service

            envs = service.list_environments()
            current = service.get_current_environment()
            lower = incomplete.lower()
            return [
                CompletionItem(
                    e.api_url,
                    help="(current)" if e.api_url == current.api_url else "",
                )
                for e in envs
                if lower in e.api_url.lower()
            ]

        return _safe_complete(_complete)


class TemplateType(click.ParamType):
    name = "template"

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        def _complete() -> list[CompletionItem]:
            lower = incomplete.lower()
            return [
                CompletionItem(t["id"], help=t["help"])
                for t in TEMPLATES
                if lower in t["id"].lower()
            ]

        return _safe_complete(_complete)


class GitShaType(click.ParamType):
    name = "git_sha"

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        def _complete() -> list[CompletionItem]:
            deployment_id = ctx.params.get("deployment_id")
            if not deployment_id:
                return []
            eh = _env_hash()
            cache_key = f"git_sha_{deployment_id}"
            items = read_cache(cache_key, eh)
            lower = incomplete.lower()
            return [
                CompletionItem(s["id"], help=s.get("help", ""))
                for s in items
                if lower in s["id"].lower()
            ]

        return _safe_complete(_complete)
