"""CLI commands for managing LlamaDeploy deployments.

This command group lets you list, create, edit, refresh, and delete deployments.
A deployment points the control plane at your Git repository and deployment file
(e.g., `llama_deploy.yaml`). The control plane pulls your code at the selected
git ref, reads the config, and runs your app.
"""

import asyncio
import subprocess

import click
from llama_agents.cli.commands.auth import validate_authenticated_profile
from llama_agents.cli.param_types import DeploymentType, GitShaType
from llama_agents.cli.styles import HEADER_COLOR, MUTED_COL, PRIMARY_COL, WARNING
from llama_agents.core.schema import LogEvent
from llama_agents.core.schema.deployments import (
    INTERNAL_CODE_REPO_SCHEME,
    DeploymentHistoryResponse,
    DeploymentResponse,
    DeploymentUpdate,
)
from rich import print as rprint
from rich.table import Table
from rich.text import Text

from ..app import app, console
from ..client import get_project_client, project_client_context
from ..options import (
    global_options,
    interactive_option,
    output_option,
    project_option,
    render_output,
)
from ..utils.capabilities import probe_code_push_support
from ..utils.git_push import (
    configure_git_remote,
    get_api_key,
    get_deployment_git_url,
    internal_push_refspec,
    push_to_remote,
)


@app.group(
    help="Deploy your app to the cloud.",
    no_args_is_help=True,
)
@global_options
def deployments() -> None:
    """Manage deployments"""
    pass


def _render_deployments_table(deployments: list[DeploymentResponse]) -> None:
    table = Table(show_edge=False, box=None, header_style=f"bold {HEADER_COLOR}")
    table.add_column("Name", style=PRIMARY_COL)
    table.add_column("ID", style=MUTED_COL)
    table.add_column("Status", style=MUTED_COL)
    table.add_column("URL", style=MUTED_COL)
    table.add_column("Repository", style=MUTED_COL)

    for deployment in deployments:
        repo_url = deployment.repo_url
        gh = "https://github.com/"
        if repo_url.startswith(gh):
            repo_url = "gh:" + repo_url.removeprefix(gh)

        table.add_row(
            deployment.display_name,
            deployment.id,
            deployment.status,
            str(deployment.apiserver_url or ""),
            repo_url,
        )

    console.print(table)


def _render_deployment_detail_table(deployment: DeploymentResponse) -> None:
    table = Table(show_edge=False, box=None, header_style=f"bold {HEADER_COLOR}")
    table.add_column("Property", style=MUTED_COL, justify="right")
    table.add_column("Value", style=PRIMARY_COL)

    table.add_row("Name", Text(deployment.display_name))
    table.add_row("ID", Text(deployment.id))
    table.add_row("Project ID", Text(deployment.project_id))
    table.add_row("Status", Text(deployment.status))
    table.add_row("Repository", Text(deployment.repo_url))
    table.add_row("Deployment File", Text(deployment.deployment_file_path))
    table.add_row("Git Ref", Text(deployment.git_ref or "-"))
    table.add_row("Last Deployed Commit", Text((deployment.git_sha or "-")[:7]))

    apiserver_url = deployment.apiserver_url
    table.add_row(
        "API Server URL",
        Text(str(apiserver_url) if apiserver_url else "-"),
    )

    secret_names = deployment.secret_names or []
    table.add_row("Secrets", Text("\n".join(secret_names), style="italic"))

    console.print(table)


def _do_get(
    deployment_id: str | None,
    interactive: bool,
    output: str,
    project: str | None,
) -> None:
    """Implementation of ``deployments get`` shared with the hidden ``list`` alias.

    No ``deployment_id`` → list all deployments (kubectl-style). With an ID →
    info table for that deployment. Never launches the TUI; users wanting a
    live view should use ``deployments logs --follow`` and
    ``deployments status``.
    """
    validate_authenticated_profile(interactive)
    try:
        client = get_project_client(project_id_override=project)

        if not deployment_id:
            # List all deployments (former `list_deployments` body).
            deployments = asyncio.run(client.list_deployments())

            if not deployments and output == "text":
                rprint(
                    f"[{WARNING}]No deployments found for project {client.project_id}[/]"
                )
                return

            render_output(
                deployments,
                output,
                lambda: _render_deployments_table(deployments),
            )
            return

        deployment = asyncio.run(client.get_deployment(deployment_id))
        render_output(
            deployment,
            output,
            lambda: _render_deployment_detail_table(deployment),
        )

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


@deployments.command("get")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@output_option
@project_option
@interactive_option
def get_deployment(
    deployment_id: str | None,
    interactive: bool,
    output: str,
    project: str | None,
) -> None:
    """Get one or more deployments.

    With no argument: lists all deployments in the project (kubectl-style).
    With a deployment ID: prints details for that deployment.

    Use ``-o json`` or ``-o yaml`` for machine-readable output. Use
    ``llamactl deployments logs <name> --follow`` to stream logs and
    ``llamactl deployments status <name>`` to check status.
    """
    _do_get(deployment_id, interactive, output, project)


@deployments.command("list", hidden=True)
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@output_option
@project_option
@interactive_option
def list_deployments(
    deployment_id: str | None,
    interactive: bool,
    output: str,
    project: str | None,
) -> None:
    """Hidden alias for ``deployments get``. Kept for backward compatibility."""
    _do_get(deployment_id, interactive, output, project)


@deployments.command("create")
@global_options
@interactive_option
def create_deployment(
    interactive: bool,
) -> None:
    """Create a new deployment."""
    validate_authenticated_profile(interactive)

    if not interactive:
        raise click.ClickException("This command requires an interactive session.")

    # Keep this import local: `llamactl --help` eagerly imports command modules,
    # and import-time profiling showed Textual adds material startup cost here.
    # Avoid adding other local imports unless instrumentation shows they are slow.
    from ..textual.deployment_form import create_deployment_form

    # Use interactive creation
    deployment_form = create_deployment_form(
        server_supports_code_push=probe_code_push_support(),
    )
    if deployment_form is None:
        rprint(f"[{WARNING}]Cancelled[/]")
        return

    rprint(
        f"[green]Created deployment: {deployment_form.display_name} (id: {deployment_form.id})[/green]"
    )


@deployments.command("configure-git-remote")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@project_option
@interactive_option
def configure_git_remote_cmd(
    deployment_id: str | None, interactive: bool, project: str | None
) -> None:
    """Configure a git remote for a deployment.

    Sets up authentication and a git remote named 'llamaagents-<deployment_id>'
    so you can push with:
      git push llamaagents-<deployment_id>

    Tip: 'llamactl deployments update' handles pushing and redeployment in one
    step. This command is useful for troubleshooting git push issues.
    """
    validate_authenticated_profile(interactive)
    try:
        # Verify we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise click.ClickException("Not a git repository")

        deployment_id = select_deployment(
            deployment_id, interactive=interactive, project_id_override=project
        )
        if not deployment_id:
            rprint(f"[{WARNING}]No deployment selected[/]")
            return

        client = get_project_client(project_id_override=project)
        git_url = get_deployment_git_url(client.base_url, deployment_id)
        api_key = get_api_key()
        remote_name = configure_git_remote(
            git_url, api_key, client.project_id, deployment_id
        )

        rprint(
            f"[green]Configured git remote '{remote_name}' for {deployment_id}[/green]"
        )
        rprint(f"Push with: [cyan]git push {remote_name}[/cyan]")

    except click.ClickException:
        raise
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


@deployments.command("delete")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@project_option
@interactive_option
def delete_deployment(
    deployment_id: str | None, interactive: bool, project: str | None
) -> None:
    """Delete a deployment"""
    # Keep this import local: the helper imports `questionary`, which import-time
    # profiling showed is a noticeable CLI startup cost. Avoid other local
    # imports unless instrumentation shows they are slow.
    from ..interactive_prompts.utils import confirm_action

    validate_authenticated_profile(interactive)
    try:
        client = get_project_client(project_id_override=project)

        deployment_id = select_deployment(
            deployment_id, interactive=interactive, project_id_override=project
        )
        if not deployment_id:
            rprint(f"[{WARNING}]No deployment selected[/]")
            return

        if interactive:
            if not confirm_action(f"Delete deployment '{deployment_id}'?"):
                rprint(f"[{WARNING}]Cancelled[/]")
                return

        asyncio.run(client.delete_deployment(deployment_id))
        rprint(f"[green]Deleted deployment: {deployment_id}[/green]")

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


@deployments.command("edit")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@project_option
@interactive_option
def edit_deployment(
    deployment_id: str | None, interactive: bool, project: str | None
) -> None:
    """Interactively edit a deployment"""
    # Keep this import local: `llamactl --help` eagerly imports command modules,
    # and import-time profiling showed Textual adds material startup cost here.
    # Avoid adding other local imports unless instrumentation shows they are slow.
    from ..textual.deployment_form import edit_deployment_form

    validate_authenticated_profile(interactive)
    try:
        client = get_project_client(project_id_override=project)

        deployment_id = select_deployment(
            deployment_id, interactive=interactive, project_id_override=project
        )
        if not deployment_id:
            rprint(f"[{WARNING}]No deployment selected[/]")
            return

        # Get current deployment details
        current_deployment = asyncio.run(client.get_deployment(deployment_id))

        # Use the interactive edit form
        updated_deployment = edit_deployment_form(
            current_deployment,
            server_supports_code_push=probe_code_push_support(),
        )
        if updated_deployment is None:
            rprint(f"[{WARNING}]Cancelled[/]")
            return

        rprint(
            f"[green]Successfully updated deployment: {updated_deployment.display_name}[/green]"
        )

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


def _push_internal_for_update(
    deployment_id: str,
    git_ref: str | None = None,
    project_id_override: str | None = None,
) -> None:
    """Push local code to the internal repo before updating.

    This ensures the S3-stored bare repo has the latest commits so the
    server can resolve the ref to a fresh SHA.
    """
    # Check we're in a git repo
    result = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True)
    if result.returncode != 0:
        rprint(
            f"[{WARNING}]Not in a git repo — skipping push, "
            "server will resolve from last pushed code[/]"
        )
        return

    client = get_project_client(project_id_override=project_id_override)
    api_key = get_api_key()
    git_url = get_deployment_git_url(client.base_url, deployment_id)
    remote_name = configure_git_remote(
        git_url, api_key, client.project_id, deployment_id
    )
    local_ref, target_ref = _internal_push_refspec(git_ref)
    with console.status("Pushing code..."):
        push_result = push_to_remote(
            remote_name, local_ref=local_ref, target_ref=target_ref
        )
    if push_result.returncode != 0:
        stderr = push_result.stderr.decode(errors="replace").strip()
        rprint(f"[{WARNING}]Push failed: {stderr}[/]")
        rprint(
            f"[{WARNING}]Continuing with update using last pushed code. "
            f"To debug, try: llamactl deployments configure-git-remote {deployment_id}[/]"
        )


def _internal_push_refspec(git_ref: str | None) -> tuple[str, str]:
    # Delegate to the shared implementation in git_push.py.
    # Keep this thin wrapper so existing call sites and tests don't break.
    return internal_push_refspec(git_ref)


@deployments.command("update")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@click.option(
    "--git-ref",
    help="Reference branch, tag, or commit SHA for the deployment. If not provided, the current reference and latest commit on it will be used.",
    default=None,
)
@project_option
@interactive_option
def refresh_deployment(
    deployment_id: str | None,
    git_ref: str | None,
    interactive: bool,
    project: str | None,
) -> None:
    """Update the deployment, pulling the latest code from it's branch"""
    validate_authenticated_profile(interactive)
    try:
        deployment_id = select_deployment(
            deployment_id, interactive=interactive, project_id_override=project
        )
        if not deployment_id:
            rprint(f"[{WARNING}]No deployment selected[/]")
            return

        # Get current deployment details to show what we're refreshing
        current_deployment = asyncio.run(
            get_project_client(project_id_override=project).get_deployment(
                deployment_id
            )
        )
        deployment_name = current_deployment.display_name
        old_git_sha = current_deployment.git_sha or ""

        # For internal repos, push local code first so the server has the
        # latest commits to resolve against.
        effective_git_ref = git_ref or current_deployment.git_ref
        if current_deployment.repo_url == INTERNAL_CODE_REPO_SCHEME:
            _push_internal_for_update(
                deployment_id, effective_git_ref, project_id_override=project
            )

        # Re-resolves the branch to the latest commit SHA.
        with console.status(f"Refreshing {deployment_name}..."):
            deployment_update = DeploymentUpdate(
                git_ref=effective_git_ref,
            )
            updated_deployment = asyncio.run(
                get_project_client(project_id_override=project).update_deployment(
                    deployment_id,
                    deployment_update,
                )
            )

        # Show the git SHA change with short SHAs
        new_git_sha = updated_deployment.git_sha or ""
        old_short = old_git_sha[:7] if old_git_sha else "none"
        new_short = new_git_sha[:7] if new_git_sha else "none"

        if old_git_sha == new_git_sha:
            rprint(f"No changes: already at {new_short}")
        else:
            rprint(f"Updated: {old_short} → {new_short}")

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


@deployments.command("history")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@output_option
@project_option
@interactive_option
def show_history(
    deployment_id: str | None,
    interactive: bool,
    output: str,
    project: str | None,
) -> None:
    """Show release history for a deployment."""
    validate_authenticated_profile(interactive)
    try:
        deployment_id = select_deployment(
            deployment_id, interactive=interactive, project_id_override=project
        )
        if not deployment_id:
            rprint(f"[{WARNING}]No deployment selected[/]")
            return

        async def _fetch_history() -> DeploymentHistoryResponse:
            async with project_client_context(project_id_override=project) as client:
                return await client.get_deployment_history(deployment_id)

        history = asyncio.run(_fetch_history())
        items = history.history
        # newest first
        items_sorted = sorted(
            items,
            key=lambda it: it.released_at,
            reverse=True,
        )

        if not items_sorted and output == "text":
            rprint(f"No history recorded for {deployment_id}")
            return

        def _render_table() -> None:
            table = Table(
                show_edge=False, box=None, header_style=f"bold {HEADER_COLOR}"
            )
            table.add_column("Released At", style=MUTED_COL)
            table.add_column("Git SHA", style=PRIMARY_COL)
            for item in items_sorted:
                table.add_row(item.released_at.isoformat(), item.git_sha)
            console.print(table)

        render_output(items_sorted, output, _render_table)
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


@deployments.command("rollback")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@click.option(
    "--git-sha", required=False, type=GitShaType(), help="Git SHA to roll back to"
)
@project_option
@interactive_option
def rollback(
    deployment_id: str | None,
    git_sha: str | None,
    interactive: bool,
    project: str | None,
) -> None:
    """Rollback a deployment to a previous git sha."""
    # Keep these imports local: profiling showed `questionary` is a noticeable
    # startup cost for `llamactl --help`. Avoid other local imports unless they
    # are measured and proven slow.
    import questionary

    from ..interactive_prompts.utils import confirm_action

    validate_authenticated_profile(interactive)
    try:
        deployment_id = select_deployment(
            deployment_id, interactive=interactive, project_id_override=project
        )
        if not deployment_id:
            rprint(f"[{WARNING}]No deployment selected[/]")
            return

        if not git_sha:
            # If not provided, prompt from history
            async def _fetch_current_and_history() -> tuple[
                DeploymentResponse, DeploymentHistoryResponse
            ]:
                async with project_client_context(
                    project_id_override=project
                ) as client:
                    current = await client.get_deployment(deployment_id)
                    hist = await client.get_deployment_history(deployment_id)
                    return current, hist

            current_deployment, history = asyncio.run(_fetch_current_and_history())
            current_sha = current_deployment.git_sha or ""

            items = history.history or []
            # Sort newest first
            items_sorted = sorted(items, key=lambda it: it.released_at, reverse=True)
            choices = []
            for it in items_sorted:
                short = it.git_sha[:7]
                suffix = (
                    " [current]" if current_sha and it.git_sha == current_sha else ""
                )
                choices.append(
                    questionary.Choice(
                        title=f"{short}{suffix} ({it.released_at})", value=it.git_sha
                    )
                )
            if not choices:
                rprint(f"[{WARNING}]No history available to rollback[/]")
                return
            git_sha = questionary.select("Select git sha:", choices=choices).ask()
            if not git_sha:
                rprint(f"[{WARNING}]Cancelled[/]")
                return

        if interactive and not confirm_action(
            f"Rollback '{deployment_id}' to {git_sha[:7]}?"
        ):
            rprint(f"[{WARNING}]Cancelled[/]")
            return

        async def _do_rollback() -> DeploymentResponse:
            async with project_client_context(project_id_override=project) as client:
                return await client.rollback_deployment(deployment_id, git_sha)

        updated = asyncio.run(_do_rollback())
        rprint(
            f"[green]Rollback initiated[/green]: {deployment_id} → {updated.git_sha[:7] if updated.git_sha else 'unknown'}"
        )
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


@deployments.command("logs")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Stream logs continuously until interrupted (Ctrl-C).",
)
@click.option(
    "--json",
    "json_lines",
    is_flag=True,
    default=False,
    help="Output one LogEvent JSON object per line (jsonl).",
)
@click.option(
    "--tail",
    "tail",
    type=int,
    default=200,
    show_default=True,
    help="Number of lines to retrieve from the end of the logs initially.",
)
@click.option(
    "--since-seconds",
    "since_seconds",
    type=int,
    default=None,
    help="Only return logs newer than this many seconds.",
)
@click.option(
    "--include-init-containers",
    is_flag=True,
    default=False,
    help="Include init container logs.",
)
@project_option
@interactive_option
def deployment_logs(
    deployment_id: str | None,
    follow: bool,
    json_lines: bool,
    tail: int,
    since_seconds: int | None,
    include_init_containers: bool,
    interactive: bool,
    project: str | None,
) -> None:
    """Stream or fetch logs for a deployment.

    By default, prints recent logs and exits. Use ``--follow`` to keep the
    stream open until you Ctrl-C. Use ``--json`` to emit one JSON
    ``LogEvent`` per line for downstream tooling (jsonl).
    """
    validate_authenticated_profile(interactive)

    deployment_id = select_deployment(
        deployment_id, interactive=interactive, project_id_override=project
    )
    if not deployment_id:
        rprint(f"[{WARNING}]No deployment selected[/]")
        return

    async def _consume() -> int:
        events_seen = 0
        async with project_client_context(project_id_override=project) as client:
            async for ev in client.stream_deployment_logs(
                deployment_id,
                include_init_containers=include_init_containers,
                tail_lines=tail,
                since_seconds=since_seconds,
                follow=follow,
            ):
                events_seen += 1
                _emit_log_event(ev, json_lines=json_lines)
        return events_seen

    try:
        events_seen = asyncio.run(_consume())
    except KeyboardInterrupt:
        # Clean exit on Ctrl-C; no traceback.
        return
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()

    if events_seen == 0 and not follow:
        click.echo("no logs available yet", err=True)


def _emit_log_event(ev: LogEvent, *, json_lines: bool) -> None:
    """Render a single LogEvent to stdout per the requested format."""
    if json_lines:
        click.echo(ev.model_dump_json())
        return

    # Defer to the lightweight plain renderer in log_format.
    from ..log_format import parse_log_body, render_plain

    body = render_plain(parse_log_body(ev.text))
    ts = ev.timestamp.isoformat() if ev.timestamp else ""
    prefix_parts = [p for p in (ts, f"{ev.pod}/{ev.container}") if p]
    prefix = " ".join(prefix_parts)
    click.echo(f"{prefix} {body}" if prefix else body)


@deployments.command("status")
@global_options
@click.argument("deployment_id", required=False, type=DeploymentType())
@output_option
@project_option
@interactive_option
def deployment_status(
    deployment_id: str | None,
    output: str,
    interactive: bool,
    project: str | None,
) -> None:
    """Show deployment status.

    Default text output is a single line: ``<id>: <status> (<short_sha>)``.
    Use ``-o json`` or ``-o yaml`` for the full ``DeploymentResponse``.
    """
    validate_authenticated_profile(interactive)

    deployment_id = select_deployment(
        deployment_id, interactive=interactive, project_id_override=project
    )
    if not deployment_id:
        rprint(f"[{WARNING}]No deployment selected[/]")
        return

    try:
        client = get_project_client(project_id_override=project)
        deployment = asyncio.run(client.get_deployment(deployment_id))

        def _render_text() -> None:
            short_sha = (deployment.git_sha or "")[:7] or "unknown"
            click.echo(f"{deployment.id}: {deployment.status} ({short_sha})")

        render_output(deployment, output, _render_text)
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise click.Abort()


def select_deployment(
    deployment_id: str | None,
    interactive: bool,
    project_id_override: str | None = None,
) -> str | None:
    """
    Select a deployment interactively if ID not provided.
    Returns the selected deployment ID or None if cancelled.

    In non-interactive sessions, returns None if deployment_id is not provided.
    """
    # Keep this import local: profiling showed `questionary` is a noticeable
    # startup cost for `llamactl --help`. Avoid other local imports unless they
    # are measured and proven slow.
    import questionary

    if deployment_id:
        return deployment_id

    # Don't attempt interactive selection in non-interactive sessions
    if not interactive:
        return None
    client = get_project_client(project_id_override=project_id_override)
    deployments = asyncio.run(client.list_deployments())

    if not deployments:
        rprint(f"[{WARNING}]No deployments found for project {client.project_id}[/]")
        return None

    choices = []
    for deployment in deployments:
        display_name = deployment.display_name
        deployment_id = deployment.id
        status = deployment.status
        choices.append(
            questionary.Choice(
                title=f"{display_name} ({deployment_id}) - {status}",
                value=deployment_id,
            )
        )

    return questionary.select("Select deployment:", choices=choices).ask()
