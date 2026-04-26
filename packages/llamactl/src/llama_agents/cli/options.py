import json
import logging
import os
from typing import Any, Callable, ParamSpec, TypeVar

import click
from llama_agents.cli.interactive_prompts.session_utils import is_interactive_session
from pydantic import BaseModel

from .debug import setup_file_logging

P = ParamSpec("P")
R = TypeVar("R")


def global_options(f: Callable[P, R]) -> Callable[P, R]:
    """Common decorator to add global options to command groups"""

    return native_tls_option(file_logging(f))


def output_option(f: Callable[P, R]) -> Callable[P, R]:
    """Add a `-o/--output` option for read commands.

    Choices: ``text`` (default; existing Rich tables), ``json``, ``yaml``.
    The value is exposed as the ``output`` keyword argument.
    """

    return click.option(
        "-o",
        "--output",
        "output",
        type=click.Choice(["text", "json", "yaml"], case_sensitive=False),
        default="text",
        show_default=True,
        help="Output format. Use 'json' or 'yaml' for machine-readable output.",
    )(f)


def project_option(f: Callable[P, R]) -> Callable[P, R]:
    """Add a ``--project`` option to override the active profile's project for a command.

    The value is exposed as the ``project`` keyword argument and should be
    threaded into ``get_project_client()`` or ``project_client_context()`` via
    the ``project_id_override`` parameter.
    """

    return click.option(
        "--project",
        "project",
        default=None,
        help="Project ID to use for this command (overrides active profile).",
    )(f)


def render_output(
    payload: BaseModel | list[BaseModel] | Any,
    output: str,
    text_renderer: Callable[[], None],
) -> None:
    """Render a payload according to ``output`` mode.

    - ``text``: invoke ``text_renderer`` (typically prints a Rich table).
    - ``json``: emit canonical JSON via ``click.echo`` (no Rich markup).
    - ``yaml``: emit YAML via ``click.echo`` (currently the naive Pydantic
      shape; the apply-format with masked secrets is a follow-up).

    ``payload`` may be a Pydantic model, a list of Pydantic models, or any
    JSON-serializable value (dict, list of dicts). The structured outputs go
    through ``click.echo`` so they pipe cleanly even when Rich would
    otherwise insert markup.
    """

    mode = output.lower()
    if mode == "text":
        text_renderer()
        return

    def _to_json_safe(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [_to_json_safe(item) for item in value]
        if isinstance(value, dict):
            return {k: _to_json_safe(v) for k, v in value.items()}
        return value

    if mode == "json":
        click.echo(json.dumps(_to_json_safe(payload), indent=2))
        return
    if mode == "yaml":
        # Defer pyyaml import: it's a measurable chunk of CLI startup and
        # only paid for by the (rare) `-o yaml` path.
        import yaml

        click.echo(yaml.safe_dump(_to_json_safe(payload), sort_keys=False))
        return

    raise click.ClickException(f"Unknown output mode: {output}")


def interactive_option(f: Callable[P, R]) -> Callable[P, R]:
    """Add an interactive option to the command"""

    default = is_interactive_session()
    return click.option(
        "--interactive/--no-interactive",
        help="Run in interactive mode. If not provided, will default to the current session's interactive state.",
        is_flag=True,
        default=default,
    )(f)


def native_tls_option(f: Callable[P, R]) -> Callable[P, R]:
    """Enable native TLS to trust system configured trust store rather than python bundled trust stores.

    When enabled, we set:
    - UV_NATIVE_TLS=1 to instruct uv to use the platform trust store
    - LLAMA_DEPLOY_USE_TRUSTSTORE=1 to use system certificate store for Python httpx clients
    """

    def _enable_native_tls(
        ctx: click.Context, param: click.Parameter, value: bool
    ) -> bool:
        if value:
            # Don't override if user explicitly set a value
            os.environ.setdefault("UV_NATIVE_TLS", "1")
            os.environ.setdefault("LLAMA_DEPLOY_USE_TRUSTSTORE", "1")
        return value

    return click.option(
        "--native-tls",
        is_flag=True,
        help=(
            "Enable native TLS mode to use system certificate store rather than runtime defaults. Can be set via LLAMACTL_NATIVE_TLS=1"
        ),
        callback=_enable_native_tls,
        expose_value=False,
        is_eager=True,
        envvar=["LLAMACTL_NATIVE_TLS"],
    )(f)


def file_logging(f: Callable[P, R]) -> Callable[P, R]:
    def debug_callback(ctx: click.Context, param: click.Parameter, value: str) -> str:
        if value:
            setup_file_logging(level=logging._nameToLevel.get(value, logging.INFO))
        return value

    return click.option(
        "--log-level",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        help="Enable debug logging to file",
        callback=debug_callback,
        expose_value=False,
        is_eager=True,
        hidden=True,
    )(f)
