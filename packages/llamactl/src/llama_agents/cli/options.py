import logging
import os
from typing import Callable, ParamSpec, TypeVar

import click
from llama_agents.cli.interactive_prompts.session_utils import is_interactive_session

from .debug import setup_file_logging

P = ParamSpec("P")
R = TypeVar("R")


def global_options(f: Callable[P, R]) -> Callable[P, R]:
    """Common decorator to add global options to command groups"""

    return native_tls_option(file_logging(f))


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
