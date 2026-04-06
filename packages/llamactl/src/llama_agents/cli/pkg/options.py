from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import click
from llama_agents.core.config import DEFAULT_DEPLOYMENT_FILE_PATH

P = ParamSpec("P")
R = TypeVar("R")

# hack around for mypy not letting you set path_type=Path on click.Path
_ClickPath = getattr(click, "Path")


def _deployment_file_option(f: Callable[P, R]) -> Callable[P, R]:
    return click.argument(
        "deployment_file",
        required=False,
        default=DEFAULT_DEPLOYMENT_FILE_PATH,
        type=_ClickPath(dir_okay=True, resolve_path=True, path_type=Path),
    )(f)


def _python_version_option(f: Callable[P, R]) -> Callable[P, R]:
    return click.option(
        "--python-version",
        help="Python version for the base image. Default is inferred from the uv project configuration (.python-version or pyproject.toml). If no version can be inferred, python 3.12 is used.",
        required=False,
        default=None,
    )(f)


def _port_option(f: Callable[P, R]) -> Callable[P, R]:
    return click.option(
        "--port",
        help="The port to run the API server on. Defaults to 4501.",
        required=False,
        default=4501,
        type=int,
    )(f)


def _dockerignore_path_option(f: Callable[P, R]) -> Callable[P, R]:
    return click.option(
        "--dockerignore-path",
        help="Path for the output .dockerignore file. Defaults to .dockerignore",
        required=False,
        default=".dockerignore",
    )(f)


def _output_file_option(f: Callable[P, R]) -> Callable[P, R]:
    return click.option(
        "--output-file",
        help="Path for the output file to build the image. Defaults to Dockerfile",
        required=False,
        default="Dockerfile",
    )(f)


def _overwrite_option(f: Callable[P, R]) -> Callable[P, R]:
    return click.option(
        "--overwrite",
        help="Overwrite output files",
        is_flag=True,
    )(f)


def _exclude_option(f: Callable[P, R]) -> Callable[P, R]:
    return click.option(
        "--exclude",
        help="Path to exclude from the build (will be appended to .dockerignore). Can be used multiple times.",
        multiple=True,
        required=False,
        default=None,
    )(f)


def pkg_container_options(f: Callable[P, R]) -> Callable[P, R]:
    return _deployment_file_option(
        _python_version_option(
            _port_option(
                _dockerignore_path_option(
                    _overwrite_option(_exclude_option(_output_file_option(f)))
                )
            )
        )
    )
