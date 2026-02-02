from __future__ import annotations

from .cli import cli, pytest_cmd

__all__ = ["cli", "main", "pytest_main"]


def main() -> None:
    """Console script entry point."""
    cli()


def pytest_main() -> None:
    """Shortcut entry point for 'workflows-dev pytest'."""
    pytest_cmd()
