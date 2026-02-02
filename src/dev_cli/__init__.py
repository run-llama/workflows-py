from __future__ import annotations

from .cli import _maybe_inject_pytest_subcommand, cli, pytest_cmd

__all__ = ["cli", "main", "pytest_main"]


def main() -> None:
    """Console script entry point."""
    _maybe_inject_pytest_subcommand()
    cli()


def pytest_main() -> None:
    """Shortcut entry point for 'dev pytest'."""
    pytest_cmd()
