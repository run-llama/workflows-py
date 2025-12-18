from __future__ import annotations

from workflows_dev.cli import cli

__all__ = ["cli", "main"]


def main() -> None:
    """Console script entry point."""
    cli()
