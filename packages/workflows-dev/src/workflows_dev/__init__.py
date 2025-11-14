from __future__ import annotations

from .cli import cli

__all__ = ["cli", "main"]


def main() -> None:
    """Console script entry point."""
    cli()
