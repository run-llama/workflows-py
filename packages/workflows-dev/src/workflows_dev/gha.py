from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import click


def write_outputs(outputs: Mapping[str, str], output_path: Path | None = None) -> None:
    """Write GitHub Actions step outputs."""
    target = output_path or os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                handle.write(f"{key}={value}\n")
    else:
        for key, value in outputs.items():
            click.echo(f"{key}={value}")

