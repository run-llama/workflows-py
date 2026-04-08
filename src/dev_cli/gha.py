from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import click


def _encode(value: Any) -> str:
    """Encode an output value for GHA. Strings pass through; everything
    else is JSON-encoded so lists/bools/dicts round-trip cleanly through
    ``fromJSON()`` in workflow expressions."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def write_outputs(
    outputs: Mapping[str, Any], output_path: str | Path | None = None
) -> None:
    """Write GitHub Actions step outputs."""
    target = output_path or os.environ.get("GITHUB_OUTPUT")
    encoded = {key: _encode(value) for key, value in outputs.items()}
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            for key, value in encoded.items():
                handle.write(f"{key}={value}\n")
    else:
        for key, value in encoded.items():
            click.echo(f"{key}={value}")
