from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import Any, BinaryIO

if sys.version_info >= (3, 11):
    # Stdlib TOML parser (Python 3.11+)
    import tomllib as _toml_backend
else:
    # Lightweight TOML backport for Python 3.10
    import tomli as _toml_backend


def get_logging_level_mapping() -> Mapping[str, int]:
    """Return a mapping of log level names to their numeric values."""
    if sys.version_info >= (3, 11):
        mapping = logging.getLevelNamesMapping()
        return {k: int(v) for k, v in mapping.items() if isinstance(v, int)}

    return {
        name: level
        for name, level in logging._nameToLevel.items()
        if isinstance(level, int)
    }


def load_toml_file(file_obj: BinaryIO) -> dict[str, Any]:
    """Load TOML data from a binary file object in a version-agnostic way."""
    return _toml_backend.load(file_obj)
