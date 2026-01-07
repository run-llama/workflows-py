# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from importlib import import_module
from typing import (
    Any,
)


def get_qualified_name(value: Any) -> str:
    """
    Get the qualified name of a value.

    Args:
        value (Any): The value to get the qualified name for.

    Returns:
        str: The qualified name in the format 'module.class'.

    Raises:
        AttributeError: If value does not have __module__ or __class__ attributes

    """
    try:
        return value.__module__ + "." + value.__class__.__name__
    except AttributeError as e:
        raise AttributeError(f"Object {value} does not have required attributes: {e}")


def import_module_from_qualified_name(qualified_name: str) -> Any:
    """
    Import a module from a qualified name.

    Args:
        qualified_name (str): The fully qualified name of the module to import.

    Returns:
        Any: The imported module object.

    Raises:
        ValueError: If qualified_name is empty or malformed
        ImportError: If module cannot be imported
        AttributeError: If attribute cannot be found in module

    """
    if not qualified_name or "." not in qualified_name:
        raise ValueError("Qualified name must be in format 'module.attribute'")

    module_path = qualified_name.rsplit(".", 1)
    try:
        module = import_module(module_path[0])
        return getattr(module, module_path[1])
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path[0]}: {e}")
    except AttributeError as e:
        raise AttributeError(
            f"Attribute {module_path[1]} not found in module {module_path[0]}: {e}"
        )
