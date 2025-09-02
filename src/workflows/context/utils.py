# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from importlib import import_module
from typing import (
    Any,
)
from enum import Enum


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


class StateModificationType(Enum):
    UPDATED_PROPERTY = "state_property_updated"
    DELETED_PROPERTY = "state_property_deleted"
    ADDED_PROPERTY = "state_property_added"
    UPDATED_STATE = "state_updated"


def compare_states(start: dict, end: dict) -> StateModificationType:
    """
    Compared two different workflow states and return what type of change occurred between them.

    Args:
        start (dict): Initial state (as dictionary)
        end (dict): Final state (as dictionary)

    Returns:
        StateModificationType
    """
    # assumption: start and end are two different dictionaries (checked with hashing)
    diffs_start_end = start.keys() - end.keys()
    diff_end_start = end.keys() - start.keys()
    if (
        len(diffs_start_end) == 0 and len(diff_end_start) == 0
    ):  # start and end have the same keys, so at least one of them has been updated
        return StateModificationType.UPDATED_PROPERTY
    elif (
        len(diffs_start_end) > 0 and len(diff_end_start) == 0
    ):  # start has one or more keys that end does not have, so at least one key has been deleted
        return StateModificationType.DELETED_PROPERTY
    elif (
        len(diffs_start_end) == 0 and len(diff_end_start) > 0
    ):  # end has one or more keys that start does not have, so at least one key has been added
        return StateModificationType.ADDED_PROPERTY
    else:  # handle cases where both dictionaries have keys that the other dictionary does not have (generic update)
        return StateModificationType.UPDATED_STATE
