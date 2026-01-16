# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import hashlib
import weakref
from typing import Iterator

from workflows.utils import get_steps_from_class
from workflows.workflow import Workflow


def _get_all_subclasses(cls: type[Workflow]) -> set[type[Workflow]]:
    """Recursively get all subclasses of a class."""
    result: set[type[Workflow]] = set()
    for subclass in cls.__subclasses__():
        result.add(subclass)
        result.update(_get_all_subclasses(subclass))
    return result


def compute_workflow_id(workflow_or_class: Workflow | type[Workflow]) -> str:
    """Compute a stable ID for a workflow based on its class name and step names.

    The ID format is: "{class_name}:{hash}" where hash is a 12-character
    hex digest of the sorted step names.

    Args:
        workflow_or_class: A Workflow instance or class

    Returns:
        A stable identifier string like "MyWorkflow:a1b2c3d4e5f6"
    """
    if isinstance(workflow_or_class, type):
        # It's a class
        cls = workflow_or_class
        class_name = cls.__name__
        # Get steps from class only (no instance methods since we don't have an instance)
        steps = get_steps_from_class(cls)
        step_names = sorted(steps.keys())
        # Also include class-level _step_functions
        if hasattr(cls, "_step_functions"):
            step_names = sorted(set(step_names) | set(cls._step_functions.keys()))
    else:
        # It's an instance
        instance = workflow_or_class
        class_name = instance.__class__.__name__
        step_names = sorted(instance._get_steps().keys())

    # Create hash from sorted step names
    hash_input = ":".join(step_names)
    step_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    return f"{class_name}:{step_hash}"


class _WorkflowRegistry:
    """Registry for tracking Workflow classes and explicitly registered instances.

    Classes are discovered automatically via Python's __subclasses__() mechanism.
    Instances must be explicitly registered via the register() function.
    """

    def __init__(self) -> None:
        # Instance registry: maps instance -> (name, computed_id)
        # Uses weak references to allow garbage collection
        self._instances: weakref.WeakKeyDictionary[Workflow, tuple[str, str]] = (
            weakref.WeakKeyDictionary()
        )

    # --- Class Discovery ---

    @property
    def classes(self) -> frozenset[type[Workflow]]:
        """All known Workflow subclasses."""
        return frozenset(_get_all_subclasses(Workflow))

    def get_class_id(self, cls: type[Workflow]) -> str:
        """Get the computed ID for a workflow class."""
        return compute_workflow_id(cls)

    def iter_classes(self) -> Iterator[tuple[type[Workflow], str]]:
        """Iterate over (class, id) pairs."""
        for cls in self.classes:
            yield cls, compute_workflow_id(cls)

    # --- Instance Registry ---

    def register_instance(
        self,
        instance: Workflow,
        name: str | None = None,
    ) -> str:
        """Register a workflow instance with an optional explicit name.

        Args:
            instance: The workflow instance to register
            name: Optional explicit name. If not provided, uses computed ID.

        Returns:
            The workflow ID (either the explicit name or computed ID)
        """
        computed_id = compute_workflow_id(instance)
        workflow_id = name if name is not None else computed_id
        self._instances[instance] = (workflow_id, computed_id)
        return workflow_id

    def get_instance_id(self, instance: Workflow) -> str | None:
        """Get the ID for a registered instance, or None if not registered."""
        entry = self._instances.get(instance)
        return entry[0] if entry else None

    def get_instance_computed_id(self, instance: Workflow) -> str | None:
        """Get the computed ID for a registered instance."""
        entry = self._instances.get(instance)
        return entry[1] if entry else None

    @property
    def instances(self) -> list[Workflow]:
        """All registered workflow instances (that haven't been garbage collected)."""
        return list(self._instances.keys())

    def iter_instances(self) -> Iterator[tuple[Workflow, str, str]]:
        """Iterate over (instance, name, computed_id) tuples."""
        for instance, (name, computed_id) in self._instances.items():
            yield instance, name, computed_id


# Global singleton
WorkflowRegistry = _WorkflowRegistry()
