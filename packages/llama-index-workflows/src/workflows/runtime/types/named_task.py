# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""NamedTask associates asyncio tasks with stable string keys for journaling."""

from __future__ import annotations

from asyncio import Task
from dataclasses import dataclass
from typing import Any

# Key prefix for pull tasks
PULL_PREFIX = "__pull__"


@dataclass
class NamedTask:
    """An asyncio task with a stable string key for identification.

    Keys are strings like "step_name:worker_id" for workers or "__pull__:0" for pull.
    """

    key: str
    task: Task[Any]

    @staticmethod
    def worker(step_name: str, worker_id: int, task: Task[Any]) -> NamedTask:
        """Create a NamedTask for a worker."""
        return NamedTask(f"{step_name}:{worker_id}", task)

    @staticmethod
    def pull(sequence: int, task: Task[Any]) -> NamedTask:
        """Create a NamedTask for a pull task."""
        return NamedTask(f"{PULL_PREFIX}:{sequence}", task)

    def is_pull(self) -> bool:
        """Check if this is a pull task."""
        return self.key.startswith(f"{PULL_PREFIX}:")

    @staticmethod
    def all_tasks(named_tasks: list[NamedTask]) -> set[Task[Any]]:
        """Extract all tasks for use with asyncio.wait."""
        return {nt.task for nt in named_tasks}

    @staticmethod
    def find_by_key(named_tasks: list[NamedTask], key: str) -> Task[Any] | None:
        """Find a task by its key, returns None if not found."""
        for nt in named_tasks:
            if nt.key == key:
                return nt.task
        return None

    @staticmethod
    def get_key(named_tasks: list[NamedTask], task: Task[Any]) -> str:
        """Get the key for a task. Raises KeyError if not found."""
        for nt in named_tasks:
            if nt.task is task:
                return nt.key
        raise KeyError(f"Task {task} not found")

    @staticmethod
    def pick_highest_priority(
        named_tasks: list[NamedTask], done: set[Task[Any]]
    ) -> Task[Any] | None:
        """Return highest priority completed task from done set.

        Priority is determined by list order - tasks earlier in the list
        have higher priority. Workers should be listed before pull.

        Returns None if done is empty.
        Raises ValueError if done is non-empty but no tasks match (indicates bug).
        """
        if not done:
            return None
        for nt in named_tasks:
            if nt.task in done:
                return nt.task
        raise ValueError(
            f"No tasks in done set match named_tasks. "
            f"done={done}, named_tasks={[nt.key for nt in named_tasks]}"
        )
