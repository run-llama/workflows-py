# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow


class ServiceNotFoundError(Exception):
    """An error raised when the service manager couldn't find a certain service name."""


class ServiceManager:
    """
    An helper class to decouple how services are managed from the Workflow class.

    A Service is nothing more than a workflow instance attached to another workflow.
    The service is made available to the steps of the main workflow.

    DEPRECATED: The ServiceManager class is deprecated and will be removed in a future version.
    """

    def __init__(self) -> None:
        warnings.warn(
            "ServiceManager is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._services: dict[str, Workflow] = {}

    def get(self, name: str, default: "Workflow | None" = None) -> "Workflow":
        try:
            return self._services[name]
        except KeyError:
            if default:
                return default

            msg = f"Service {name} not found"
            raise ServiceNotFoundError(msg)

    def add(self, name: str, service: "Workflow") -> None:
        self._services[name] = service
