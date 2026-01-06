# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.


class WorkflowValidationError(Exception):
    """Raised when the workflow configuration or step signatures are invalid."""


class WorkflowTimeoutError(Exception):
    """Raised when a workflow run exceeds the configured timeout."""


class WorkflowRuntimeError(Exception):
    """Raised for runtime errors during step execution or event routing."""


class WorkflowDone(Exception):
    """Internal control-flow exception used to terminate workers at run end."""


class WorkflowCancelledByUser(Exception):
    """Raised when a run is cancelled via the handler or programmatically."""


class WorkflowStepDoesNotExistError(Exception):
    """Raised when addressing a step that does not exist in the workflow."""


class WorkflowConfigurationError(Exception):
    """Raised when a logical configuration error is detected pre-run."""


class ContextSerdeError(Exception):
    """Raised when serializing/deserializing a `Context` fails."""
