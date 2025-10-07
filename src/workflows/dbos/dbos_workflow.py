from __future__ import annotations

import logging

from llama_index_instrumentation import get_dispatcher

from workflows.resource import ResourceManager
from workflows.workflow import Workflow

dispatcher = get_dispatcher(__name__)
logger = logging.getLogger()


class DBOSWorkflow(Workflow):
    """DBOS Workflow base class."""

    def __init__(
        self,
        timeout: float | None = 45.0,
        disable_validation: bool = False,
        verbose: bool = False,
        resource_manager: ResourceManager | None = None,
        num_concurrent_runs: int | None = None,
    ) -> None:
        """
        Initialize a workflow instance.

        Args:
            timeout (float | None): Max seconds to wait for completion. `None`
                disables the timeout.
            disable_validation (bool): Skip pre-run validation of the event graph
                (not recommended).
            verbose (bool): If True, print step activity.
            resource_manager (ResourceManager | None): Custom resource manager
                for dependency injection.
            num_concurrent_runs (int | None): Limit on concurrent `run()` calls.
        """
        super().__init__(
            timeout=timeout,
            disable_validation=disable_validation,
            verbose=verbose,
            resource_manager=resource_manager,
            num_concurrent_runs=num_concurrent_runs,
        )

        # TODO (Qian): initialize DBOS specific attributes here
