from __future__ import annotations

import os
import sys
import pytest

if sys.version_info < (3, 10):
    pytest.skip("Requires Python 3.10 or higher", allow_module_level=True)

from dbos import DBOS, DBOSConfig

try:
    from typing import Union
except ImportError:
    from typing_extensions import Union

from typing import Any, Generator, Iterator, Optional

import pytest

from workflows.context import Context
from workflows.context.state_store import DictState
from workflows.decorators import step
from workflows.events import (
    StartEvent,
    StopEvent,
)
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow
from workflows.dbos.dbos_context import DBOSContext

from ..conftest import AnotherTestEvent, OneTestEvent

DBOS_SQLITE_FILE = "dbostest.sqlite"
DBOS_CONFIG: DBOSConfig = {
    "name": "pydantic_dbos_tests",
    "database_url": f"sqlite:///{DBOS_SQLITE_FILE}",
    "system_database_url": f"sqlite:///{DBOS_SQLITE_FILE}",
    "run_admin_server": False,
}


@pytest.fixture(scope="module")
def dbos() -> Generator[DBOS, Any, None]:
    dbos = DBOS(config=DBOS_CONFIG)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()


# Automatically clean up old DBOS sqlite files
@pytest.fixture(autouse=True, scope="module")
def cleanup_test_sqlite_file() -> Iterator[None]:
    if os.path.exists(DBOS_SQLITE_FILE):
        os.remove(DBOS_SQLITE_FILE)
    try:
        yield
    finally:
        if os.path.exists(DBOS_SQLITE_FILE):
            os.remove(DBOS_SQLITE_FILE)


@pytest.mark.asyncio
async def test_collect_events(dbos: DBOS) -> None:
    ev1 = OneTestEvent()
    ev2 = AnotherTestEvent()

    class TestWorkflow(Workflow):
        @step(num_workers=1)
        async def step1(self, _: StartEvent) -> OneTestEvent:
            return ev1

        @step(num_workers=1)
        async def step2(self, _: StartEvent) -> AnotherTestEvent:
            return ev2

        @step(num_workers=1)
        async def step3(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent]
        ) -> Optional[StopEvent]:
            assert isinstance(ctx, DBOSContext)
            events = ctx.collect_events(ev, [OneTestEvent, AnotherTestEvent])
            if events is None:
                return None
            return StopEvent(result=events)

    test_workflow = TestWorkflow()
    # For failure recovery, DBOS context should be defined statically so that the recovery process can find it.
    dbos_ctx: DBOSContext[DictState] = DBOSContext(test_workflow, "test_collect_events")
    r = await WorkflowTestRunner(test_workflow).run(ctx=dbos_ctx)
    assert r.result == [ev1, ev2]

    # Make sure the workflow is stored in DBOS.
    wf_list = dbos.list_workflows()
    # 3 + 1 _done steps -> four DBOS workflows.
    assert len(wf_list) == 4
    for wf in wf_list:
        print(
            f"Workflow in DBOS: {wf.workflow_id} with status {wf.status}, name {wf.name}"
        )
