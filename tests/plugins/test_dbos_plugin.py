from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest

# Require Python 3.10+ for dbos typing
# if sys.version_info < (3, 10):  # pragma: no cover - environment guard
#     pytest.skip("Requires Python 3.10 or higher", allow_module_level=True)
from dbos import DBOS, DBOSConfig  # pyright: ignore[reportMissingImports]
from workflows.decorators import step
from workflows.events import StartEvent, StopEvent
from workflows.plugins.dbos import DBOSRuntime
from workflows.workflow import Workflow


@pytest.fixture()
def dbos_config(tmp_path: Path) -> Generator[None, Any, None]:
    """Configure DBOS without launching - let the plugin handle launch."""
    # Use a file-based SQLite DB so the schema persists across connections/threads
    db_file: Path = tmp_path / "dbos_test.sqlite3"
    # Allow usage across threads in tests
    system_db_url: str = f"sqlite+pysqlite:///{db_file}?check_same_thread=false"

    config: DBOSConfig = {
        "name": "workflows-py-dbostest",
        "system_database_url": system_db_url,
        "run_admin_server": False,
    }
    DBOS(config=config)
    try:
        yield None
    finally:
        DBOS.destroy()


class SimpleWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="ok")


@pytest.mark.asyncio
async def test_dbos_plugin_simple_run(dbos_config: None) -> None:
    """Test workflow runs with DBOS runtime using new lifecycle."""
    runtime = DBOSRuntime()
    wf = SimpleWorkflow(runtime=runtime)
    runtime.launch()  # Registers workflows and calls DBOS.launch()
    try:
        handler = wf.run()
        result = await handler
        assert result == "ok"
    finally:
        runtime.destroy()
