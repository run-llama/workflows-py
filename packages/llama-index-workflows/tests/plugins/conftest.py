from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

import pytest
from workflows.plugins.basic import BasicRuntime
from workflows.runtime.types.plugin import Runtime


@pytest.fixture(
    params=[
        pytest.param("basic", id="basic"),
        pytest.param("dbos", id="dbos"),
    ]
)
async def runtime(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncGenerator[Runtime, None]:
    """Yield an unlaunched runtime.

    DBOS requires workflows to be tracked (via ``Workflow(runtime=runtime)``)
    *before* ``launch()`` is called, so the fixture purposely does **not**
    call ``launch()``.  Each test must call ``runtime.launch()`` after
    creating all its workflows.
    """
    if request.param == "basic":
        rt = BasicRuntime()
        try:
            yield rt
        finally:
            rt.destroy()
    elif request.param == "dbos":
        try:
            from dbos import DBOS, DBOSConfig
            from workflows.plugins.dbos import DBOSRuntime
        except ImportError:
            pytest.skip("dbos not installed")
            return

        db_file: Path = tmp_path / "dbos_test.sqlite3"
        system_db_url: str = f"sqlite+pysqlite:///{db_file}?check_same_thread=false"
        config: DBOSConfig = {
            "name": "workflows-py-dbostest",
            "system_database_url": system_db_url,
            "run_admin_server": False,
        }
        DBOS(config=config)
        rt = DBOSRuntime()
        try:
            yield rt
        finally:
            rt.destroy()
