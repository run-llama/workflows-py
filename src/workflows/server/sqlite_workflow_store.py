from workflows.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)
from typing import List
import sqlite3
import json


class SqliteWorkflowStore(AbstractWorkflowStore):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS handlers (handler_id TEXT PRIMARY KEY, workflow_name TEXT, status TEXT, ctx TEXT)"
        )
        conn.commit()
        conn.close()

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        sql = "SELECT handler_id, workflow_name, status, ctx FROM handlers WHERE 1=1"
        params: list = []

        # Filter by workflow_name list
        if query.workflow_name_in is not None:
            if len(query.workflow_name_in) == 0:
                return []
            placeholders = ",".join(["?"] * len(query.workflow_name_in))
            sql += f" AND workflow_name IN ({placeholders})"
            params.extend(query.workflow_name_in)

        # Filter by handler_id list
        if query.handler_id_in is not None:
            if len(query.handler_id_in) == 0:
                return []
            placeholders = ",".join(["?"] * len(query.handler_id_in))
            sql += f" AND handler_id IN ({placeholders})"
            params.extend(query.handler_id_in)

        # Filter by completed flag
        if query.status_in is not None:
            if len(query.status_in) == 0:
                return []
            placeholders = ",".join(["?"] * len(query.status_in))
            sql += f" AND status IN ({placeholders})"
            params.extend(query.status_in)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        finally:
            conn.close()

        return [_row_to_persistent_handler(row) for row in rows]

    async def update(self, handler: PersistentHandler) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO handlers (handler_id, workflow_name, status, ctx)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(handler_id) DO UPDATE SET
                workflow_name = excluded.workflow_name,
                status = excluded.status,
                ctx = excluded.ctx
            """,
            (
                handler.handler_id,
                handler.workflow_name,
                handler.status,
                json.dumps(handler.ctx),
            ),
        )
        conn.commit()
        conn.close()


def _row_to_persistent_handler(row: tuple) -> PersistentHandler:
    return PersistentHandler(
        handler_id=row[0],
        workflow_name=row[1],
        status=row[2],
        ctx=json.loads(row[3]),
    )
