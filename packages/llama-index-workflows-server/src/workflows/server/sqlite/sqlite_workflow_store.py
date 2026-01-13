import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from workflows.context import JsonSerializer
from workflows.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)
from workflows.server.sqlite.migrate import run_migrations


class SqliteWorkflowStore(AbstractWorkflowStore):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            run_migrations(conn)
            conn.commit()
        finally:
            conn.close()

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        filter_spec = self._build_filters(query)
        if filter_spec is None:
            return []

        clauses, params = filter_spec
        sql = """SELECT handler_id, workflow_name, status, run_id, error, result,
                        started_at, updated_at, completed_at, idle_since, ctx FROM handlers"""
        if clauses:
            sql = f"{sql} WHERE {' AND '.join(clauses)}"
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
            INSERT INTO handlers (handler_id, workflow_name, status, run_id, error, result,
                                  started_at, updated_at, completed_at, idle_since, ctx)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(handler_id) DO UPDATE SET
                workflow_name = excluded.workflow_name,
                status = excluded.status,
                run_id = excluded.run_id,
                error = excluded.error,
                result = excluded.result,
                started_at = excluded.started_at,
                updated_at = excluded.updated_at,
                completed_at = excluded.completed_at,
                idle_since = excluded.idle_since,
                ctx = excluded.ctx
            """,
            (
                handler.handler_id,
                handler.workflow_name,
                handler.status,
                handler.run_id,
                handler.error,
                JsonSerializer().serialize(handler.result)
                if handler.result is not None
                else None,
                handler.started_at.isoformat() if handler.started_at else None,
                handler.updated_at.isoformat() if handler.updated_at else None,
                handler.completed_at.isoformat() if handler.completed_at else None,
                handler.idle_since.isoformat() if handler.idle_since else None,
                json.dumps(handler.ctx),
            ),
        )
        conn.commit()
        conn.close()

    async def delete(self, query: HandlerQuery) -> int:
        filter_spec = self._build_filters(query)
        if filter_spec is None:
            return 0

        clauses, params = filter_spec
        if not clauses:
            return 0

        sql = f"DELETE FROM handlers WHERE {' AND '.join(clauses)}"
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            deleted = cursor.rowcount
            conn.commit()
        finally:
            conn.close()

        return int(deleted)

    def _build_filters(
        self, query: HandlerQuery
    ) -> Optional[Tuple[List[str], List[str]]]:
        clauses: List[str] = []
        params: List[str] = []

        def add_in_clause(column: str, values: Sequence[str]) -> None:
            placeholders = ",".join(["?"] * len(values))
            clauses.append(f"{column} IN ({placeholders})")
            params.extend(values)

        if query.workflow_name_in is not None:
            if len(query.workflow_name_in) == 0:
                return None
            add_in_clause("workflow_name", query.workflow_name_in)

        if query.handler_id_in is not None:
            if len(query.handler_id_in) == 0:
                return None
            add_in_clause("handler_id", query.handler_id_in)

        if query.status_in is not None:
            if len(query.status_in) == 0:
                return None
            add_in_clause("status", query.status_in)

        if query.is_idle is not None:
            if query.is_idle:
                clauses.append("idle_since IS NOT NULL")
            else:
                clauses.append("idle_since IS NULL")

        if not clauses:
            return clauses, params

        return clauses, params


def _row_to_persistent_handler(row: tuple) -> PersistentHandler:
    return PersistentHandler(
        handler_id=row[0],
        workflow_name=row[1],
        status=row[2],
        run_id=row[3],
        error=row[4],
        result=json.loads(row[5]) if row[5] else None,
        started_at=datetime.fromisoformat(row[6]) if row[6] else None,
        updated_at=datetime.fromisoformat(row[7]) if row[7] else None,
        completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
        idle_since=datetime.fromisoformat(row[9]) if row[9] else None,
        ctx=json.loads(row[10]),
    )
