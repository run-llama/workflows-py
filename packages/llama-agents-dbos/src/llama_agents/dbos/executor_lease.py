# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import logging
import uuid

import asyncpg
from llama_agents.dbos.journal.crud import _qualified_table_ref

logger = logging.getLogger(__name__)


class ExecutorLeaseManager:
    """Manages exclusive executor slot leases backed by a Postgres table."""

    def __init__(
        self,
        dsn: str,
        pool_size: int,
        heartbeat_interval: float = 10.0,
        lease_timeout: float = 30.0,
        slot_prefix: str = "executor",
        schema: str = "dbos",
    ) -> None:
        self._dsn = dsn
        self._pool_size = pool_size
        self._heartbeat_interval = heartbeat_interval
        self._lease_timeout = lease_timeout
        self._slot_prefix = slot_prefix
        self._schema = schema

        self._holder = str(uuid.uuid4())
        self._table = _qualified_table_ref("executor_leases", schema)
        self._pool: asyncpg.Pool | None = None
        self._slot_id: str | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._lease_lost_event = asyncio.Event()

    @property
    def executor_id(self) -> str:
        if self._slot_id is None:
            raise RuntimeError("Lease not acquired")
        return self._slot_id

    @property
    def lease_lost_event(self) -> asyncio.Event:
        return self._lease_lost_event

    async def _seed_slots(self) -> None:
        """Ensure slot rows exist in the executor_leases table."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            for i in range(self._pool_size):
                slot_id = f"{self._slot_prefix}-{i}"
                await conn.execute(
                    f"""
                    INSERT INTO {self._table} (slot_id, holder, heartbeat_at, acquired_at)
                    VALUES ($1, NULL, NULL, NULL)
                    ON CONFLICT (slot_id) DO NOTHING
                    """,
                    slot_id,
                )

    async def acquire(self, timeout: float | None = None) -> str:
        self._pool = await asyncpg.create_pool(dsn=self._dsn)
        await self._seed_slots()

        poll = 0.1
        elapsed = 0.0

        while True:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        f"""
                        SELECT slot_id FROM {self._table}
                        WHERE holder IS NULL
                           OR heartbeat_at < NOW() - INTERVAL '{self._lease_timeout} seconds'
                        ORDER BY slot_id
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                        """,
                    )
                    if row is not None:
                        slot_id: str = row["slot_id"]
                        await conn.execute(
                            f"""
                            UPDATE {self._table}
                            SET holder = $1, heartbeat_at = NOW(), acquired_at = NOW()
                            WHERE slot_id = $2
                            """,
                            self._holder,
                            slot_id,
                        )
                        self._slot_id = slot_id
                        self._heartbeat_task = asyncio.create_task(
                            self._heartbeat_loop()
                        )
                        logger.info("Acquired lease on slot %s", slot_id)
                        return slot_id

            if timeout is not None and elapsed >= timeout:
                await self._pool.close()
                self._pool = None
                raise TimeoutError(
                    f"Could not acquire executor lease within {timeout}s"
                )

            await asyncio.sleep(poll)
            elapsed += poll
            poll = min(poll * 2, 2.0)

    async def release(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._pool is not None and self._slot_id is not None:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    UPDATE {self._table}
                    SET holder = NULL, heartbeat_at = NULL
                    WHERE slot_id = $1 AND holder = $2
                    """,
                    self._slot_id,
                    self._holder,
                )
            logger.info("Released lease on slot %s", self._slot_id)

        self._slot_id = None

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _heartbeat_loop(self) -> None:
        assert self._pool is not None
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        f"""
                        UPDATE {self._table}
                        SET heartbeat_at = NOW()
                        WHERE slot_id = $1 AND holder = $2
                        RETURNING slot_id
                        """,
                        self._slot_id,
                        self._holder,
                    )
                    if row is None:
                        logger.warning(
                            "Lease lost on slot %s — holder no longer matches",
                            self._slot_id,
                        )
                        self._lease_lost_event.set()
                        return
            except Exception:
                logger.exception("Heartbeat failed for slot %s", self._slot_id)

    async def __aenter__(self) -> ExecutorLeaseManager:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()
