# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

SQLITE_MIGRATION_SOURCE: tuple[str, str] = (
    "dbos",
    "llama_agents.dbos._store.sqlite.migrations",
)

POSTGRES_MIGRATION_SOURCE: tuple[str, str] = (
    "dbos",
    "llama_agents.dbos._store.postgres.migrations",
)
