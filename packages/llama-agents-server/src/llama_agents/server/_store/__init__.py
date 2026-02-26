# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

SQLITE_MIGRATION_SOURCE: tuple[str, str] = (
    "server",
    "llama_agents.server._store.sqlite.migrations",
)

POSTGRES_MIGRATION_SOURCE: tuple[str, str] = (
    "server",
    "llama_agents.server._store.postgres.migrations",
)
