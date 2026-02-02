# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Basic Docker/PostgreSQL connectivity test."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine


@pytest.mark.docker
def test_postgres_connection(postgres_engine: Engine) -> None:
    """Test basic PostgreSQL connectivity with SELECT 1."""
    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
