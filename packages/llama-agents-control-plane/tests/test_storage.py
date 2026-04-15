# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Unit tests for the S3ObjectStorage client-construction path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore import UNSIGNED
from botocore.client import Config
from llama_agents.control_plane.storage import S3ObjectStorage


def _make_async_cm() -> AsyncMock:
    cm = AsyncMock()
    cm.__aenter__.return_value = MagicMock()
    cm.__aexit__.return_value = None
    return cm


@pytest.mark.asyncio
async def test_client_uses_unsigned_config_when_unsigned_is_true() -> None:
    storage = S3ObjectStorage(
        bucket="b",
        endpoint_url="https://s3proxy.example",
        region="us-east-1",
        access_key="leaked",
        secret_key="leaked",
        unsigned=True,
    )
    with patch.object(
        storage._session, "client", return_value=_make_async_cm()
    ) as mock_client:
        async with storage._client():
            pass

    _, kwargs = mock_client.call_args
    cfg = kwargs.get("config")
    assert isinstance(cfg, Config)
    assert getattr(cfg, "signature_version") is UNSIGNED
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs


@pytest.mark.asyncio
async def test_client_omits_config_and_passes_creds_when_unsigned_is_false() -> None:
    storage = S3ObjectStorage(
        bucket="b",
        endpoint_url="https://s3.example",
        region="us-east-1",
        access_key="ak",
        secret_key="sk",
        unsigned=False,
    )
    with patch.object(
        storage._session, "client", return_value=_make_async_cm()
    ) as mock_client:
        async with storage._client():
            pass

    _, kwargs = mock_client.call_args
    assert "config" not in kwargs
    assert kwargs.get("aws_access_key_id") == "ak"
    assert kwargs.get("aws_secret_access_key") == "sk"
