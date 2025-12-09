from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from httpx import MockTransport, Response

from workflows.file_ref import (
    FileRef,
    FileServiceRegistry,
    RemoteHttpFileService,
)


@pytest.mark.asyncio
async def test_remote_http_file_service_caches_and_cleans(tmp_path) -> None:
    body = b"file-content"
    call_count = 0

    def handler(_: httpx.Request) -> Response:
        nonlocal call_count
        call_count += 1
        return Response(200, content=body)

    transport = MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        service = RemoteHttpFileService(
            client=client,
            cache_dir=tmp_path,
            cleanup_after=timedelta(seconds=0),
        )
        registry = FileServiceRegistry()
        registry.register(service)
        file_ref = FileRef(
            service=service.name,
            locator="https://example.com/data.bin",
            filename="data.bin",
        )

        async with await registry.hydrate(file_ref) as hydrated:
            first_path = hydrated.path
            assert hydrated.read_bytes() == body

        async with await file_ref.hydrate(registry) as hydrated_again:
            assert hydrated_again.path == first_path
            assert hydrated_again.read_bytes() == body

        assert call_count == 1, "cached path should be reused without re-downloading"

        await registry.cleanup()
        assert not first_path.exists(), "cleanup removes hydrated file once released"


@pytest.mark.asyncio
async def test_remote_http_file_service_respects_ttl(tmp_path) -> None:
    body_holder = {"payload": b"alpha"}

    def handler(_: httpx.Request) -> Response:
        return Response(200, content=body_holder["payload"])

    transport = MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        service = RemoteHttpFileService(
            client=client,
            cache_dir=tmp_path,
            ttl=timedelta(seconds=0),
        )
        registry = FileServiceRegistry()
        registry.register(service)
        file_ref = FileRef(service=service.name, locator="https://example.com/data.bin")

        hydrated_first = await registry.hydrate(file_ref)
        assert hydrated_first.read_bytes() == b"alpha"
        await hydrated_first.aclose()

        body_holder["payload"] = b"beta"
        hydrated_second = await registry.hydrate(file_ref)
        assert hydrated_second.read_bytes() == b"beta", "expired cache should redownload"
        await hydrated_second.aclose()


def test_file_service_registry_enforces_uniqueness(tmp_path) -> None:
    registry = FileServiceRegistry()
    service = RemoteHttpFileService(cache_dir=tmp_path)
    registry.register(service)

    with pytest.raises(ValueError):
        registry.register(service)

    with pytest.raises(KeyError):
        registry.get("missing")
