"""File reference abstractions for durable workflows.

Provides light-weight containers that describe remotely hosted files alongside
services that can hydrate them onto local disk storage on demand.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field


class FileRef(BaseModel):
    """Serializable container describing how to obtain a remote file."""

    model_config = ConfigDict(frozen=True)

    service: str = Field(description="Name of the registered file service.")
    locator: str = Field(description="Opaque identifier understood by the service.")
    filename: str | None = Field(
        default=None,
        description="Optional preferred filename when persisted locally.",
    )
    content_type: str | None = Field(
        default=None,
        description="Optional MIME type metadata for the file.",
    )

    async def hydrate(self, registry: FileServiceRegistry) -> HydratedFile:
        """Hydrate the file by delegating to the registry."""

        return await registry.hydrate(self)


@dataclass
class HydratedFile:
    """Represents a locally hydrated file."""

    path: Path
    _release_cb: Callable[[], Awaitable[None]] | None = None
    _closed: bool = False

    async def __aenter__(self) -> HydratedFile:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the underlying resource."""

        if self._closed:
            return
        if self._release_cb is not None:
            await self._release_cb()
            self._release_cb = None
        self._closed = True

    def open(self, mode: str = "rb") -> IO[bytes]:
        """Open the hydrated file. Caller is responsible for closing the handle."""

        return self.path.open(mode)

    def read_bytes(self) -> bytes:
        """Read the hydrated file contents."""

        return self.path.read_bytes()


@runtime_checkable
class FileService(Protocol):
    """Protocol for pluggable file services."""

    @property
    def name(self) -> str: ...

    async def hydrate(self, file_ref: FileRef) -> HydratedFile: ...

    async def cleanup(self) -> None: ...


class FileServiceRegistry:
    """Registry that wires FileRef instances to their backing services."""

    def __init__(self) -> None:
        self._services: dict[str, FileService] = {}

    def register(self, service: FileService) -> None:
        if service.name in self._services:
            raise ValueError(f"File service '{service.name}' already registered")
        self._services[service.name] = service

    def unregister(self, name: str) -> None:
        self._services.pop(name, None)

    def get(self, name: str) -> FileService:
        try:
            return self._services[name]
        except KeyError as exc:
            raise KeyError(f"File service '{name}' is not registered") from exc

    async def hydrate(self, file_ref: FileRef) -> HydratedFile:
        service = self.get(file_ref.service)
        return await service.hydrate(file_ref)

    async def cleanup(self) -> None:
        for service in self._services.values():
            await service.cleanup()


@dataclass
class _CacheEntry:
    path: Path
    downloaded_at: float
    active_handles: int
    released_at: float | None = None


class RemoteHttpFileService(FileService):
    """File service that lazily downloads files via HTTP(S)."""

    def __init__(
        self,
        *,
        name: str = "remote_http",
        client: httpx.AsyncClient | None = None,
        cache_dir: Path | None = None,
        cleanup_after: timedelta = timedelta(minutes=5),
        ttl: timedelta | None = None,
        timeout: float | httpx.Timeout | None = 30.0,
    ) -> None:
        self._name = name
        self._client = client
        self._cache_dir = Path(cache_dir or (Path(tempfile.gettempdir()) / "llama_workflows_files" / name))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_after = cleanup_after
        self._ttl = ttl
        self._timeout = timeout
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    async def hydrate(self, file_ref: FileRef) -> HydratedFile:
        key = self._cache_key(file_ref)
        entry = await self._try_checkout_entry(key)
        if entry is None:
            path = await self._download(file_ref, key)
            entry = await self._store_entry(key, path)
        return HydratedFile(path=entry.path, _release_cb=lambda: self._release(key))

    async def cleanup(self) -> None:
        await self._cleanup_expired()

    def _cache_key(self, file_ref: FileRef) -> str:
        return hashlib.sha256(file_ref.locator.encode("utf-8")).hexdigest()

    def _target_path(self, key: str, filename: str | None) -> Path:
        safe_name = Path(filename or "payload").name
        subdir = self._cache_dir / key[:2] / key[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / safe_name

    async def _try_checkout_entry(self, key: str) -> _CacheEntry | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            now = time.monotonic()
            ttl_seconds = self._ttl.total_seconds() if self._ttl is not None else None
            if ttl_seconds is not None and (now - entry.downloaded_at) >= ttl_seconds and entry.active_handles == 0:
                await self._remove_entry_locked(key, entry)
                return None
            if not entry.path.exists():
                await self._remove_entry_locked(key, entry)
                return None
            entry.active_handles += 1
            entry.released_at = None
            return entry

    async def _store_entry(self, key: str, path: Path) -> _CacheEntry:
        async with self._lock:
            existing = self._entries.get(key)
            if existing:
                existing.active_handles += 1
                existing.released_at = None
                path.unlink(missing_ok=True)
                return existing
            entry = _CacheEntry(
                path=path,
                downloaded_at=time.monotonic(),
                active_handles=1,
            )
            self._entries[key] = entry
            return entry

    async def _release(self, key: str) -> None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.active_handles = max(0, entry.active_handles - 1)
            if entry.active_handles == 0:
                entry.released_at = time.monotonic()

    async def _cleanup_expired(self) -> None:
        async with self._lock:
            if self._cleanup_after.total_seconds() <= 0:
                grace = 0.0
            else:
                grace = self._cleanup_after.total_seconds()
            now = time.monotonic()
            keys_to_remove: list[str] = []
            for key, entry in self._entries.items():
                if entry.active_handles > 0 or entry.released_at is None:
                    continue
                if now - entry.released_at >= grace:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                entry = self._entries.pop(key)
                entry.path.unlink(missing_ok=True)

    async def _remove_entry_locked(self, key: str, entry: _CacheEntry) -> None:
        entry.path.unlink(missing_ok=True)
        self._entries.pop(key, None)

    async def _download(self, file_ref: FileRef, key: str) -> Path:
        target = self._target_path(key, file_ref.filename)
        tmp_path = target.with_suffix(".tmp")

        @asynccontextmanager
        async def _client_ctx() -> AsyncIterator[httpx.AsyncClient]:
            if self._client is not None:
                yield self._client
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    yield client

        async with _client_ctx() as client:
            async with client.stream(
                "GET",
                file_ref.locator,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("wb") as fp:
                    async for chunk in response.aiter_bytes():
                        fp.write(chunk)
        tmp_path.replace(target)
        return target
