"""S3-compatible object storage base class."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, TypedDict

import aioboto3
from botocore import UNSIGNED
from botocore.client import Config

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client

logger = logging.getLogger(__name__)


class _S3ClientKwargs(TypedDict, total=False):
    endpoint_url: str
    region_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    config: Any


class S3ObjectStorage:
    """Base class for S3-compatible object storage backends.

    Provides shared session/client management. Subclasses define their own
    key scheme and domain methods.
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        key_prefix: str = "",
        unsigned: bool = False,
    ) -> None:
        self._bucket = bucket
        self._key_prefix = key_prefix.strip("/")
        self._session = aioboto3.Session()
        self._client_kwargs: _S3ClientKwargs = {}
        if endpoint_url:
            self._client_kwargs["endpoint_url"] = endpoint_url
        if region:
            self._client_kwargs["region_name"] = region
        if unsigned:
            # UNSIGNED bypasses boto's credential chain entirely — any creds
            # set via env/IRSA/etc. are ignored. Intended for authless
            # S3-compatible backends (s3proxy, LocalStack, public buckets).
            self._client_kwargs["config"] = Config(signature_version=UNSIGNED)
        elif access_key and secret_key:
            self._client_kwargs["aws_access_key_id"] = access_key
            self._client_kwargs["aws_secret_access_key"] = secret_key

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[S3Client]:
        async with self._session.client("s3", **self._client_kwargs) as client:
            yield client
