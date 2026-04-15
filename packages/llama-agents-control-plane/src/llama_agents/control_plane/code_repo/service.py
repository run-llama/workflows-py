"""Code repo service — creates and manages CodeRepoStorage."""

from __future__ import annotations

from ..settings import settings
from .storage import CodeRepoStorage


def create_code_repo_storage() -> CodeRepoStorage | None:
    """Create a CodeRepoStorage if S3 is configured, else return None."""
    if not settings.s3_bucket:
        return None

    return CodeRepoStorage(
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        region=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        key_prefix=settings.code_repo_s3_key_prefix,
        unsigned=settings.s3_unsigned,
    )


code_repo_storage = create_code_repo_storage()
