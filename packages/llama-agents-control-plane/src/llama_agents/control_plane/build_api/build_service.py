"""Build artifact service — creates and manages BuildArtifactStorage."""

from __future__ import annotations

from ..build_api.build_storage import BuildArtifactStorage
from ..settings import settings


def create_build_artifact_storage() -> BuildArtifactStorage | None:
    """Create a BuildArtifactStorage if S3 is configured, else return None."""
    if not settings.s3_bucket:
        return None

    return BuildArtifactStorage(
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        region=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        key_prefix=settings.build_s3_key_prefix,
    )


build_artifact_storage = create_build_artifact_storage()
