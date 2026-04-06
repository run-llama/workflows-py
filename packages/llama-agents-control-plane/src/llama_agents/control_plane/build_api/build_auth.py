import logging
from typing import Annotated

from aiocache import cached
from fastapi import HTTPException, Path, Security
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from llama_agents.core.schema.deployments import (
    LlamaDeploymentCRD,
)

from ..k8s_client import validate_deployment_token

logger = logging.getLogger(__name__)


@cached(ttl=15)
async def _validate_token_raw(deployment_id: str, token: str) -> str:
    result = await validate_deployment_token(deployment_id, token)
    if result is None:
        logger.warning(
            "Auth rejected for deployment=%s: invalid/expired token or deployment not found",
            deployment_id,
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired token, or deployment does not exist",
        )

    return result.model_dump_json()


async def validate_token(deployment_id: str, token: str) -> LlamaDeploymentCRD:
    result = await _validate_token_raw(deployment_id, token)
    return LlamaDeploymentCRD.model_validate_json(result)


bearer = HTTPBearer(auto_error=True)


async def authenticate_deployment(
    deployment_id: Annotated[str, Path()],
    creds: Annotated[HTTPAuthorizationCredentials, Security(bearer)],
) -> LlamaDeploymentCRD:
    """
    FastAPI dependency to validate the token for a deployment.
    """
    token = creds.credentials
    return await validate_token(deployment_id, token)


basic = HTTPBasic(auto_error=True)


async def authenticate_deployment_basic(
    deployment_id: Annotated[str, Path()],
    creds: Annotated[HTTPBasicCredentials, Security(basic)],
) -> LlamaDeploymentCRD:
    """
    FastAPI dependency to validate basic auth for a deployment.
    Uses password as token, or username as token if password is empty.
    """
    token = creds.password if creds.password else creds.username
    return await validate_token(deployment_id, token)
