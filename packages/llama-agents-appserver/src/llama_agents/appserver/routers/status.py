from fastapi import APIRouter
from llama_agents.appserver.types import Status, StatusEnum

health_router = APIRouter()


@health_router.get("/health", include_in_schema=False)
async def health() -> Status:
    return Status(
        status=StatusEnum.HEALTHY,
    )


@health_router.get("/healthz", include_in_schema=False)
async def healthz() -> Status:
    return Status(status=StatusEnum.HEALTHY)


@health_router.get("/livez", include_in_schema=False)
async def livez() -> Status:
    return Status(status=StatusEnum.HEALTHY)


@health_router.get("/readyz", include_in_schema=False)
async def readyz() -> Status:
    return Status(status=StatusEnum.HEALTHY)
