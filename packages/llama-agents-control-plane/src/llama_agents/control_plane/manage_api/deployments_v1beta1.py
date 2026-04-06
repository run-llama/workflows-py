from llama_agents.core.server.manage_api import create_v1beta1_deployments_router

from .deployments_service import deployments_service, public_service

router = create_v1beta1_deployments_router(deployments_service, public_service)
