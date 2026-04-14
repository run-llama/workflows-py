"""Helpers for probing server capabilities."""

from __future__ import annotations

import logging

from llama_agents.core.schema.public import Capabilities

logger = logging.getLogger(__name__)


def _probe_capability(capability: str) -> bool | None:
    """Probe whether the current environment's server advertises a capability.

    Returns True/False if the probe succeeds, or None if it fails (fail open).
    """
    from llama_agents.cli.config.env_service import service

    try:
        env = service.get_current_environment()
        if not env.capabilities:
            env = service.auto_update_env(env)
        return capability in env.capabilities
    except Exception:
        logger.debug("Failed to probe server capabilities", exc_info=True)
        return None


def probe_code_push_support() -> bool | None:
    """Probe the current environment's server for code_push capability.

    Returns True/False if the probe succeeds, or None if it fails (fail open).
    """
    return _probe_capability(Capabilities.CODE_PUSH)


def probe_organizations_support() -> bool | None:
    """Probe the current environment's server for organizations capability.

    Returns True/False if the probe succeeds, or None if it fails (fail open).
    """
    return _probe_capability(Capabilities.ORGANIZATIONS)
