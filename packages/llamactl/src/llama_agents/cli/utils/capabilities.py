"""Helpers for probing server capabilities."""

from __future__ import annotations

import logging

from llama_agents.core.schema.public import Capabilities

logger = logging.getLogger(__name__)


def probe_code_push_support() -> bool | None:
    """Probe the current environment's server for code_push capability.

    Returns True/False if the probe succeeds, or None if it fails (fail open).
    """
    # Local import: env_service transitively pulls in httpx/auth_service which
    # are too heavy for CLI startup (guarded by test_cli_imports).
    from llama_agents.cli.config.env_service import service

    try:
        env = service.get_current_environment()
        if not env.capabilities:
            # Capabilities not yet populated — try a fresh probe
            env = service.auto_update_env(env)
        return Capabilities.CODE_PUSH in env.capabilities
    except Exception:
        logger.debug("Failed to probe server capabilities", exc_info=True)
        return None
