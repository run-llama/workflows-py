"""Utility functions for SSL/TLS configuration with optional truststore support."""

from __future__ import annotations

import os
import ssl
from typing import Any

import truststore


def get_ssl_context() -> ssl.SSLContext | bool:
    """Get SSL context for httpx clients.

    Returns an SSL context using truststore if LLAMA_DEPLOY_USE_TRUSTSTORE is set,
    otherwise returns True (default SSL verification).

    Truststore allows Python to use the system certificate store, which is useful
    for corporate environments with MITM proxies.
    """
    if os.getenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "").lower() in ("1", "true", "yes"):
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return True


def get_httpx_verify_param() -> Any:
    """Get the verify parameter for httpx clients.

    Returns an SSL context using truststore if configured, otherwise returns True.
    This can be passed directly to httpx.Client/AsyncClient's verify parameter.
    """
    return get_ssl_context()
