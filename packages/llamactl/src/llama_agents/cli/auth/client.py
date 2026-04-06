from __future__ import annotations

import asyncio
import logging
import sys
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Awaitable,
    Callable,
)

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from jwt.algorithms import RSAAlgorithm
from llama_agents.cli.config.schema import DeviceOIDC
from llama_agents.core.client.ssl_util import get_httpx_verify_param
from pydantic import BaseModel

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


logger = logging.getLogger(__name__)


class OidcDiscoveryResponse(BaseModel):
    discovery_url: str
    client_ids: dict[str, str] | None = None


class OidcProviderConfiguration(BaseModel):
    device_authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    scopes_supported: list[str] | None = None
    jwks_uri: str | None = None


class JsonWebKey(BaseModel):
    kty: str
    kid: str | None = None
    use: str | None = None
    alg: str | None = None
    n: str | None = None
    e: str | None = None
    x5c: list[str] | None = None
    x5t: str | None = None
    x5t_s256: str | None = None


class JsonWebKeySet(BaseModel):
    keys: list[JsonWebKey]


class AuthMeResponse(BaseModel):
    id: str
    email: str | None = None
    last_login_provider: str | None = None
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    claims: dict[str, Any] | None = None
    restrict: Any | None = None
    created_at: str | None = None


class ClientContextManager(AsyncContextManager):
    def __init__(self, base_url: str | None, auth: httpx.Auth | None = None) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        verify = get_httpx_verify_param()
        if self.base_url:
            self.client = httpx.AsyncClient(
                base_url=self.base_url, auth=auth, verify=verify
            )
        else:
            self.client = httpx.AsyncClient(auth=auth, verify=verify)

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:
            pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()


class PlatformAuthDiscoveryClient(ClientContextManager):
    """Client for ad hoc auth endpoints under /api/v1/auth."""

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)

    async def oidc_discovery(self) -> OidcDiscoveryResponse:
        resp = await self.client.get("/api/v1/auth/oidc/discovery", timeout=10.0)
        resp.raise_for_status()
        return OidcDiscoveryResponse.model_validate(resp.json())


class APIToken(BaseModel):
    token: str
    id: str


class PlatformAuthClient(ClientContextManager):
    """Client for user introspection under /api/v1/auth/me."""

    def __init__(
        self, base_url: str, id_token: str | None = None, auth: httpx.Auth | None = None
    ) -> None:
        self.id_token = id_token
        super().__init__(base_url, auth=auth)

    async def me(self) -> AuthMeResponse:
        headers = (
            {"Authorization": f"Bearer {self.id_token}"} if self.id_token else None
        )
        resp = await self.client.get("/api/v1/auth/me", headers=headers, timeout=10.0)
        resp.raise_for_status()
        return AuthMeResponse.model_validate(resp.json())

    async def create_agent_api_key(self, name: str) -> APIToken:
        resp = await self.client.post(
            "/api/v1/api-keys",
            json={"name": name, "project_id": None},
        )
        resp.raise_for_status()
        json = resp.json()
        token = json["redacted_api_key"]
        id = json["id"]
        return APIToken(token=token, id=id)

    async def delete_api_key(self, id: str) -> None:
        response = await self.client.delete(f"/api/v1/api-keys/{id}")
        response.raise_for_status()


class RefreshMiddleware(httpx.Auth):
    def __init__(
        self,
        device_oidc: DeviceOIDC,
        on_refresh: Callable[[DeviceOIDC], Awaitable[None]],
    ) -> None:
        self.device_oidc = device_oidc
        self.on_refresh = on_refresh
        self.lock = asyncio.Lock()

    async def _refresh_and_update(self) -> None:
        new_device_oidc = await refresh(self.device_oidc)
        self.device_oidc = new_device_oidc
        try:
            await self.on_refresh(new_device_oidc)
        except Exception:
            logger.exception("Error in on_refresh callback")

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = self.device_oidc.device_access_token
        request.headers["Authorization"] = f"Bearer {token}"

        response = yield request
        if response.status_code == 401:
            async with self.lock:
                if token == self.device_oidc.device_access_token:
                    await self._refresh_and_update()
                    request.headers["Authorization"] = (
                        f"Bearer {self.device_oidc.device_access_token}"
                    )
                yield request


class DeviceAuthorizationRequest(BaseModel):
    client_id: str
    scope: str


class DeviceAuthorizationResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None = None
    expires_in: int
    interval: int | None = None


class TokenRequestDeviceCode(BaseModel):
    grant_type: str = "urn:ietf:params:oauth:grant-type:device_code"
    device_code: str
    client_id: str


class TokenResponse(BaseModel):
    # Success fields
    id_token: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    expires_in: int | None = None
    token_type: str | None = None
    scope: str | None = None
    # Error fields
    error: str | None = None
    error_description: str | None = None


class TokenRequestRefresh(BaseModel):
    grant_type: str = "refresh_token"
    refresh_token: str
    client_id: str


class OIDCClient(ClientContextManager):
    def __init__(self) -> None:
        super().__init__(None)

    async def fetch_provider_configuration(
        self, discovery_url: str
    ) -> OidcProviderConfiguration:
        resp = await self.client.get(discovery_url, timeout=10.0)
        resp.raise_for_status()
        return OidcProviderConfiguration.model_validate(resp.json())

    async def device_authorization(
        self, device_endpoint: str, request: DeviceAuthorizationRequest
    ) -> DeviceAuthorizationResponse:
        resp = await self.client.post(
            device_endpoint,
            data=request.model_dump(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return DeviceAuthorizationResponse.model_validate(resp.json())

    async def token_with_device_code(
        self, token_endpoint: str, request: TokenRequestDeviceCode
    ) -> TokenResponse:
        resp = await self.client.post(
            token_endpoint,
            data=request.model_dump(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=10.0,
        )
        # Do not raise for status; callers inspect error payloads during polling
        try:
            payload = resp.json()
        except Exception:
            # Fall back to minimal error information
            return TokenResponse(error="invalid_response", error_description=resp.text)
        return TokenResponse.model_validate(payload)

    async def token_with_refresh(
        self, token_endpoint: str, request: TokenRequestRefresh
    ) -> TokenResponse:
        resp = await self.client.post(
            token_endpoint,
            data=request.model_dump(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=10.0,
        )
        try:
            payload = resp.json()
        except Exception:
            return TokenResponse(error="invalid_response", error_description=resp.text)
        return TokenResponse.model_validate(payload)

    async def get_jwks(self, jwks_uri: str) -> JsonWebKeySet:
        resp = await self.client.get(jwks_uri, timeout=10.0)
        resp.raise_for_status()
        return JsonWebKeySet.model_validate(resp.json())


async def decode_jwt_claims_from_device_oidc(
    oidc_device: DeviceOIDC,
    verify_audience: bool = True,
    verify_expiration: bool = True,
    audience: str | None = None,
) -> dict[str, Any]:
    """Decode JWT claims by discovering provider and verifying via JWKS.

    Assumes RSA signing. Audience verification can be toggled and, when enabled,
    an audience value can be provided.
    """
    if not oidc_device.device_id_token:
        raise ValueError("Device ID token is missing. Cannot decode claims.")
    async with OIDCClient() as oidc:
        provider = await oidc.fetch_provider_configuration(oidc_device.discovery_url)
        jwks_uri = provider.jwks_uri
        if not jwks_uri:
            raise ValueError("Provider does not expose jwks_uri")
    return await decode_jwt_claims(
        oidc_device.device_id_token,
        jwks_uri,
        verify_audience,
        verify_expiration,
        audience,
    )


async def decode_jwt_claims(
    token: str,
    jwks_uri: str,
    verify_audience: bool = True,
    verify_expiration: bool = True,
    audience: str | None = None,
) -> dict[str, Any]:
    async with OIDCClient() as oidc:
        jwks = await oidc.get_jwks(jwks_uri)

    # Select key
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    keys = jwks.keys
    key = next((k for k in keys if k.kid == kid), None) or next(iter(keys), None)
    if not key:
        raise ValueError("Signing key not found in JWKS")

    # Build public key (RSA-only)
    if key.kty != "RSA":
        raise ValueError("Unsupported JWK kty; only RSA is supported")
    key_json = key.model_dump_json()
    raw_key = RSAAlgorithm.from_jwk(key_json)
    if not isinstance(raw_key, RSAPublicKey):
        raise ValueError("Unsupported RSA key type; expected RSAPublicKey from JWKS")
    public_key = raw_key

    return jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        options={"verify_aud": verify_audience, "verify_exp": verify_expiration},
        audience=audience,
    )


async def refresh(device_oidc: DeviceOIDC) -> DeviceOIDC:
    """
    Run a refresh on the access token, storing updated tokens in a new DeviceOIDC.
    """
    async with OIDCClient() as oidc:
        provider = await oidc.fetch_provider_configuration(device_oidc.discovery_url)
        token_endpoint = provider.token_endpoint
        if not token_endpoint:
            raise ValueError("Provider does not expose token_endpoint")
        if not device_oidc.device_refresh_token:
            raise ValueError("Device refresh token is missing. Cannot refresh.")
        token = await oidc.token_with_refresh(
            token_endpoint,
            TokenRequestRefresh(
                refresh_token=device_oidc.device_refresh_token,
                client_id=device_oidc.client_id,
            ),
        )
        copy = device_oidc.model_copy()
        if not token.access_token:
            raise ValueError("Refresh failed: token response missing access_token")
        copy.device_access_token = token.access_token
        copy.device_refresh_token = token.refresh_token or copy.device_refresh_token
        copy.device_id_token = token.id_token or copy.device_id_token
        return copy
