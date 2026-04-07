from typing import Callable

import httpx
from aiocache import cached
from pydantic import TypeAdapter

from ._github_auth import get_github_app_auth
from .github_api_schema import GithubAppInstallation, GitHubOwnerInfo, GitHubRepository


class GitHubApiClient:
    def __init__(
        self,
        auth_middleware: Callable[[httpx.Request], httpx.Request] = lambda x: x,
    ):
        self.client = httpx.AsyncClient(
            base_url="https://api.github.com", auth=auth_middleware
        )

    async def get_owner_info(self, owner: str) -> GitHubOwnerInfo | None:
        """Get owner information including numeric ID for GitHub App installation URLs."""
        response = await self.client.get(f"/users/{owner}", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            return GitHubOwnerInfo(
                id=data["id"], login=data["login"], type=data["type"]
            )
        elif response.status_code == 404:
            return None
        else:
            response.raise_for_status()
            return None  # unreachable, but satisfies type checker

    async def get_repository_info(
        self, owner: str, repo: str
    ) -> GitHubRepository | None:
        """Get repository information if accessible."""
        # WARNING! This will throw 401 errors if you have an App JWT token
        response = await self.client.get(f"/repos/{owner}/{repo}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return GitHubRepository.model_validate(response.json())


def get_app_jwt_client() -> httpx.AsyncClient:
    github_app_auth = get_github_app_auth()
    if github_app_auth is None:
        raise ValueError("Github app auth is not configured")
    jwt_token = github_app_auth.get_jwt()
    return httpx.AsyncClient(
        base_url="https://api.github.com",
        headers={"Authorization": f"Bearer {jwt_token}"},
    )


class GitHubAppApiClient(GitHubApiClient):
    def __init__(self) -> None:
        super().__init__(auth_middleware=self._auth_middleware)

    def _auth_middleware(self, request: httpx.Request) -> httpx.Request:
        github_app_auth = get_github_app_auth()
        if github_app_auth is None:
            raise ValueError("Github app auth is not configured")
        jwt_token = github_app_auth.get_jwt()
        request.headers.update({"Authorization": f"Bearer {jwt_token}"})
        return request

    async def get_repository_installation(
        self, owner: str, repo: str
    ) -> GithubAppInstallation | None:
        response = await self.client.get(f"/repos/{owner}/{repo}/installation")
        if response.status_code == 404:
            return None
        response.raise_for_status()

        return GithubAppInstallation.model_validate(response.json())

    async def get_org_installation(self, org: str) -> GithubAppInstallation | None:
        response = await self.client.get(f"/orgs/{org}/installation")
        if response.status_code == 404:
            return None
        response.raise_for_status()

        return GithubAppInstallation.model_validate(response.json())

    async def list_installations(self) -> list[GithubAppInstallation]:
        response = await self.client.get("/app/installations")
        response.raise_for_status()
        return ListInstallations.validate_python(response.json())

    @cached(ttl=60)
    async def get_installation_access_token(self, installation_id: int) -> str:
        response = await self.client.post(
            f"/app/installations/{installation_id}/access_tokens"
        )
        response.raise_for_status()
        return response.json()["token"]


def pat_api_client(pat: str) -> GitHubApiClient:
    def authenticate_with_pat(request: httpx.Request) -> httpx.Request:
        request.headers.update({"Authorization": f"token {pat}"})
        return request

    return GitHubApiClient(auth_middleware=authenticate_with_pat)


def installation_api_client(access_token: str) -> GitHubApiClient:
    def authenticate_with_installation(request: httpx.Request) -> httpx.Request:
        request.headers.update({"Authorization": f"token {access_token}"})
        return request

    return GitHubApiClient(auth_middleware=authenticate_with_installation)


github_app_api_client = GitHubAppApiClient()

ListInstallations = TypeAdapter(list[GithubAppInstallation])
