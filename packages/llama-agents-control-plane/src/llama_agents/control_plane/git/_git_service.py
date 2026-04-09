import asyncio
import logging
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from aiocache import cached
from httpx import HTTPStatusError
from llama_agents.core.config import DEFAULT_DEPLOYMENT_FILE_PATH
from llama_agents.core.deployment_config import (
    read_deployment_config,
    resolve_config_parent,
)
from llama_agents.core.git.git_util import (
    GitAccessError,
    clone_repo,
    parse_github_repo_url,
    validate_git_credential_access,
    validate_git_public_access,
)
from llama_agents.core.schema.git_validation import (
    GitApplicationValidationResponse,
    RepositoryValidationResponse,
)
from llama_agents.core.ui_build import ui_build_output_path

from .. import k8s_client
from ._github_auth import GitHubAppAuth, get_github_app_auth
from .github_api_client import (
    GitHubApiClient,
    github_app_api_client,
    installation_api_client,
    pat_api_client,
)


@dataclass
class GitHubAppAccess:
    owner: str
    repo: str
    installation_id: int


@dataclass
class GitRepository:
    url: str
    access_token: str | None  # passed as basic auth colon delimited username:password


@dataclass
class InaccessibleRepository:
    message: str
    github_app_name: str | None = None
    github_app_installation_url: str | None = None


GitAccessType = GitHubAppAccess | GitRepository | InaccessibleRepository

logger = logging.getLogger(__name__)


@dataclass
class RepositoryExistenceResult:
    exists: bool | None
    message: str | None = None
    via_installation: bool = False


class GitService:
    @cached(ttl=10)
    async def get_access(
        self,
        repository_url: str,
        deployment_id: str,
        pat: str | None = None,
        existing_pat: str | None = None,
    ) -> GitAccessType:
        if self._is_github_repository(repository_url):
            return await self._check_github_access_type(
                repository_url, pat, existing_pat
            )
        else:
            return await self._check_generic_access_type(
                repository_url, deployment_id, pat, existing_pat
            )

    async def validate_repository(
        self,
        repository_url: str,
        project_id: str,
        deployment_id: str | None = None,
        pat: str | None = None,
    ) -> RepositoryValidationResponse:
        """
        Validates repository access and returns unified response.

        Args:
            repository_url: The repository URL to validate
            deployment_id: Optional existing deployment ID to check for existing credentials
            pat: Optional PAT to validate (for new deployments or PAT updates)
        """

        existing_pat = (
            None
            if deployment_id is None
            else await k8s_client.get_deployment_pat(deployment_id)
        )

        access = await self.get_access(
            repository_url, deployment_id, pat, existing_pat=existing_pat
        )

        # Check if PAT is obsolete - this happens when:
        # 1. Repo is public and deployment has a PAT
        # 2. GitHub App is available for GitHub repos and deployment has a PAT
        pat_is_obsolete = False
        if deployment_id and existing_pat:
            github_app_auth = get_github_app_auth()
            if self._is_github_repository(repository_url):
                # For GitHub repos, PAT is obsolete if GitHub App is available
                pat_is_obsolete = github_app_auth is not None
            else:
                # For non-GitHub repos, PAT is obsolete if repo is public
                pat_is_obsolete = (
                    isinstance(access, GitRepository) and access.access_token is None
                )

        # For GitHub repos, always try to include the app name and connect URL
        github_app_name: str | None = None
        github_app_installation_url: str | None = None
        github_app_settings_url: str | None = None
        if self._is_github_repository(repository_url):
            github_app_auth = get_github_app_auth()
            if github_app_auth:
                github_app_name = github_app_auth.app_name
                try:
                    (owner, _) = parse_github_repo_url(repository_url)
                    github_app_installation_url = await self._construct_install_url(
                        github_app_auth, owner
                    )
                except ValueError:
                    # Invalid URL format, skip connect URL
                    pass

        # For GitHubAppAccess, construct the settings URL for managing the installation
        if isinstance(access, GitHubAppAccess):
            github_app_settings_url = await self._construct_settings_url(
                access.owner, access.installation_id
            )

        response: RepositoryValidationResponse
        match access:
            case GitHubAppAccess():
                response = RepositoryValidationResponse(
                    accessible=True,
                    message="Access confirmed via GitHub App installation.",
                    pat_is_obsolete=pat_is_obsolete,
                    github_app_name=github_app_name,
                    github_app_installation_url=github_app_installation_url,
                    github_app_settings_url=github_app_settings_url,
                )
            case GitRepository() as git_repository:
                if git_repository.access_token is None:
                    message = "Repository is a public repository."
                else:
                    if existing_pat and pat is None:
                        message = "Access confirmed via existing Personal Access Token."
                    else:
                        message = "Access confirmed via Personal Access Token."

                response = RepositoryValidationResponse(
                    accessible=True,
                    message=message,
                    pat_is_obsolete=pat_is_obsolete,
                    github_app_name=github_app_name,
                    github_app_installation_url=github_app_installation_url,
                )
            case InaccessibleRepository() as inaccessible_repository:
                response = RepositoryValidationResponse(
                    accessible=False,
                    message=inaccessible_repository.message,
                    github_app_name=inaccessible_repository.github_app_name,
                    github_app_installation_url=inaccessible_repository.github_app_installation_url,
                )
            case _:
                raise ValueError(f"Invalid access type: {access}")

        return response

    def _is_github_repository(self, repository_url: str) -> bool:
        """Check if the repository URL is a GitHub repository."""
        # Handle SSH shorthand: git@github.com:owner/repo.git
        if repository_url.startswith("git@"):
            host = repository_url.split("@", 1)[1].split(":", 1)[0]
            return host == "github.com"

        # Handle URLs with scheme (https://, ssh://, git://) and schemeless (github.com/...)
        url = repository_url
        if "://" not in url:
            url = "https://" + url
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        return hostname in {"github.com", "www.github.com"}

    async def _check_github_access_type(
        self,
        repository_url: str,
        pat: str | None = None,
        existing_pat: str | None = None,
    ) -> GitAccessType:
        """Validate GitHub repository access."""
        try:
            (owner, repo) = parse_github_repo_url(repository_url)
        except ValueError:
            return InaccessibleRepository(
                message=(
                    "Invalid GitHub repository URL. "
                    "Expected format 'https://github.com/<owner>/<repository>'."
                )
            )

        # First, probe public access via git directly to avoid GitHub API rate limits.
        if await asyncio.to_thread(validate_git_public_access, repository_url):
            logger.info("Access resolved for %s/%s: public", owner, repo)
            return GitRepository(url=repository_url, access_token=None)

        logger.info(
            "Public probe failed for %s/%s, trying authenticated methods", owner, repo
        )

        github_app_auth = get_github_app_auth()
        org_installation = None

        # Check GitHub App access first (preferred method)
        if github_app_auth:
            repo_installation = await github_app_api_client.get_repository_installation(
                owner, repo
            )

            if repo_installation and repo_installation.id:
                logger.info(
                    "Access resolved for %s/%s: github_app repo_installation=%d",
                    owner,
                    repo,
                    repo_installation.id,
                )
                return GitHubAppAccess(
                    owner=owner,
                    repo=repo,
                    installation_id=repo_installation.id,
                )

            org_installation = await github_app_api_client.get_org_installation(owner)
            if org_installation:
                logger.info(
                    "Found org installation for %s: id=%s selection=%s",
                    owner,
                    org_installation.id,
                    org_installation.repository_selection,
                )
            else:
                logger.info("No GitHub App installation found for %s/%s", owner, repo)

        # Try PAT validation (provided PAT or existing deployment PAT)
        pat_to_test = pat or existing_pat

        if pat_to_test:
            if await self._validate_pat_access(owner, repo, pat_to_test):
                logger.info("Access resolved for %s/%s: pat", owner, repo)
                return GitRepository(
                    url=repository_url,
                    access_token=pat_to_test,
                )
            logger.info("PAT validation failed for %s/%s", owner, repo)

        installation_token: str | None = None
        if org_installation and org_installation.id:
            try:
                installation_token = (
                    await github_app_api_client.get_installation_access_token(
                        org_installation.id
                    )
                )
            except HTTPStatusError:
                installation_token = None

        existence = await self._check_github_repository_exists(
            owner=owner,
            repo_name=repo,
            pat=pat_to_test,
            installation_token=installation_token,
        )

        if (
            existence.exists
            and existence.via_installation
            and org_installation
            and org_installation.id
        ):
            logger.info(
                "Access resolved for %s/%s: github_app org_installation=%d",
                owner,
                repo,
                org_installation.id,
            )
            return GitHubAppAccess(
                owner=owner,
                repo=repo,
                installation_id=org_installation.id,
            )

        if (
            existence.exists is None
            and org_installation is not None
            and org_installation.repository_selection == "all"
        ):
            existence = RepositoryExistenceResult(
                exists=False,
                message=f"GitHub repository '{owner}/{repo}' does not exist.",
            )
        elif (
            existence.exists is None
            and org_installation is not None
            and org_installation.repository_selection == "selected"
        ):
            existence = RepositoryExistenceResult(
                exists=None,
                message=(
                    "GitHub App installation is limited to selected repositories and "
                    f"does not currently include '{owner}/{repo}'."
                ),
            )

        github_app_auth = get_github_app_auth()
        app_name = github_app_auth.app_name if github_app_auth else None

        app_install_url = (
            await self._construct_install_url(github_app_auth, owner)
            if github_app_auth
            else None
        )

        if existence.exists is False:
            message = (
                existence.message
                or f"GitHub repository '{owner}/{repo}' does not exist."
            )
            app_name = None
            app_install_url = None
        else:
            if existence.message:
                message = existence.message
            elif pat_to_test:
                message = (
                    "Personal Access Token does not have access to this repository."
                )
            else:
                message = (
                    f"Unable to access GitHub repository '{owner}/{repo}'. "
                    "If the repository is private, install the GitHub App for this owner "
                    "or provide a Personal Access Token."
                )

        logger.warning(
            "GitHub repo inaccessible: %s/%s — %s",
            owner,
            repo,
            message,
        )
        return InaccessibleRepository(
            message=message,
            github_app_name=app_name,
            github_app_installation_url=app_install_url,
        )

    async def _check_github_repository_exists(
        self,
        owner: str,
        repo_name: str,
        pat: str | None,
        installation_token: str | None,
    ) -> RepositoryExistenceResult:
        """
        Attempt to verify that a GitHub repository exists using anonymous and PAT-backed requests.

        Returns:
            RepositoryExistenceResult where `exists` is:
                True  - repository confirmed to exist
                False - repository confirmed not to exist (e.g. owner missing)
                None  - repository existence could not be confirmed
        """

        clients: list[tuple[GitHubApiClient, str]] = []

        if pat:
            clients.append((pat_api_client(pat), "pat"))

        if installation_token:
            clients.append(
                (installation_api_client(installation_token), "installation")
            )

        clients.append((GitHubApiClient(), "unauthenticated"))

        for client, source in clients:
            try:
                repo_info = await client.get_repository_info(owner, repo_name)
            except HTTPStatusError:
                repo_info = None

            if repo_info is not None:
                return RepositoryExistenceResult(
                    exists=True, via_installation=source == "installation"
                )

        for client, _ in clients:
            try:
                owner_info = await client.get_owner_info(owner)
            except HTTPStatusError:
                owner_info = None

            if owner_info is not None:
                return RepositoryExistenceResult(exists=None)

        return RepositoryExistenceResult(
            exists=False,
            message=f"GitHub owner '{owner}' does not exist.",
        )

    async def _check_generic_access_type(
        self,
        repository_url: str,
        deployment_id: str | None = None,
        pat: str | None = None,
        existing_pat: str | None = None,
    ) -> GitAccessType:
        """Validate non-GitHub repository access using git commands."""

        # First, try public access
        if await asyncio.to_thread(validate_git_public_access, repository_url):
            return GitRepository(
                url=repository_url,
                access_token=None,
            )

        # Try PAT/credential validation (provided PAT or existing deployment PAT)
        pat_to_test = pat or existing_pat

        if pat_to_test:
            if await asyncio.to_thread(
                validate_git_credential_access, repository_url, pat_to_test
            ):
                return GitRepository(
                    url=repository_url,
                    access_token=pat_to_test,
                )
            else:
                logger.warning(
                    "Generic repo inaccessible: %s — PAT rejected",
                    repository_url,
                )
                return InaccessibleRepository(
                    message="Personal Access Token does not have access to this repository.",
                )

        # No access method worked
        logger.warning(
            "Generic repo inaccessible: %s — private or does not exist",
            repository_url,
        )
        return InaccessibleRepository(
            message="Repository is private or does not exist."
        )

    async def _validate_pat_access(self, owner: str, repo: str, pat: str) -> bool:
        """Validate that a PAT has access to the repository by attempting to fetch it."""
        return bool(await pat_api_client(pat).get_repository_info(owner, repo))

    async def _check_pat_obsolete(self, deployment_id: str) -> bool:
        """Check if a deployment has PAT but GitHub App access is now available."""
        if not deployment_id:
            return False

        # Only obsolete if we have both PAT and GitHub App configured
        github_app_auth = get_github_app_auth()
        if not github_app_auth:
            return False

        return await k8s_client.has_deployment_pat(deployment_id)

    async def _check_has_existing_pat(self, deployment_id: str) -> bool:
        """Check if a deployment has PAT for a now-public non-GitHub repository."""
        if not deployment_id:
            return False

        # For non-GitHub repos, PAT is obsolete if repo is now public and deployment has PAT
        has_pat = await k8s_client.has_deployment_pat(deployment_id)
        return has_pat

    async def _construct_install_url(self, app: GitHubAppAuth, owner: str) -> str:
        """
        Construct a targeted GitHub App installation URL.

        Uses the owner's numeric ID to skip the account picker and jump straight
        to the installation flow for that specific account.

        https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-using-url-parameters
        """
        owner_info = await GitHubApiClient().get_owner_info(owner)
        if owner_info:
            return f"https://github.com/apps/{app.app_name}/installations/new/permissions?target_id={owner_info.id}"

        # Fallback to basic installation URL that shows account picker.
        # This should probably also return an error or warning, as it's likely to mean that the owner is invalid.
        # All owners should be accessible via this API.
        return f"https://github.com/apps/{app.app_name}/installations/new"

    async def _construct_settings_url(
        self, owner: str, installation_id: int
    ) -> str | None:
        """Construct a GitHub App installation settings URL.

        Returns the URL to manage an existing installation's repository access.
        The URL format differs for organizations vs personal accounts.
        """
        owner_info = await GitHubApiClient().get_owner_info(owner)
        if not owner_info:
            return None
        if owner_info.type == "Organization":
            return f"https://github.com/organizations/{owner}/settings/installations/{installation_id}"
        return f"https://github.com/settings/installations/{installation_id}"

    async def obtain_basic_auth_token(
        self,
        repository_url: str,
        deployment_id: str | None = None,
        pat: str | None = None,
    ) -> tuple[str | None, GitAccessType]:
        """
        Obtain a basic auth token for a repository.
        Returns a tuple of the basic auth token and the access type. (If token is none, you may want to check and validate uf the reason being InaccessibleRepository)
        """
        access = await self.get_access(repository_url, deployment_id, pat)

        auth = None
        match access:
            case GitHubAppAccess() as github_app_access:
                installation_auth = (
                    await github_app_api_client.get_installation_access_token(
                        github_app_access.installation_id
                    )
                )
                auth = f"x-access-token:{installation_auth}"
            case GitRepository() as git_repository:
                auth = git_repository.access_token

        return auth, access

    async def validate_git_application(
        self,
        repository_url: str,
        git_ref: str | None = None,
        deployment_file_path: str | None = None,
        deployment_id: str | None = None,
        pat: str | None = None,
    ) -> GitApplicationValidationResponse:
        """Verify that the specified configuration is 1. reachable, and 2. has a valid deployment file."""

        pat = (
            await k8s_client.get_deployment_pat(deployment_id)
            if deployment_id is not None and pat is None
            else pat
        )

        auth, access = await self.obtain_basic_auth_token(
            repository_url, deployment_id, pat
        )

        if isinstance(access, InaccessibleRepository):
            return GitApplicationValidationResponse(
                is_valid=False,
                error_message=access.message,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result = await asyncio.to_thread(
                    clone_repo,
                    repository_url,
                    git_ref=git_ref,
                    basic_auth=auth,
                    dest_dir=Path(temp_dir),
                )
            except GitAccessError as e:
                return GitApplicationValidationResponse(
                    is_valid=False,
                    error_message=e.message,
                )
            repo_root = Path(temp_dir)
            deployment_rel_path = deployment_file_path or DEFAULT_DEPLOYMENT_FILE_PATH
            # Parse config; this supersedes the older heuristic validate_deployment_file
            config_path = repo_root / deployment_rel_path
            try:
                config = read_deployment_config(repo_root, config_path)
                config.validate_config()
                is_valid = True
            except Exception as e:
                return GitApplicationValidationResponse(
                    is_valid=False,
                    error_message=f"Invalid deployment config: {str(e)}",
                )

            config_parent = resolve_config_parent(repo_root, config_path)
            ui_dist_relative_to_config = ui_build_output_path(config_parent, config)
            ui_dist_relative_to_repo = (
                (config_parent / ui_dist_relative_to_config).relative_to(repo_root)
                if ui_dist_relative_to_config
                else None
            )

            return GitApplicationValidationResponse(
                is_valid=is_valid,
                git_sha=result.git_sha,
                git_ref=result.git_ref,
                valid_deployment_file_path=deployment_file_path,
                ui_build_output_path=ui_dist_relative_to_repo,
            )


git_service = GitService()
