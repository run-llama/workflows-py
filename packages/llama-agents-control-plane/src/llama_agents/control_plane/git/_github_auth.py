import datetime
import functools
import os
from pathlib import Path

import jwt


class GitHubAppAuth:
    """Handles GitHub App authentication and JWT generation."""

    jwt: str | None = None
    jwt_expires_at: datetime.datetime | None = None
    # max 10 minutes for github apps
    jwt_expiration_seconds: int = 600

    def __init__(self, client_id: str, private_key: str | Path, app_name: str):
        """
        Initialize GitHub App authentication.

        Args:
            app_id: GitHub App ID
            private_key: Either the private key content as string or path to private key file
        """
        self.client_id = client_id
        self.app_name = app_name
        self.private_key = self._load_private_key(private_key)

    def _load_private_key(self, private_key: str | Path) -> str:
        """Load private key from string or file path."""
        if isinstance(private_key, Path) or (
            isinstance(private_key, str) and private_key.startswith("/")
        ):
            # Treat as file path
            key_path = Path(private_key)
            if not key_path.exists():
                raise FileNotFoundError(f"Private key file not found: {key_path}")
            return key_path.read_text().strip()
        else:
            # Treat as key content
            return private_key.strip()

    def get_jwt(self) -> str:
        """Get a JWT for GitHub App authentication."""
        if self.jwt is None:
            expiration, jwt = self._generate_jwt()
            self.jwt_expires_at = expiration
            self.jwt = jwt
        elif (
            self.jwt_expires_at is not None
            and self.jwt_expires_at
            < datetime.datetime.now(tz=datetime.timezone.utc)
            + datetime.timedelta(seconds=self.jwt_expiration_seconds / 20)
        ):
            expiration, jwt = self._generate_jwt()
            self.jwt_expires_at = expiration
            self.jwt = jwt
        return self.jwt

    def _generate_jwt(self) -> tuple[datetime.datetime, str]:
        """
        Generate a JWT for GitHub App authentication.

        Args:
            expiration_seconds: JWT expiration time in seconds (max 600 for GitHub Apps)

        Returns:
            JWT token string

        Raises:
            ValueError: If expiration exceeds GitHub's 10-minute limit
        """

        now = datetime.datetime.now(tz=datetime.timezone.utc)
        expires_at = now + datetime.timedelta(seconds=self.jwt_expiration_seconds)
        payload = {
            "iss": self.client_id,  # Issuer: GitHub App ID
            "iat": int(now.timestamp()),  # Issued at
            "exp": int(  # Expiration
                (
                    now + datetime.timedelta(seconds=self.jwt_expiration_seconds)
                ).timestamp()
            ),
        }

        return expires_at, jwt.encode(payload, self.private_key, algorithm="RS256")


@functools.lru_cache(maxsize=None)
def get_github_app_auth() -> GitHubAppAuth | None:
    """Get the GitHubAppAuth instance."""
    client_id = os.getenv("GITHUB_APP_CLIENT_ID")
    private_key = os.getenv("GITHUB_APP_PRIVATE_KEY")
    app_name = os.getenv("GITHUB_APP_NAME")

    if client_id is None or private_key is None or app_name is None:
        return None

    return GitHubAppAuth(client_id, private_key, app_name)
