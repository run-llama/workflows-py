from datetime import datetime
from typing import Any

import pydantic


class GitHubOwnerInfo(pydantic.BaseModel):
    id: int
    login: str
    type: str  # "User" or "Organization"


class GitHubRepository(pydantic.BaseModel):
    id: int | None = None
    name: str | None = None
    full_name: str | None = None
    owner: GitHubOwnerInfo | None = None
    private: bool = False
    html_url: str | None = None
    description: str | None = None
    fork: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    clone_url: str | None = None
    ssh_url: str | None = None
    size: int = 0
    stargazers_count: int = 0
    watchers_count: int = 0
    language: str | None = None
    forks_count: int = 0
    archived: bool = False
    disabled: bool = False
    open_issues_count: int = 0
    topics: list[str] = []
    visibility: str | None = None
    default_branch: str | None = None


class GithubAppInstallation(pydantic.BaseModel):
    id: int | None = None
    account: dict[str, Any] | None = None
    repository_selection: str | None = (
        None  # must making these optional since idk what is
    )
    access_tokens_url: str | None = None
    repositories_url: str | None = None
    html_url: str | None = None
    permissions: dict[str, str] | None = None
    events: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    single_file_name: str | None = None
    has_multiple_single_files: bool | None = None
    single_file_paths: list[str] | None = None
    app_slug: str | None = None
    # Not sure of these types
    # suspended_at: datetime | None = None
    # suspended_by: dict[str, Any] | None = None

    # Example response:
    # {
    #   "id": 1,
    #   "account": {
    #     "login": "github",
    #     "id": 1,
    #     "node_id": "MDEyOk9yZ2FuaXphdGlvbjE=",
    #     "avatar_url": "https://github.com/images/error/hubot_happy.gif",
    #     "gravatar_id": "",
    #     "url": "https://api.github.com/orgs/github",
    #     "html_url": "https://github.com/github",
    #     "followers_url": "https://api.github.com/users/github/followers",
    #     "following_url": "https://api.github.com/users/github/following{/other_user}",
    #     "gists_url": "https://api.github.com/users/github/gists{/gist_id}",
    #     "starred_url": "https://api.github.com/users/github/starred{/owner}{/repo}",
    #     "subscriptions_url": "https://api.github.com/users/github/subscriptions",
    #     "organizations_url": "https://api.github.com/users/github/orgs",
    #     "repos_url": "https://api.github.com/orgs/github/repos",
    #     "events_url": "https://api.github.com/orgs/github/events",
    #     "received_events_url": "https://api.github.com/users/github/received_events",
    #     "type": "Organization",
    #     "site_admin": false
    #   },
    #   "repository_selection": "all",
    #   "access_tokens_url": "https://api.github.com/app/installations/1/access_tokens",
    #   "repositories_url": "https://api.github.com/installation/repositories",
    #   "html_url": "https://github.com/organizations/github/settings/installations/1",
    #   "app_id": 1,
    #   "client_id": "Iv1.ab1112223334445c",
    #   "target_id": 1,
    #   "target_type": "Organization",
    #   "permissions": {
    #     "checks": "write",
    #     "metadata": "read",
    #     "contents": "read"
    #   },
    #   "events": [
    #     "push",
    #     "pull_request"
    #   ],
    #   "created_at": "2018-02-09T20:51:14Z",
    #   "updated_at": "2018-02-09T20:51:14Z",
    #   "single_file_name": "config.yml",
    #   "has_multiple_single_files": true,
    #   "single_file_paths": [
    #     "config.yml",
    #     ".github/issue_TEMPLATE.md"
    #   ],
    #   "app_slug": "github-actions",
    #   "suspended_at": null,
    #   "suspended_by": null
    # }
