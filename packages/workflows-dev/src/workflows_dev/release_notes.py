from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import httpx

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"


@dataclass(frozen=True)
class Repository:
    owner: str
    name: str


@dataclass
class PullRequest:
    number: int
    title: str
    author: str
    merge_commit_sha: Optional[str]
    merged: bool
    labels: tuple[str, ...]


class GitHubClient:
    """Thin wrapper around httpx for GitHub REST calls."""

    def __init__(self, token: str) -> None:
        self._client = httpx.Client(
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "workflows-dev-cli",
                "X-GitHub-Api-Version": API_VERSION,
            },
            timeout=30.0,
        )

    def get(self, url: str, params: Optional[Mapping[str, str]] = None) -> httpx.Response:
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()


def parse_repository(value: str) -> Repository:
    if "/" not in value:
        raise ValueError("Repository must be in the form OWNER/NAME")
    owner, name = value.split("/", 1)
    return Repository(owner=owner, name=name)


def parse_pull_request(raw: Mapping[str, object]) -> PullRequest:
    user = raw.get("user")
    author = ""
    if isinstance(user, Mapping):
        login = user.get("login")
        author = str(login) if isinstance(login, str) else ""

    labels: list[str] = []
    raw_labels = raw.get("labels", [])
    if isinstance(raw_labels, Iterable):
        for label in raw_labels:
            if isinstance(label, Mapping):
                name = label.get("name")
                if isinstance(name, str):
                    labels.append(name)

    return PullRequest(
        number=int(raw["number"]),
        title=str(raw["title"]),
        author=author or "unknown",
        merge_commit_sha=str(raw.get("merge_commit_sha") or "") or None,
        merged=bool(raw.get("merged_at")),
        labels=tuple(labels),
    )


def parse_link_header(header_value: Optional[str]) -> dict[str, str]:
    links: dict[str, str] = {}
    if not header_value:
        return links
    parts = [chunk.strip() for chunk in header_value.split(",")]
    for part in parts:
        if not part:
            continue
        section, _, rel = part.partition(";")
        url = section.strip().strip("<>")
        rel = rel.strip()
        if rel.startswith('rel="') and rel.endswith('"'):
            links[rel[5:-1]] = url
    return links


def fetch_pull_requests(client: GitHubClient, repo: Repository) -> list[PullRequest]:
    prs: list[PullRequest] = []
    url = f"{API_ROOT}/repos/{repo.owner}/{repo.name}/pulls"
    params = {"state": "closed", "per_page": "100", "sort": "updated", "direction": "desc"}

    while url:
        response = client.get(url, params=params)
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected response while listing pull requests.")
        for entry in data:
            if isinstance(entry, Mapping):
                prs.append(parse_pull_request(entry))
        params = None
        links = parse_link_header(response.headers.get("Link"))
        url = links.get("next")

    return prs


def collect_commit_shas(
    client: GitHubClient, repo: Repository, previous_tag: str, current_tag: str
) -> Optional[set[str]]:
    if not previous_tag:
        return None
    url = f"{API_ROOT}/repos/{repo.owner}/{repo.name}/compare/{previous_tag}...{current_tag}"
    try:
        response = client.get(url)
    except httpx.HTTPStatusError:
        return None
    data = response.json()
    commits = data.get("commits")
    if not isinstance(commits, list):
        return None
    return {
        str(commit.get("sha"))
        for commit in commits
        if isinstance(commit, Mapping) and isinstance(commit.get("sha"), str)
    }


def filter_pull_requests(
    pull_requests: Iterable[PullRequest],
    package_label: str,
    commit_shas: Optional[set[str]],
) -> list[PullRequest]:
    relevant: list[PullRequest] = []
    for pr in pull_requests:
        if not pr.merged or not pr.merge_commit_sha:
            continue
        if package_label not in pr.labels:
            continue
        if commit_shas is None or pr.merge_commit_sha in commit_shas:
            relevant.append(pr)
    return relevant


def format_release_notes(
    repo: Repository,
    package_name: str,
    semver: str,
    current_tag: str,
    previous_tag: str,
    pull_requests: Iterable[PullRequest],
) -> str:
    lines: list[str] = [f"## {package_name} {semver}", ""]
    added = False
    for pr in pull_requests:
        lines.append(f"- {pr.title} (#{pr.number}) by @{pr.author}")
        added = True
    if not added:
        lines.append("_No labeled pull requests for this package in this release._")
    if previous_tag:
        lines.append("")
        lines.append(
            f"[View changes between {previous_tag} and {current_tag}]"
            f"(https://github.com/{repo.owner}/{repo.name}/compare/{previous_tag}...{current_tag})."
        )
    return "\n".join(lines).strip()


def generate_release_notes(
    token: str,
    repository: str,
    package_label: str,
    package_name: str,
    current_tag: str,
    previous_tag: str,
    semver: str,
) -> str:
    repo = parse_repository(repository)
    client = GitHubClient(token)
    try:
        pull_requests = fetch_pull_requests(client, repo)
        commit_shas = collect_commit_shas(client, repo, previous_tag, current_tag)
        relevant = filter_pull_requests(pull_requests, package_label, commit_shas)
        return format_release_notes(repo, package_name, semver, current_tag, previous_tag, relevant)
    finally:
        client.close()

