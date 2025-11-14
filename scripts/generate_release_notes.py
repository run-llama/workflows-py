from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
from urllib import error, request

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"


@dataclass
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
    labels: Sequence[str]


class GitHubClient:
    def __init__(self, token: str) -> None:
        self._token = token

    def request_json(self, url: str) -> tuple[object, Mapping[str, str]]:
        req = request.Request(url)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", API_VERSION)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("User-Agent", "workflow-release-script")
        try:
            with request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data, resp.headers
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"GitHub API call failed ({exc.code} {exc.reason}): {detail}") from exc


def parse_repository(value: str) -> Repository:
    if "/" not in value:
        raise ValueError(f"Repository value '{value}' must be in the form OWNER/REPO.")
    owner, name = value.split("/", 1)
    return Repository(owner=owner, name=name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate release notes for a package.")
    parser.add_argument("--repository", required=True, help="GitHub repository in OWNER/REPO format.")
    parser.add_argument("--github-token", required=True, help="GitHub token with repo scope.")
    parser.add_argument("--package-label", required=True, help="Label used to select pull requests.")
    parser.add_argument("--package-name", required=True, help="Human readable package name.")
    parser.add_argument("--current-tag", required=True, help="Current release tag.")
    parser.add_argument("--previous-tag", default="", help="Previous release tag (if available).")
    parser.add_argument("--semver", required=True, help="Semantic version being released.")
    parser.add_argument("--output", required=True, help="Path to the GitHub Actions output file.")
    return parser.parse_args()


def parse_pull_request(raw: Mapping[str, object]) -> PullRequest:
    user = raw.get("user") or {}
    author = user.get("login") if isinstance(user, Mapping) else None
    labels = []
    raw_labels = raw.get("labels") or []
    if isinstance(raw_labels, Iterable):
        for label in raw_labels:
            if isinstance(label, Mapping):
                name = label.get("name")
                if isinstance(name, str):
                    labels.append(name)
    return PullRequest(
        number=int(raw["number"]),
        title=str(raw["title"]),
        author=str(author or "unknown"),
        merge_commit_sha=str(raw.get("merge_commit_sha") or "") or None,
        merged=bool(raw.get("merged_at")),
        labels=labels,
    )


def parse_link_header(header_value: Optional[str]) -> dict[str, str]:
    links: dict[str, str] = {}
    if not header_value:
        return links
    for part in header_value.split(","):
        section = part.strip()
        if not section:
            continue
        url_part, _, rel_part = section.partition(";")
        url = url_part.strip().strip("<>")
        rel = rel_part.strip()
        if rel.startswith('rel="') and rel.endswith('"'):
            rel_value = rel[5:-1]
            links[rel_value] = url
    return links


def fetch_pull_requests(client: GitHubClient, repo: Repository) -> list[PullRequest]:
    prs: list[PullRequest] = []
    url = (
        f"{API_ROOT}/repos/{repo.owner}/{repo.name}/pulls"
        "?state=closed&per_page=100&sort=updated&direction=desc"
    )
    while url:
        data, headers = client.request_json(url)
        if not isinstance(data, list):
            raise RuntimeError("Unexpected response while listing pull requests.")
        for raw_pr in data:
            if isinstance(raw_pr, Mapping):
                prs.append(parse_pull_request(raw_pr))
        links = parse_link_header(headers.get("Link"))
        url = links.get("next")
    return prs


def collect_commit_shas(
    client: GitHubClient, repo: Repository, previous_tag: str, current_tag: str
) -> Optional[set[str]]:
    if not previous_tag:
        return None
    url = f"{API_ROOT}/repos/{repo.owner}/{repo.name}/compare/{previous_tag}...{current_tag}"
    try:
        data, _ = client.request_json(url)
    except RuntimeError as exc:
        print(f"Warning: unable to compare {previous_tag}...{current_tag}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, Mapping):
        print("Warning: unexpected compare response shape.", file=sys.stderr)
        return None
    commits = data.get("commits")
    if not isinstance(commits, list):
        return None
    return {str(commit.get("sha")) for commit in commits if isinstance(commit, Mapping) and commit.get("sha")}


def filter_pull_requests(
    pull_requests: Sequence[PullRequest],
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
    pull_requests: Sequence[PullRequest],
) -> str:
    lines: list[str] = [f"## {package_name} {semver}", ""]
    if not pull_requests:
        lines.append("_No labeled pull requests for this package in this release._")
    else:
        for pr in pull_requests:
            lines.append(f"- {pr.title} (#{pr.number}) by @{pr.author}")
    if previous_tag:
        lines.append("")
        lines.append(
            f"[View changes between {previous_tag} and {current_tag}]"
            f"(https://github.com/{repo.owner}/{repo.name}/compare/{previous_tag}...{current_tag})."
        )
    return "\n".join(lines).strip()


def write_output(output_path: Path, body: str) -> None:
    output_path.write_text(f"body={body}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = parse_repository(args.repository)
    client = GitHubClient(args.github_token)
    pull_requests = fetch_pull_requests(client, repo)
    commit_shas = collect_commit_shas(client, repo, args.previous_tag, args.current_tag)
    relevant = filter_pull_requests(pull_requests, args.package_label, commit_shas)
    body = format_release_notes(
        repo=repo,
        package_name=args.package_name,
        semver=args.semver,
        current_tag=args.current_tag,
        previous_tag=args.previous_tag,
        pull_requests=relevant,
    )
    write_output(Path(args.output), body)


if __name__ == "__main__":
    main()

