"""Git HTTP server backed by dulwich and S3.

Provides WSGI-based git serving for both the manage API (read+write)
and the build API (read-only). Bare repos are stored as tarballs in S3.

The readonly path uses ``a2wsgi.WSGIMiddleware`` for true bidirectional
streaming (no full-body buffering).  The read+write path spools the
request body to a ``SpooledTemporaryFile`` to cap memory usage while
preserving post-processing (ref diff, S3 upload, callback).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from tempfile import SpooledTemporaryFile
from typing import Any, cast

from a2wsgi.asgi_typing import HTTPScope
from a2wsgi.wsgi import Body, WSGIResponder, build_environ
from a2wsgi.wsgi_typing import Environ
from dulwich.refs import Ref
from dulwich.repo import Repo
from dulwich.server import BackendRepo, DictBackend
from dulwich.web import make_wsgi_chain
from fastapi import Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .storage import CodeRepoStorage

logger = logging.getLogger(__name__)

# Shared thread pool for WSGIMiddleware instances (readonly path).
_wsgi_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="WSGI-git")

# Request bodies smaller than this stay in memory; larger ones spool to disk.
_SPOOL_MAX_SIZE = 10 * 1024 * 1024  # 10 MB


def _create_wsgi_app(
    repo: Repo,
) -> Any:
    """Create a dulwich WSGI app for the given bare repo."""
    backend = DictBackend({"/": cast("BackendRepo", repo)})
    return make_wsgi_chain(backend)


def _call_wsgi(
    wsgi_app: Any,
    environ: Environ,
) -> tuple[int, list[tuple[str, str]], bytes]:
    """Call a WSGI app synchronously, returning (status_code, headers, body)."""
    status_code = 500
    response_headers: list[tuple[str, str]] = []
    body_parts: list[bytes] = []

    def start_response(
        status: str,
        headers: list[tuple[str, str]],
        exc_info: Any = None,
    ) -> Any:
        nonlocal status_code, response_headers
        status_code = int(status.split(" ", 1)[0])
        response_headers = list(headers)
        return lambda s: body_parts.append(s)

    result = wsgi_app(environ, start_response)
    try:
        for chunk in result:
            body_parts.append(chunk)
    finally:
        # PEP 3333: WSGI iterators may optionally have a close() method
        if hasattr(result, "close"):
            result.close()

    return status_code, response_headers, b"".join(body_parts)


def _git_scope(request: Request, git_path: str) -> HTTPScope:
    """Build an ASGI scope with the path overridden to the git sub-path."""
    scope = dict(request.scope)
    scope["path"] = f"/{git_path}"
    scope["root_path"] = ""
    return cast(HTTPScope, scope)


_HOP_BY_HOP_HEADERS = frozenset(("transfer-encoding", "connection"))


async def _serve_wsgi_git(
    request: Request,
    repo: Repo,
    git_path: str,
) -> tuple[int, dict[str, str], bytes]:
    """Run a dulwich WSGI git request and return (status, headers, body).

    The request body is spooled to a ``SpooledTemporaryFile`` so that
    large pack files (e.g. from ``git push``) don't exhaust memory.
    """
    wsgi_app = _create_wsgi_app(repo)

    spooled: SpooledTemporaryFile[bytes] = SpooledTemporaryFile(
        max_size=_SPOOL_MAX_SIZE
    )
    try:
        async for chunk in request.stream():
            spooled.write(chunk)
        content_length = spooled.tell()
        spooled.seek(0)

        environ = build_environ(_git_scope(request, git_path), cast(Body, spooled))

        # The ASGI server (uvicorn) already de-chunks the request body, but
        # build_environ preserves the original Transfer-Encoding header.
        # Dulwich's WSGI handler tries to de-chunk the body when it sees
        # this header, causing a ValueError on already-de-chunked data.
        # Set the real content length and remove the stale header.
        environ["CONTENT_LENGTH"] = str(content_length)
        environ.pop("HTTP_TRANSFER_ENCODING", None)

        status_code, response_headers, response_body = await run_in_threadpool(
            _call_wsgi, wsgi_app, environ
        )
    finally:
        spooled.close()

    headers = {
        k: v for k, v in response_headers if k.lower() not in _HOP_BY_HOP_HEADERS
    }

    return status_code, headers, response_body


def _get_resolved_refs(repo: Repo) -> dict[Ref, bytes]:
    """Safely get resolved refs, handling unresolvable symbolic refs.

    ``repo.get_refs()`` raises ``KeyError`` when HEAD is a symbolic ref
    pointing to a branch that does not yet exist (e.g. freshly initialised
    bare repo).  This helper skips refs that cannot be resolved.
    """
    result: dict[Ref, bytes] = {}
    for key in repo.refs.allkeys():
        try:
            result[key] = repo.refs[key]
        except KeyError:
            continue
    return result


async def handle_git_request(
    request: Request,
    deployment_id: str,
    git_path: str,
    storage: CodeRepoStorage,
    on_push_complete: Callable[[str, str | None, str | None], Awaitable[None]]
    | None = None,
) -> Response:
    """Handle a git HTTP request (read+write).

    Downloads the bare repo from S3, serves the git request via dulwich,
    and if refs changed (receive-pack), uploads the updated repo back to S3.

    Args:
        request: The incoming HTTP request.
        deployment_id: The deployment to serve.
        git_path: The git-specific path (e.g., "info/refs", "git-receive-pack").
        storage: The CodeRepoStorage instance.
        on_push_complete: Optional async callback called after a successful push.
            Called with (deployment_id, new_sha, git_ref).
    """
    repo_path = await storage.download_repo(deployment_id)
    if repo_path is None:
        repo_path = CodeRepoStorage.init_bare_repo(deployment_id)

    try:
        with Repo(str(repo_path)) as repo:
            refs_before = _get_resolved_refs(repo)
            status_code, headers, response_body = await _serve_wsgi_git(
                request, repo, git_path
            )
            refs_after = _get_resolved_refs(repo)

        new_sha: str | None = None
        git_ref: str | None = None
        if refs_before != refs_after:
            # Find the first changed non-HEAD ref — this is the branch
            # that was actually pushed (e.g. refs/heads/my-feature).
            # Use its SHA directly rather than trying to resolve HEAD,
            # which may point to a different branch (e.g. dulwich defaults
            # HEAD to refs/heads/master on a fresh bare repo).
            for ref_name in refs_after:
                if ref_name == b"HEAD":
                    continue
                if (
                    ref_name not in refs_before
                    or refs_after[ref_name] != refs_before[ref_name]
                ):
                    new_sha = refs_after[ref_name].decode()
                    git_ref = ref_name.decode().removeprefix("refs/heads/")
                    break

            logger.info(
                "Refs changed for deployment %s, uploading to S3",
                deployment_id,
            )
            await storage.upload_repo(deployment_id, repo_path)

            if on_push_complete:
                await on_push_complete(deployment_id, new_sha, git_ref)

        return Response(
            content=response_body,
            status_code=status_code,
            headers=headers,
        )
    finally:
        if repo_path:
            shutil.rmtree(repo_path.parent, ignore_errors=True)


class _StreamingWSGIResponse(Response):
    """A Response that delegates to a2wsgi.WSGIMiddleware for streaming.

    Starlette calls ``Response.__call__(scope, receive, send)`` to send
    the response.  This subclass overrides ``__call__`` to forward
    directly to the WSGIMiddleware ASGI app, giving us true streaming
    without buffering the entire response body in memory.

    Cleanup (temp directory removal) runs in the ``finally`` block after
    ``WSGIMiddleware.__call__`` returns — which only happens after the
    response is fully sent.
    """

    def __init__(
        self,
        asgi_app: WSGIResponder,
        git_scope: HTTPScope,
        receive: Any,
        cleanup: Callable[[], None] | None,
    ) -> None:
        self.background = None
        self.asgi_app = asgi_app
        self.git_scope = git_scope
        self._receive = receive
        self.cleanup = cleanup

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await self.asgi_app(self.git_scope, self._receive, send)
        finally:
            if self.cleanup:
                self.cleanup()


async def handle_git_request_readonly(
    request: Request,
    deployment_id: str,
    git_path: str,
    storage: CodeRepoStorage,
) -> Response:
    """Handle a read-only git HTTP request (upload-pack only).

    Used by the build API for git clone operations.  The response is
    streamed directly via ``a2wsgi.WSGIMiddleware`` — never fully
    buffered in memory.
    """
    repo_path = await storage.download_repo(deployment_id)
    if repo_path is None:
        return Response(
            content="No code has been pushed to this deployment yet.",
            status_code=404,
        )

    # Reject receive-pack requests
    if "git-receive-pack" in git_path:
        shutil.rmtree(repo_path.parent, ignore_errors=True)
        return Response(
            content="Push not allowed on this endpoint.",
            status_code=403,
        )

    repo: Repo | None = None
    try:
        repo = Repo(str(repo_path))
        wsgi_app = _create_wsgi_app(repo)
        responder = WSGIResponder(wsgi_app, _wsgi_executor, send_queue_size=10)
    except Exception:
        if repo is not None:
            repo.close()
        shutil.rmtree(repo_path.parent, ignore_errors=True)
        raise

    def _cleanup() -> None:
        repo.close()
        shutil.rmtree(repo_path.parent, ignore_errors=True)

    return _StreamingWSGIResponse(
        asgi_app=responder,
        git_scope=_git_scope(request, git_path),
        receive=request.receive,
        cleanup=_cleanup,
    )
