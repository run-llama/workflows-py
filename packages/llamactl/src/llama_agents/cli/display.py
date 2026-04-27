# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CLI-side display projection of ``DeploymentResponse``.

The wire model (``llama_agents.core.schema.deployments.DeploymentResponse``)
carries deprecated ``@computed_field`` aliases (``name``, ``llama_deploy_version``)
and a flag (``has_personal_access_token``) that we don't want to leak into the
public CLI output. ``DeploymentDisplay`` is the canonical CLI shape:

- ``name`` (the stable id) lives at the top level.
- Editable fields live under a nested ``spec:`` block (kubectl-style).
- Read-only / system-set fields live under a nested ``status:`` block.
- Secrets are masked; ``personal_access_token`` is presented as a top-level
  field of ``spec`` even though server-side it is the ``GITHUB_PAT`` secret.
  We document the leaky abstraction here and otherwise hide it.

This is the contract Slice B's ``apply -f`` consumes as input — see
``thoughts/shared/plans/2026-04-26-llamactl-slice-a5-display-model.md`` for the
input-tolerance rules (top-level ``status:`` is stripped before validation;
``spec:`` is the editable surface).
"""

from __future__ import annotations

from typing import Any

from llama_agents.core.schema.deployments import (
    DeploymentResponse,
    LlamaDeploymentPhase,
)
from pydantic import BaseModel, ConfigDict

SECRET_MASK = "********"


class DeploymentSpec(BaseModel):
    """Editable deployment fields.

    These are the fields a user can set via ``apply``. ``personal_access_token``
    is a leaky abstraction over the server-side ``GITHUB_PAT`` secret — it is
    surfaced here as a dedicated field rather than mixed into ``secrets`` so
    the apply input shape is explicit.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str
    repo_url: str
    deployment_file_path: str
    git_ref: str | None = None
    appserver_version: str | None = None
    suspended: bool = False
    secrets: dict[str, str] | None = None
    personal_access_token: str | None = None


class DeploymentStatus(BaseModel):
    """Read-only / system-set deployment status block.

    These fields reflect runtime state and are not editable via ``apply``.
    ``warning`` is intentionally always serialized (explicit ``null`` when no
    warning is present) — absence vs. explicit-null carries meaning for the
    future ``describe`` command.
    """

    model_config = ConfigDict(extra="forbid")

    phase: LlamaDeploymentPhase
    git_sha: str | None = None
    apiserver_url: str | None = None
    project_id: str
    warning: str | None = None


class DeploymentDisplay(BaseModel):
    """CLI projection of a deployment.

    Top-level ``name`` is the stable id (immutable on update). ``spec``
    carries the editable surface. ``status`` carries everything set by the
    server. Use :meth:`from_response` to translate a ``DeploymentResponse``
    into this shape; use :meth:`to_output_dict` to obtain a dict suitable for
    JSON/YAML emission with the omit-when-empty rules applied.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    spec: DeploymentSpec
    status: DeploymentStatus | None = None

    @classmethod
    def from_response(cls, r: DeploymentResponse) -> DeploymentDisplay:
        """Project a wire ``DeploymentResponse`` into the CLI display shape."""
        secret_names = r.secret_names or []
        secrets: dict[str, str] | None = (
            {name: SECRET_MASK for name in secret_names} if secret_names else None
        )
        pat = SECRET_MASK if r.has_personal_access_token else None
        spec = DeploymentSpec(
            display_name=r.display_name,
            repo_url=r.repo_url,
            deployment_file_path=r.deployment_file_path,
            git_ref=r.git_ref,
            appserver_version=r.appserver_version,
            suspended=r.suspended,
            secrets=secrets,
            personal_access_token=pat,
        )
        status = DeploymentStatus(
            phase=r.status,
            git_sha=r.git_sha,
            apiserver_url=str(r.apiserver_url) if r.apiserver_url else None,
            project_id=r.project_id,
            warning=r.warning,
        )
        return cls(name=r.id, spec=spec, status=status)

    def to_output_dict(self) -> dict[str, Any]:
        """Return the dict shape used for JSON/YAML rendering.

        Omits fields inside ``spec`` whose value is None (e.g., unset
        ``personal_access_token``, empty ``secrets``). The nested ``status``
        block is preserved verbatim so its ``warning`` key remains explicit
        even when ``null``.
        """
        spec_data = self.spec.model_dump(mode="json")
        spec_data = {k: v for k, v in spec_data.items() if v is not None}
        data: dict[str, Any] = {"name": self.name, "spec": spec_data}
        if self.status is not None:
            data["status"] = self.status.model_dump(mode="json")
        return data
