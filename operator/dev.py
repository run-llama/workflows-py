#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["click"]
# ///
"""
Local development environment for cloud_llama_deploy.

  up     — create/ensure cluster and start tilt
  down   — tear down tilt resources, retaining data
  down --delete — also delete the kind cluster (kind target only)

Targets:
  kind            — (default) creates a kind cluster
  docker-desktop  — uses Docker Desktop's built-in Kubernetes
"""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import click

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
NAMESPACE = "llama-agents"

TARGETS = ("kind", "docker-desktop")
K8S_CONTEXTS = {
    "kind": "kind-kind",
    "docker-desktop": "docker-desktop",
}


def run(
    cmd: list[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd, check=False, capture_output=capture, text=True, cwd=PROJECT_ROOT
    )
    if check and result.returncode != 0:
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result


# ---------------------------------------------------------------------------
# kind helpers
# ---------------------------------------------------------------------------


def kind_cluster_exists() -> bool:
    result = run(["kind", "get", "clusters"], check=False, capture=True)
    return "kind" in result.stdout


def ensure_kind_cluster() -> None:
    if kind_cluster_exists():
        result = run(
            ["kind", "export", "kubeconfig", "--name", "kind"],
            check=False,
            capture=True,
        )
        if result.returncode == 0:
            return
        print("Cluster 'kind' exists but is broken, recreating...")
        run(["kind", "delete", "cluster", "--name", "kind"])

    print("Creating kind cluster 'kind'...")

    kind_config = dedent("""\
        kind: Cluster
        apiVersion: kind.x-k8s.io/v1alpha4
        nodes:
        - role: control-plane
          kubeadmConfigPatches:
          - |
            kind: InitConfiguration
            nodeRegistration:
              kubeletExtraArgs:
                node-labels: "ingress-ready=true"
          extraPortMappings:
          - containerPort: 80
            hostPort: 8090
            protocol: TCP
          - containerPort: 443
            hostPort: 8444
            protocol: TCP
    """)

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(kind_config)
        config_path = f.name

    try:
        run(
            [
                "kind",
                "create",
                "cluster",
                "--name",
                "kind",
                "--config",
                config_path,
            ]
        )
    finally:
        os.unlink(config_path)

    install_ingress_controller("kind")


# ---------------------------------------------------------------------------
# docker-desktop helpers
# ---------------------------------------------------------------------------


def ensure_docker_desktop_cluster() -> None:
    context = K8S_CONTEXTS["docker-desktop"]
    result = run(
        ["kubectl", "--context", context, "cluster-info"],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        print(
            f"Kubernetes context '{context}' is not reachable.\n"
            "Enable Kubernetes in Docker Desktop → Settings → Kubernetes.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Switch to the docker-desktop context
    run(["kubectl", "config", "use-context", context])
    install_ingress_controller("docker-desktop")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

INGRESS_MANIFESTS = {
    "kind": "https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml",
    "docker-desktop": "https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml",
}


def install_ingress_controller(target: str) -> None:
    result = run(
        ["kubectl", "get", "namespace", "ingress-nginx"], check=False, capture=True
    )
    if result.returncode == 0:
        return

    print("Installing nginx ingress controller...")
    run(["kubectl", "apply", "-f", INGRESS_MANIFESTS[target]])

    import time

    start = time.time()
    while time.time() - start < 10:
        result = run(
            [
                "kubectl",
                "get",
                "pods",
                "--namespace",
                "ingress-nginx",
                "--selector=app.kubernetes.io/component=controller",
                "--no-headers",
            ],
            check=False,
            capture=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            break
        time.sleep(1)

    run(
        [
            "kubectl",
            "wait",
            "--namespace",
            "ingress-nginx",
            "--for=condition=ready",
            "pod",
            "--selector=app.kubernetes.io/component=controller",
            "--timeout=300s",
        ]
    )


@click.group()
@click.option(
    "--target",
    type=click.Choice(TARGETS),
    default="kind",
    envvar="DEV_TARGET",
    show_default=True,
    help="Kubernetes target cluster.",
)
@click.pass_context
def cli(ctx: click.Context, target: str) -> None:
    """Local development environment for cloud_llama_deploy."""
    ctx.ensure_object(dict)
    ctx.obj["target"] = target


@cli.command()
@click.pass_context
def up(ctx: click.Context) -> None:
    """Create/ensure cluster and start tilt."""
    target: str = ctx.obj["target"]

    # Check required tools
    version_cmds: dict[str, list[str]] = {
        "kubectl": ["kubectl", "version", "--client"],
        "docker": ["docker", "--version"],
        "tilt": ["tilt", "version"],
    }
    if target == "kind":
        version_cmds["kind"] = ["kind", "--version"]

    for tool, cmd in version_cmds.items():
        if run(cmd, check=False, capture=True).returncode != 0:
            print(f"Missing required tool: {tool}", file=sys.stderr)
            sys.exit(1)

    if target == "kind":
        ensure_kind_cluster()
    else:
        ensure_docker_desktop_cluster()

    # Ensure namespace exists
    result = run(["kubectl", "get", "namespace", NAMESPACE], check=False, capture=True)
    if result.returncode != 0:
        run(["kubectl", "create", "namespace", NAMESPACE])

    if not (PROJECT_ROOT / ".env").exists():
        print(
            "Note: no .env file found. GitHub integration requires GITHUB_APP_PRIVATE_KEY, GITHUB_APP_CLIENT_ID, GITHUB_APP_NAME, GITHUB_APP_SECRET."
        )

    ingress_port = "8090" if target == "kind" else "80"
    print("Starting tilt...")
    print("  API:     http://localhost:8011")
    print("  Tilt UI: http://localhost:10350")
    print(f"  Ingress: *.127.0.0.1.nip.io:{ingress_port}")
    os.execvp(
        "tilt",
        [
            "tilt",
            "up",
            "-f",
            str(PROJECT_ROOT / "operator" / "Tiltfile"),
            "--",
            f"--target={target}",
        ],
    )


@cli.command()
@click.option("--delete", is_flag=True, help="Also delete the kind cluster")
@click.pass_context
def down(ctx: click.Context, delete: bool) -> None:
    """Tear down tilt resources. Use --delete to also remove the cluster (kind only)."""
    target: str = ctx.obj["target"]

    run(
        ["tilt", "down", "-f", str(PROJECT_ROOT / "operator" / "Tiltfile")], check=False
    )

    if delete:
        if target != "kind":
            print("--delete only applies to the kind target", file=sys.stderr)
            return
        if kind_cluster_exists():
            print("Deleting kind cluster 'kind'...")
            run(["kind", "delete", "cluster", "--name", "kind"])


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show cluster and deployment status."""
    target: str = ctx.obj["target"]
    context = K8S_CONTEXTS[target]

    if target == "kind":
        if not kind_cluster_exists():
            print("No kind cluster 'kind' found")
            return
        run(["kind", "export", "kubeconfig", "--name", "kind"])
    else:
        result = run(
            ["kubectl", "--context", context, "cluster-info"],
            check=False,
            capture=True,
        )
        if result.returncode != 0:
            print(f"Kubernetes context '{context}' is not reachable")
            return
        run(["kubectl", "config", "use-context", context])

    run(["kubectl", "get", "pods", "-n", NAMESPACE], check=False)


if __name__ == "__main__":
    cli()
