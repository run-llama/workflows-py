#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["click"]
# ///
"""
Local development environment for cloud_llama_deploy.

  up     — create kind cluster (if needed) and start tilt
  down   — tear down tilt resources, retaining data
  down --delete — also delete the kind cluster
"""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import click

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CLUSTER_NAME = "kind"
NAMESPACE = "llama-agents"


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


def cluster_exists() -> bool:
    result = run(["kind", "get", "clusters"], check=False, capture=True)
    return CLUSTER_NAME in result.stdout


def ensure_cluster() -> None:
    if cluster_exists():
        result = run(
            ["kind", "export", "kubeconfig", "--name", CLUSTER_NAME],
            check=False,
            capture=True,
        )
        if result.returncode == 0:
            return
        print(f"Cluster '{CLUSTER_NAME}' exists but is broken, recreating...")
        run(["kind", "delete", "cluster", "--name", CLUSTER_NAME])

    print(f"Creating kind cluster '{CLUSTER_NAME}'...")

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
                CLUSTER_NAME,
                "--config",
                config_path,
            ]
        )
    finally:
        os.unlink(config_path)

    install_ingress_controller()


def install_ingress_controller() -> None:
    result = run(
        ["kubectl", "get", "namespace", "ingress-nginx"], check=False, capture=True
    )
    if result.returncode == 0:
        return

    print("Installing nginx ingress controller...")
    run(
        [
            "kubectl",
            "apply",
            "-f",
            "https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml",
        ]
    )

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
def cli() -> None:
    """Local development environment for cloud_llama_deploy."""


@cli.command()
def up() -> None:
    """Create kind cluster (if needed) and start tilt."""
    version_cmds = {
        "kind": ["kind", "--version"],
        "kubectl": ["kubectl", "version", "--client"],
        "docker": ["docker", "--version"],
        "tilt": ["tilt", "version"],
    }
    for tool, cmd in version_cmds.items():
        if run(cmd, check=False, capture=True).returncode != 0:
            print(f"Missing required tool: {tool}", file=sys.stderr)
            sys.exit(1)

    ensure_cluster()

    # Ensure namespace exists
    result = run(["kubectl", "get", "namespace", NAMESPACE], check=False, capture=True)
    if result.returncode != 0:
        run(["kubectl", "create", "namespace", NAMESPACE])

    if not (PROJECT_ROOT / ".env").exists():
        print(
            "Note: no .env file found. GitHub integration requires GITHUB_APP_PRIVATE_KEY, GITHUB_APP_CLIENT_ID, GITHUB_APP_NAME, GITHUB_APP_SECRET."
        )

    print("Starting tilt...")
    print("  API:     http://localhost:8011")
    print("  Tilt UI: http://localhost:10350")
    print("  Ingress: *.127.0.0.1.nip.io:8090")
    os.execvp("tilt", ["tilt", "up", "-f", str(PROJECT_ROOT / "operator" / "Tiltfile")])


@cli.command()
@click.option("--delete", is_flag=True, help="Also delete the kind cluster")
def down(delete: bool) -> None:
    """Tear down tilt resources. Use --delete to also remove the cluster."""
    run(
        ["tilt", "down", "-f", str(PROJECT_ROOT / "operator" / "Tiltfile")], check=False
    )

    if delete:
        if cluster_exists():
            print(f"Deleting kind cluster '{CLUSTER_NAME}'...")
            run(["kind", "delete", "cluster", "--name", CLUSTER_NAME])


@cli.command()
def status() -> None:
    """Show cluster and deployment status."""
    if not cluster_exists():
        print(f"No kind cluster '{CLUSTER_NAME}' found")
        return

    run(["kind", "export", "kubeconfig", "--name", CLUSTER_NAME])
    run(["kubectl", "get", "pods", "-n", NAMESPACE], check=False)


if __name__ == "__main__":
    cli()
