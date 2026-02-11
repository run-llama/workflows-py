#!/usr/bin/env python3
"""
Multi-Replica Demo Orchestrator

Launches docker postgres, two server replicas, and demonstrates
cross-replica workflow execution with DBOS durability.

Usage:
    python examples/multi_replica/run.py              # Start new
    python examples/multi_replica/run.py --resume     # Resume last
    python examples/multi_replica/run.py --clean      # Reset state
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

_DIR = Path(__file__).parent
_HANDLER_FILE = _DIR / ".last_handler_id"
_COMPOSE_FILE = _DIR / "docker-compose.yml"


def run_cmd(*args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)  # type: ignore[arg-type]


def start_postgres() -> None:
    print("Starting postgres...")
    run_cmd("docker", "compose", "-f", str(_COMPOSE_FILE), "up", "-d")
    # Wait for postgres to be ready
    for i in range(30):
        try:
            run_cmd(
                "docker",
                "compose",
                "-f",
                str(_COMPOSE_FILE),
                "exec",
                "-T",
                "postgres",
                "pg_isready",
                "-U",
                "workflows",
            )
            print("Postgres ready.")
            return
        except subprocess.CalledProcessError:
            time.sleep(1)
    raise RuntimeError("Postgres failed to start")


def start_replica(port: int) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, str(_DIR / "serve.py"), "--port", str(port)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(port: int, timeout: float = 30.0) -> None:
    """Wait until the server responds on the given port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"http://localhost:{port}/workflows", timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server on port {port} did not start in {timeout}s")


def start_workflow(port: int) -> str:
    """POST to start the counter workflow, return handler_id."""
    resp = httpx.post(
        f"http://localhost:{port}/workflows/counter/run",
        json={},
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    handler_id = data["handler_id"]
    print(f"Started workflow, handler_id={handler_id}")
    return handler_id


def stream_events(port: int, handler_id: str) -> bool:
    """Stream SSE events from a replica. Returns True if workflow completed."""
    url = f"http://localhost:{port}/workflows/counter/results/{handler_id}/stream"
    print(f"Streaming events from port {port}...")
    completed = False
    with httpx.stream("GET", url, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[len("data: ") :])
            event_type = payload.get("event_type", "")
            event_data = payload.get("event_data", {})
            if event_type == "Tick":
                count = event_data.get("count", "?")
                print(f"  [Stream] Tick {count}")
            elif event_type == "CounterResult":
                final = event_data.get("final_count", "?")
                print(f"  [Stream] Done! final_count={final}")
                completed = True
            else:
                print(f"  [Stream] {event_type}: {event_data}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Replica Demo")
    parser.add_argument("--resume", action="store_true", help="Resume last workflow")
    parser.add_argument(
        "--clean", action="store_true", help="Remove state and docker volumes"
    )
    args = parser.parse_args()

    if args.clean:
        if _HANDLER_FILE.exists():
            _HANDLER_FILE.unlink()
            print(f"Removed {_HANDLER_FILE}")
        subprocess.run(
            ["docker", "compose", "-f", str(_COMPOSE_FILE), "down", "-v"],
            check=False,
        )
        print("Cleaned up.")
        return

    replicas: list[subprocess.Popen[str]] = []

    def cleanup() -> None:
        for proc in replicas:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted.")
        cleanup()
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        start_postgres()

        print("Starting replicas...")
        replicas.append(start_replica(8001))
        replicas.append(start_replica(8002))

        wait_for_server(8001)
        print("Replica A (8001) ready.")
        wait_for_server(8002)
        print("Replica B (8002) ready.")

        if args.resume and _HANDLER_FILE.exists():
            handler_id = _HANDLER_FILE.read_text().strip()
            print(f"Resuming handler_id={handler_id}")
        else:
            # Start workflow on replica A
            handler_id = start_workflow(8001)
            _HANDLER_FILE.write_text(handler_id)

        # Stream from replica B
        completed = stream_events(8002, handler_id)
        if completed:
            print("\nWorkflow completed successfully across replicas!")
            if _HANDLER_FILE.exists():
                _HANDLER_FILE.unlink()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
