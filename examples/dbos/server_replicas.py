#!/usr/bin/env python3
"""
Multi-Replica Demo
==================

Demonstrates durable workflow execution across multiple server replicas
backed by a shared Postgres database with DBOS.

  - Two WorkflowServer replicas share a Postgres-backed event store
  - A counter workflow is triggered on Replica A (port 8001)
  - Events are streamed in real-time from Replica B (port 8002)
  - Ctrl+C interrupts the workflow mid-flight
  - --resume picks up exactly where it left off via DBOS recovery

Usage:
    python examples/dbos/server_replicas.py              # Start new
    python examples/dbos/server_replicas.py --resume     # Resume after Ctrl+C
    python examples/dbos/server_replicas.py --clean      # Tear down everything
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from llama_agents.client import WorkflowClient

_DIR = Path(__file__).parent
_HANDLER_FILE = _DIR / ".last_handler_id"
_COMPOSE_FILE = _DIR / "docker-compose.yml"

# -- Pretty output -----------------------------------------------------------

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, color: str = DIM) -> None:
    print(f"  {color}{ts()}{RESET}  {msg}")


# -- Infrastructure -----------------------------------------------------------


def run_cmd(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, check=True, text=True, capture_output=True, **kwargs
        )
    except subprocess.CalledProcessError as e:
        parts = [f"Command failed: {' '.join(args)}"]
        if e.stdout:
            parts.append(f"stdout:\n{e.stdout}")
        if e.stderr:
            parts.append(f"stderr:\n{e.stderr}")
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd, e.stdout, e.stderr
        ) from RuntimeError("\n".join(parts))


def start_postgres() -> None:
    log("Starting Postgres container...", BLUE)
    run_cmd("docker", "compose", "-f", str(_COMPOSE_FILE), "up", "-d")
    for _ in range(30):
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
            log("Postgres ready", GREEN)
            return
        except subprocess.CalledProcessError:
            time.sleep(1)
    raise RuntimeError("Postgres failed to start")


def start_replica(port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(_DIR / "_replica.py"), "--port", str(port)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_server(port: int, timeout: float = 30.0) -> None:
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


# -- Workflow operations ------------------------------------------------------


async def start_workflow(client: WorkflowClient) -> str:
    handler = await client.run_workflow_nowait("counter")
    return handler.handler_id


async def stream_events(
    client: WorkflowClient, handler_id: str, resume: bool = False
) -> bool:
    after: int | str = "now" if resume else -1
    completed = False
    async for event in client.get_workflow_events(handler_id, after_sequence=after):
        event_type = event.type
        event_data = event.value or {}
        if event_type == "Tick":
            count = event_data.get("count", "?")
            bar = "#" * int(count) + "." * (20 - int(count))
            log(f"Tick {count:>2}/20  [{bar}]", CYAN)
        elif event_type == "CounterResult":
            final = event_data.get("final_count", "?")
            log(f"Done! final_count={final}", GREEN)
            completed = True
        else:
            log(f"{event_type}: {event_data}", DIM)
    return completed


# -- Main ---------------------------------------------------------------------


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Replica Demo")
    parser.add_argument("--resume", action="store_true", help="Resume last workflow")
    parser.add_argument("--clean", action="store_true", help="Tear down everything")
    args = parser.parse_args()

    if args.clean:
        if _HANDLER_FILE.exists():
            _HANDLER_FILE.unlink()
        subprocess.run(
            ["docker", "compose", "-f", str(_COMPOSE_FILE), "down", "-v"],
            check=False,
        )
        print("Cleaned up.")
        return

    print()
    print(f"  {BOLD}Multi-Replica Workflow Demo{RESET}")
    print(f"  {DIM}Two servers, one Postgres, durable execution{RESET}")
    print()

    replicas: list[subprocess.Popen[str]] = []

    def cleanup() -> None:
        for proc in replicas:
            proc.kill()
        for proc in replicas:
            proc.wait()

    def handle_sigint() -> None:
        print()
        log("Interrupted. Workflow state is safe in Postgres.", YELLOW)
        log(f"Run with {BOLD}--resume{RESET} to continue where you left off.", YELLOW)
        print()
        cleanup()
        os._exit(130)

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, handle_sigint)

    replica_a = WorkflowClient(base_url="http://localhost:8001")
    replica_b = WorkflowClient(base_url="http://localhost:8002")

    try:
        # --- Postgres ---
        start_postgres()
        print()

        # --- Replicas ---
        log(
            f"Starting Replica A on :8001  {DIM}(executor_id=replica-8001){RESET}", BLUE
        )
        replicas.append(start_replica(8001))
        log(
            f"Starting Replica B on :8002  {DIM}(executor_id=replica-8002){RESET}", BLUE
        )
        replicas.append(start_replica(8002))
        wait_for_server(8001)
        log("Replica A ready", GREEN)
        wait_for_server(8002)
        log("Replica B ready", GREEN)
        print()

        # --- Workflow ---
        if args.resume and _HANDLER_FILE.exists():
            handler_id = _HANDLER_FILE.read_text().strip()
            log(f"Resuming workflow  handler_id={BOLD}{handler_id}{RESET}", YELLOW)
            log(f"{DIM}DBOS recovers the workflow on the owning replica{RESET}", DIM)
        else:
            log("Triggering counter workflow on Replica A (:8001)...", BLUE)
            handler_id = await start_workflow(replica_a)
            _HANDLER_FILE.write_text(handler_id)
            log(f"Workflow started  handler_id={BOLD}{handler_id}{RESET}", GREEN)

        print()
        log("Streaming events from Replica B (:8002)...", BLUE)
        log(f"{DIM}Events flow: Replica A -> Postgres -> Replica B -> here{RESET}", DIM)
        print()

        completed = await stream_events(replica_b, handler_id, resume=args.resume)

        print()
        if completed:
            log(f"{GREEN}{BOLD}Workflow completed across replicas!{RESET}", GREEN)
            if _HANDLER_FILE.exists():
                _HANDLER_FILE.unlink()
        print()
    finally:
        cleanup()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
