#!/usr/bin/env python3
"""
DBOS Durability Manual Test CLI

Run durability test scenarios to verify DBOS recovery behavior.

Usage:
    python run.py multi-step           # Start new multi-step workflow
    python run.py multi-step --resume  # Resume last multi-step workflow
    python run.py --list               # List available scenarios
    python run.py --clean              # Remove database and state files
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from dbos import DBOS, DBOSConfig

from .shared import BASE_DIR, DB_FILE, LAST_RUN_FILE, SCENARIOS


def get_dbos_config() -> DBOSConfig:
    """Get DBOS configuration with persistent SQLite database."""
    system_db_url = f"sqlite+pysqlite:///{DB_FILE}?check_same_thread=false"
    return {
        "name": "dbos-durability-tests",
        "system_database_url": system_db_url,
        "run_admin_server": False,
    }


def load_last_run_ids() -> dict[str, str]:
    """Load last run IDs from state file."""
    if LAST_RUN_FILE.exists():
        try:
            with open(LAST_RUN_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_last_run_id(scenario: str, run_id: str) -> None:
    """Save run ID for a scenario to state file."""
    state = load_last_run_ids()
    state[scenario] = run_id
    with open(LAST_RUN_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_last_run_id(scenario: str) -> str | None:
    """Get last run ID for a scenario."""
    return load_last_run_ids().get(scenario)


def clean_state() -> None:
    """Remove database and state files."""
    files_removed = []
    for path in [DB_FILE, LAST_RUN_FILE]:
        if path.exists():
            path.unlink()
            files_removed.append(str(path))

    # Also clean up api_calls.log if present
    api_log = BASE_DIR / "api_calls.log"
    if api_log.exists():
        api_log.unlink()
        files_removed.append(str(api_log))

    if files_removed:
        print(f"Removed: {', '.join(files_removed)}")
    else:
        print("Nothing to clean.")


def list_scenarios() -> None:
    """Print available scenarios."""
    print("Available scenarios:")
    for name in sorted(SCENARIOS.keys()):
        print(f"  - {name}")


async def run_scenario(scenario: str, resume: bool, run_id: str | None) -> None:
    """Run a scenario with the given parameters."""
    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario}")
        print("Use --list to see available scenarios.")
        sys.exit(1)

    # Determine run_id
    active_run_id: str
    if run_id:
        active_run_id = run_id
    elif resume:
        last_run = get_last_run_id(scenario)
        if not last_run:
            print(
                f"No previous run found for '{scenario}'. Run without --resume first."
            )
            sys.exit(1)
        active_run_id = last_run
        print(f"Resuming run: {active_run_id}")
    else:
        active_run_id = f"{scenario}-{uuid.uuid4().hex[:8]}"
        print(f"Starting new run: {active_run_id}")

    # Save run_id
    save_last_run_id(scenario, active_run_id)

    # Initialize DBOS
    DBOS(config=get_dbos_config())

    try:
        # Run the scenario
        await SCENARIOS[scenario](active_run_id, resume)
    finally:
        DBOS.destroy()


# Import scenarios to register them
def _import_scenarios() -> None:
    """Import scenario modules to register them."""
    # Import scenarios here so they can use the @register_scenario decorator
    from . import scenarios  # noqa: F401

    # Import each scenario module
    from .scenarios import counter, hitl, multi_step  # noqa: F401


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DBOS Durability Manual Test CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        help="Scenario to run (use --list to see available)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last workflow for this scenario",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Use explicit workflow ID instead of generating one",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_scenarios",
        help="List available scenarios",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove database and state files",
    )

    args = parser.parse_args()

    # Import scenarios first
    _import_scenarios()

    if args.clean:
        clean_state()
        return

    if args.list_scenarios:
        list_scenarios()
        return

    if not args.scenario:
        parser.print_help()
        return

    asyncio.run(run_scenario(args.scenario, args.resume, args.run_id))


if __name__ == "__main__":
    main()
