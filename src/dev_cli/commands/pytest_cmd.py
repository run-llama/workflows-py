# SPDX-FileCopyrightText: 2026 LlamaIndex Authors
# SPDX-License-Identifier: MIT
"""Pytest command for running tests across packages."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class PackageInfo:
    """A package with tests to run."""

    path: Path
    name: str

    @classmethod
    def from_path(cls, path: Path) -> PackageInfo:
        """Create a PackageInfo from a directory path, reading name from pyproject.toml."""
        pyproject_path = path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
                name = data.get("project", {}).get("name", path.name)
            except Exception:
                name = path.name
        else:
            name = path.name
        return cls(path=path, name=name)


# Regex to strip ANSI escape codes from text
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def extract_failures_section(stdout: str) -> str | None:
    """Extract the FAILURES section from pytest output.

    Parses pytest stdout to find content between "=== FAILURES ===" and the next
    section header. Returns just the failure stack traces without warnings or
    other pytest noise.

    Args:
        stdout: The complete stdout from a pytest run.

    Returns:
        The extracted failures section as a string, or None if no failures found.
    """
    lines = stdout.splitlines()
    in_failures = False
    failures_lines: list[str] = []

    for line in lines:
        if "FAILURES" in line and line.strip().startswith("="):
            in_failures = True
            failures_lines.append(line)  # Include the header
            continue
        if in_failures:
            # Stop at next section header (warnings, short test summary, etc.)
            if line.strip().startswith("=") and "=" * 10 in line:
                break
            failures_lines.append(line)

    return "\n".join(failures_lines).strip() if failures_lines else None


def extract_failed_test_names(stdout: str) -> list[tuple[str, str | None]]:
    """Extract failed test names and reasons from pytest output.

    Looks for the "short test summary info" section and extracts FAILED lines,
    or falls back to parsing the FAILURES section headers.

    Args:
        stdout: The complete stdout from a pytest run.

    Returns:
        List of tuples (test_name, reason) where reason may be None.
    """
    failed_tests: list[tuple[str, str | None]] = []
    lines = stdout.splitlines()

    # Try to find short test summary section first
    in_summary = False
    for line in lines:
        if "short test summary info" in line:
            in_summary = True
            continue
        if in_summary:
            # Strip ANSI color codes for pattern matching
            clean_line = _ANSI_ESCAPE_RE.sub("", line).strip()
            if clean_line.startswith("=") and "=" * 10 in clean_line:
                break
            if clean_line.startswith("FAILED"):
                # Extract test name and reason: "FAILED tests/foo.py::test_bar - reason"
                parts = clean_line.split(" ", 2)
                if len(parts) >= 2:
                    test_and_reason = parts[1]
                    if " - " in test_and_reason:
                        test_name, reason = test_and_reason.split(" - ", 1)
                    elif len(parts) >= 3 and parts[2].startswith("- "):
                        test_name = parts[1]
                        reason = parts[2][2:]  # Remove "- " prefix
                    else:
                        test_name = test_and_reason
                        reason = None
                    failed_tests.append((test_name, reason))

    # Fall back to parsing FAILURES section headers if no summary found
    if not failed_tests:
        for line in lines:
            # Look for test name headers like "_____ test_name _____"
            stripped = line.strip()
            if (
                stripped.startswith("_")
                and stripped.endswith("_")
                and len(stripped) > 20
            ):
                # Extract the test name from between the underscores
                test_name = stripped.strip("_").strip()
                # Verify it contains actual test name characters, not just
                # spaces/underscores (which would match separator lines like
                # "_ _ _ _ _ _ _ _ _")
                if test_name and any(c.isalnum() for c in test_name):
                    failed_tests.append((test_name, None))

    return failed_tests


def discover_test_packages(repo_root: Path) -> list[PackageInfo]:
    """Discover packages that contain a tests/ directory.

    Returns a sorted list of PackageInfo objects, including the repo root
    if it has tests. Package names are read from pyproject.toml.
    """
    packages: list[PackageInfo] = []

    # Check repo root for tests (e.g., tests/dev_cli/)
    root_tests = repo_root / "tests"
    if root_tests.is_dir() and any(root_tests.iterdir()):
        packages.append(PackageInfo.from_path(repo_root))

    # Check packages/ directory
    packages_dir = repo_root / "packages"
    if packages_dir.exists():
        for item in packages_dir.iterdir():
            if item.is_dir() and (item / "tests").is_dir():
                packages.append(PackageInfo.from_path(item))

    return sorted(packages, key=lambda p: p.name)


def run_package_tests(
    pkg: PackageInfo, pytest_args: tuple[str, ...]
) -> dict[str, bool | str | float]:
    """Run pytest for a single package and return the results.

    Args:
        pkg: The PackageInfo to run tests for.
        pytest_args: Additional arguments to pass to pytest.

    Returns:
        Dictionary with success status, stdout, stderr, and duration.
    """
    # Pass through terminal settings
    env = os.environ.copy()
    env["COLUMNS"] = str(shutil.get_terminal_size().columns)
    # Enable colors if we're in an interactive terminal
    if sys.stdout.isatty():
        env["FORCE_COLOR"] = "1"

    start_time = time.time()

    # Ensure dependencies are installed before running tests (--inexact to only add, not remove)
    sync_cmd = ["uv", "sync", "--directory", str(pkg.path), "--inexact"]
    sync_result = subprocess.run(sync_cmd, capture_output=True, text=True, env=env)
    if sync_result.returncode != 0:
        duration = time.time() - start_time
        return {
            "success": False,
            "stdout": sync_result.stdout,
            "stderr": f"uv sync failed:\n{sync_result.stderr}",
            "duration": duration,
        }

    cmd = ["uv", "run", "--directory", str(pkg.path), "pytest", *pytest_args]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    duration = time.time() - start_time
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration": duration,
    }


def _render_progress_table(
    packages: list[PackageInfo],
    results: dict[str, dict[str, bool | str | float]],
    start_times: dict[str, float],
    spinners: dict[str, Spinner],
) -> Table:
    """Render the progress table for rich Live display."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("name", style="bold")
    table.add_column("status", width=12)
    table.add_column("time", justify="right", width=10)

    for pkg in packages:
        if pkg.name in results:
            success = results[pkg.name]["success"]
            duration = results[pkg.name]["duration"]
            status: Text | Spinner = (
                Text("PASSED", style="green")
                if success
                else Text("FAILED", style="red")
            )
            time_str = f"({duration:.1f}s)"
        elif pkg.name in start_times:
            elapsed = time.time() - start_times[pkg.name]
            status = spinners.get(pkg.name, Spinner("dots", style="yellow"))
            time_str = f"({elapsed:.1f}s)"
        else:
            status = Text("pending", style="dim")
            time_str = ""
        table.add_row(pkg.name, status, time_str)
    return table


def run_tests_with_rich_progress(
    packages: list[PackageInfo],
    pytest_args: tuple[str, ...],
    max_workers: int,
) -> dict[str, dict[str, bool | str | float]]:
    """Run tests with a live rich progress display.

    Shows a live-updating table with package status, spinner for running
    packages, and elapsed time.

    Args:
        packages: List of PackageInfo objects to test.
        pytest_args: Additional arguments to pass to pytest.
        max_workers: Maximum number of parallel workers.

    Returns:
        Dictionary mapping package names to their test results.
    """
    console = Console()
    results: dict[str, dict[str, bool | str | float]] = {}
    start_times: dict[str, float] = {}
    spinners: dict[str, Spinner] = {}

    def render() -> Table:
        return _render_progress_table(packages, results, start_times, spinners)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all and track start times
        futures = {}
        for pkg in packages:
            start_times[pkg.name] = time.time()
            spinners[pkg.name] = Spinner("dots", style="yellow")
            future = executor.submit(run_package_tests, pkg, pytest_args)
            futures[future] = pkg

        pending = set(futures.keys())

        with Live(render(), console=console, refresh_per_second=10) as live:
            while pending:
                # Wait with short timeout to allow display updates
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)

                for future in done:
                    pkg = futures[future]
                    result_data = future.result()
                    results[pkg.name] = result_data
                    # Remove spinner for completed package
                    spinners.pop(pkg.name, None)

                live.update(render())

    return results


@click.command(
    "pytest",
    context_settings={
        "ignore_unknown_options": True,
        "allow_interspersed_args": False,
    },
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show full pytest output for each package instead of compact progress.",
)
@click.option(
    "--package",
    "-p",
    "packages",
    multiple=True,
    help="Run tests only for specified package(s). Can be used multiple times.",
)
@click.option(
    "--parallel",
    "-j",
    default=10,
    type=int,
    help="Number of packages to test in parallel. Default 10, use 1 for sequential.",
)
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def pytest_cmd(
    verbose: bool,
    packages: tuple[str, ...],
    parallel: int,
    pytest_args: tuple[str, ...],
) -> None:
    """Run pytest across all packages in the repository.

    Any additional arguments after -- are passed through to pytest.

    Examples:
        dev pytest                      # Run all tests (10 parallel)
        dev pytest -v                   # Verbose: show all output
        dev pytest -j 1                 # Run sequentially
        dev pytest -- -v --tb=short     # Pass -v to pytest itself
        dev pytest -p llama-index-workflows  # Specific package
    """
    repo_root = Path(__file__).parents[3]
    all_packages = discover_test_packages(repo_root)

    if not all_packages:
        click.echo("No packages with tests/ directory found.")
        sys.exit(0)

    # Filter to specified packages if provided
    if packages:
        available_names = {p.name for p in all_packages}
        requested = set(packages)
        unknown = requested - available_names
        if unknown:
            click.echo(f"Unknown package(s): {', '.join(sorted(unknown))}", err=True)
            click.echo(f"Available packages: {', '.join(sorted(available_names))}")
            sys.exit(1)
        target_packages = [p for p in all_packages if p.name in requested]
    else:
        target_packages = all_packages

    # Run tests in each package
    total = len(target_packages)
    max_workers = min(parallel, len(target_packages))
    use_rich_progress = sys.stdout.isatty() and not verbose

    if use_rich_progress:
        # Use rich live progress display for interactive terminals
        results = run_tests_with_rich_progress(
            target_packages, pytest_args, max_workers
        )
    else:
        # Fall back to simple output for non-TTY or verbose mode
        results: dict[str, dict[str, bool | str | float]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_package_tests, pkg, pytest_args): pkg
                for pkg in target_packages
            }
            for idx, future in enumerate(as_completed(futures), 1):
                pkg = futures[future]
                result_data = future.result()
                results[pkg.name] = result_data

                if verbose:
                    click.echo(f"\n{'=' * 60}")
                    click.echo(f"Completed tests in {pkg.name}")
                    click.echo("=" * 60)
                    if result_data["stdout"]:
                        click.echo(result_data["stdout"], nl=False)
                    if result_data["stderr"]:
                        click.echo(result_data["stderr"], nl=False, err=True)
                else:
                    # Compact progress line (for non-TTY output)
                    status = (
                        click.style("PASSED", fg="green")
                        if result_data["success"]
                        else click.style("FAILED", fg="red")
                    )
                    click.echo(
                        f"[{idx}/{total}] {pkg.name}... {status} "
                        f"({result_data['duration']:.1f}s)"
                    )

    # Print summary
    passed = sum(1 for v in results.values() if v["success"])
    failed = len(results) - passed

    # Print failures section first (after progress), before summary
    if failed:
        click.echo(f"\n{'=' * 20} FAILURES {'=' * 20}")
        for name, result_data in results.items():
            if not result_data["success"]:
                click.echo(f"\n[{name}]")
                # Extract just the failures section, fall back to full output
                stdout = str(result_data["stdout"]) if result_data["stdout"] else ""
                failures = extract_failures_section(stdout)
                if failures:
                    click.echo(failures)
                elif stdout:
                    # Fall back to full output if no FAILURES section found
                    click.echo(stdout, nl=False)
                if result_data["stderr"]:
                    click.echo(result_data["stderr"], nl=False, err=True)

    # Print summary table at the end
    click.echo(f"\n{'=' * 50}")
    click.echo("Test Summary")
    click.echo("=" * 50)

    max_name_len = max(len(name) for name in results)
    for name, result_data in results.items():
        status = (
            click.style("PASSED", fg="green")
            if result_data["success"]
            else click.style("FAILED", fg="red")
        )
        click.echo(f"{name.ljust(max_name_len)}  {status}")

    click.echo("=" * 50)
    summary_parts = []
    if failed:
        summary_parts.append(click.style(f"{failed} failed", fg="red"))
    if passed:
        summary_parts.append(click.style(f"{passed} passed", fg="green"))
    click.echo(", ".join(summary_parts))

    # Show failed test names at the very end for quick reference
    if failed:
        click.echo("\nFailed tests:")
        for name, result_data in results.items():
            if not result_data["success"]:
                stdout = str(result_data["stdout"]) if result_data["stdout"] else ""
                test_info = extract_failed_test_names(stdout)
                for test_name, reason in test_info:
                    reason_str = f" - {reason}" if reason else ""
                    click.echo(
                        f"  {click.style('FAILED', fg='red')} [{name}] "
                        f"{test_name}{reason_str}"
                    )
        sys.exit(1)
