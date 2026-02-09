# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Pytest command for running tests across packages."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
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


def extract_test_counts(stdout: str) -> str | None:
    """Extract test count summary from pytest output.

    Parses the final pytest summary line like "= 42 passed, 1 failed in 3.2s ="
    and returns a short string like "42 passed" or "42 passed, 1 failed".

    Args:
        stdout: The complete stdout from a pytest run.

    Returns:
        A short summary string, or None if no summary line found.
    """
    if not isinstance(stdout, str):
        return None
    for line in reversed(stdout.splitlines()):
        clean = _ANSI_ESCAPE_RE.sub("", line).strip()
        match = re.match(r"^=+\s+(.+?)\s+in\s+[\d.]+s\s+=+$", clean)
        if match:
            return match.group(1)
        # Also match lines without timing, e.g. "= 42 passed ="
        match = re.match(r"^=+\s+(.+?)\s+=+$", clean)
        if match:
            inner = match.group(1)
            # Avoid matching section headers like "FAILURES" or "test session starts"
            if re.search(r"\d+\s+(passed|failed|error|skipped|warning)", inner):
                return inner
    return None


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
    # Only include directories with both tests/ and pyproject.toml
    # (empty dirs may linger after package removal because git doesn't track empty dirs)
    packages_dir = repo_root / "packages"
    if packages_dir.exists():
        for item in packages_dir.iterdir():
            has_tests = item.is_dir() and (item / "tests").is_dir()
            has_pyproject = (item / "pyproject.toml").is_file()
            if has_tests and has_pyproject:
                packages.append(PackageInfo.from_path(item))

    return sorted(packages, key=lambda p: p.name)


# Active subprocesses tracked for cleanup on interrupt
_active_procs: list[subprocess.Popen[str]] = []
_active_procs_lock = threading.Lock()


def _run_tracked(
    cmd: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess while tracking it for interrupt cleanup."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    with _active_procs_lock:
        _active_procs.append(proc)
    try:
        stdout, stderr = proc.communicate()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        with _active_procs_lock:
            try:
                _active_procs.remove(proc)
            except ValueError:
                pass
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _kill_active_procs() -> None:
    """Kill all tracked subprocesses."""
    with _active_procs_lock:
        for proc in _active_procs:
            try:
                proc.kill()
            except OSError:
                pass
        _active_procs.clear()


def run_package_tests(
    pkg: PackageInfo, pytest_args: tuple[str, ...]
) -> dict[str, bool | str | float | None]:
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
    sync_result = _run_tracked(sync_cmd, env)
    if sync_result.returncode != 0:
        duration = time.time() - start_time
        return {
            "success": False,
            "no_tests": False,
            "stdout": sync_result.stdout,
            "stderr": f"uv sync failed:\n{sync_result.stderr}",
            "duration": duration,
        }

    cmd = ["uv", "run", "--directory", str(pkg.path), "pytest", *pytest_args]
    result = _run_tracked(cmd, env)
    duration = time.time() - start_time
    # Exit code 0 = success, exit code 5 = no tests collected (not a failure)
    no_tests = result.returncode == 5
    success = result.returncode == 0 or no_tests
    return {
        "success": success,
        "no_tests": no_tests,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration": duration,
        "test_summary": extract_test_counts(result.stdout),
    }


def _render_progress_table(
    packages: list[PackageInfo],
    results: dict[str, dict[str, bool | str | float | None]],
    start_times: dict[str, float],
    spinners: dict[str, Spinner],
) -> Table:
    """Render the progress display for rich Live."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("name", style="bold", no_wrap=True)
    table.add_column("time", justify="right", style="dim", no_wrap=True)

    for pkg in packages:
        if pkg.name in results:
            result_data = results[pkg.name]
            duration = result_data["duration"]
            test_summary = result_data.get("test_summary")
            table.add_row(pkg.name, f"{duration:.1f}s")
            if result_data.get("no_tests"):
                table.add_row(Text("  no tests collected", style="yellow"), "")
            elif test_summary:
                style = "green" if result_data["success"] else "red"
                table.add_row(Text(f"  {test_summary}", style=style), "")
            elif result_data["success"]:
                table.add_row(Text("  passed", style="green"), "")
            else:
                table.add_row(Text("  failed", style="red"), "")
        elif pkg.name in start_times:
            elapsed = time.time() - start_times[pkg.name]
            table.add_row(pkg.name, f"{elapsed:.1f}s")
            spinner = spinners.get(pkg.name, Spinner("dots", style="yellow"))
            table.add_row(spinner, "")
        else:
            table.add_row(Text(pkg.name, style="dim"), "")
    return table


def run_tests_with_rich_progress(
    packages: list[PackageInfo],
    pytest_args: tuple[str, ...],
    max_workers: int,
) -> dict[str, dict[str, bool | str | float | None]]:
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
    results: dict[str, dict[str, bool | str | float | None]] = {}
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


def _run_tests_verbose(
    target_packages: list[PackageInfo],
    pytest_args: tuple[str, ...],
    max_workers: int,
    total: int,
    verbose: bool,
) -> dict[str, dict[str, bool | str | float | None]]:
    """Run tests with simple sequential output (non-TTY or verbose mode)."""
    results: dict[str, dict[str, bool | str | float | None]] = {}
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
                test_summary = result_data.get("test_summary")
                summary_suffix = f" ({test_summary})" if test_summary else ""
                if result_data.get("no_tests"):
                    status = click.style("NO TESTS", fg="yellow")
                elif result_data["success"]:
                    status = click.style(f"PASSED{summary_suffix}", fg="green")
                else:
                    status = click.style(f"FAILED{summary_suffix}", fg="red")
                click.echo(
                    f"[{idx}/{total}] {pkg.name}... {status} "
                    f"({result_data['duration']:.1f}s)"
                )
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
    help="Filter packages by substring match. Can be used multiple times.",
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
        dev pytest                  # Run all tests
        dev pytest -p workflows     # Packages matching "workflows"
        dev pytest -p server client # Multiple filters
        dev pytest -- -k test_name  # Pass args to pytest
    """
    repo_root = Path(__file__).parents[3]
    all_packages = discover_test_packages(repo_root)

    if not all_packages:
        click.echo("No packages with tests/ directory found.")
        sys.exit(0)

    # Filter to specified packages if provided
    # Per filter: exact match takes precedence, otherwise substring match
    if packages:
        available_names = {p.name for p in all_packages}
        matched: set[str] = set()
        for filt in packages:
            if filt in available_names:
                # Exact match - only add this one
                matched.add(filt)
            else:
                # Substring match
                matched.update(p.name for p in all_packages if filt in p.name)
        target_packages = [p for p in all_packages if p.name in matched]
        if not target_packages:
            click.echo(f"No packages matched: {', '.join(packages)}", err=True)
            click.echo(f"Available: {', '.join(sorted(available_names))}")
            sys.exit(1)
    else:
        target_packages = all_packages

    # Run tests in each package
    total = len(target_packages)
    max_workers = min(parallel, len(target_packages))
    use_rich_progress = sys.stdout.isatty() and not verbose

    # Single package: always show full output directly
    if total == 1:
        verbose = True
        use_rich_progress = False

    # Install a SIGINT handler that exits immediately. The default Python handler
    # raises KeyboardInterrupt, but ThreadPoolExecutor's shutdown(wait=True) blocks
    # the main thread in thread joins, preventing the interrupt from being handled
    # until all subprocess children finish.
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum: int, frame: object) -> None:
        _kill_active_procs()
        click.echo("\nInterrupted.")
        # Restore default handler so a second Ctrl+C kills immediately
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        if use_rich_progress:
            results = run_tests_with_rich_progress(
                target_packages, pytest_args, max_workers
            )
        else:
            results = _run_tests_verbose(
                target_packages, pytest_args, max_workers, total, verbose
            )
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # Print summary
    passed = sum(1 for v in results.values() if v["success"] and not v.get("no_tests"))
    no_tests = sum(1 for v in results.values() if v.get("no_tests"))
    failed = len(results) - passed - no_tests

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

    # Count total individual tests across all packages
    total_tests = 0
    for result_data in results.values():
        ts = result_data.get("test_summary")
        if ts and isinstance(ts, str):
            total_tests += sum(
                int(n)
                for n in re.findall(r"(\d+)\s+(?:passed|failed|error|skipped)", ts)
            )

    if failed or no_tests:
        # Print summary table only when there are failures or missing tests
        click.echo(f"\n{'=' * 50}")
        click.echo("Test Summary")
        click.echo("=" * 50)

        max_name_len = max(len(name) for name in results)
        for name, result_data in results.items():
            test_summary = result_data.get("test_summary")
            summary_suffix = f" ({test_summary})" if test_summary else ""
            if result_data.get("no_tests"):
                status = click.style("NO TESTS", fg="yellow")
            elif result_data["success"]:
                status = click.style(f"PASSED{summary_suffix}", fg="green")
            else:
                status = click.style(f"FAILED{summary_suffix}", fg="red")
            click.echo(f"{name.ljust(max_name_len)}  {status}")

        click.echo("=" * 50)
        pkg_parts = []
        if failed:
            pkg_parts.append(click.style(f"{failed} failed", fg="red"))
        if passed:
            pkg_parts.append(click.style(f"{passed} passed", fg="green"))
        if no_tests:
            pkg_parts.append(click.style(f"{no_tests} no tests", fg="yellow"))
        pkg_label = "package" if (passed + failed + no_tests) == 1 else "packages"
        summary_line = f"{', '.join(pkg_parts)} {pkg_label}"
        if total_tests:
            summary_line += f", {total_tests} tests total"
        click.echo(summary_line)
    else:
        # All passed — just a short totals line
        pkg_label = "package" if passed == 1 else "packages"
        total_str = f", {total_tests} tests" if total_tests else ""
        click.echo(click.style(f"\n{passed} {pkg_label} passed{total_str}", fg="green"))

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
