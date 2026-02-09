# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for the pytest CLI command."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from conftest import is_sync_call, sync_success

from dev_cli.cli import cli
from dev_cli.commands.pytest_cmd import (
    PackageInfo,
    _render_progress_table,
    discover_test_packages,
    extract_failed_test_names,
    extract_failures_section,
    extract_test_counts,
    run_tests_with_rich_progress,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def packages_dir(tmp_path: Path) -> Path:
    """Create a packages directory."""
    packages = tmp_path / "packages"
    packages.mkdir()
    return packages


@pytest.fixture
def create_pkg(packages_dir: Path) -> Callable[[str], PackageInfo]:
    """Factory to create a package with tests subdirectory and pyproject.toml."""

    def _create(name: str) -> PackageInfo:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()
        (pkg / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
        return PackageInfo(path=pkg, name=name)

    return _create


# --- discover_test_packages tests ---


def test_discover_finds_packages_with_tests(packages_dir: Path) -> None:
    # Create package with tests and pyproject.toml
    pkg = packages_dir / "pkg-with-tests"
    pkg.mkdir()
    (pkg / "tests").mkdir()
    (pkg / "pyproject.toml").write_text('[project]\nname = "pkg-with-tests"\n')
    # Package without tests (but with pyproject.toml)
    no_tests = packages_dir / "pkg-without-tests"
    no_tests.mkdir()
    (no_tests / "pyproject.toml").write_text('[project]\nname = "pkg-without-tests"\n')
    # Package with tests but no pyproject.toml (should be ignored)
    orphan = packages_dir / "orphan-pkg"
    orphan.mkdir()
    (orphan / "tests").mkdir()

    result = discover_test_packages(packages_dir.parent)
    assert len(result) == 1
    assert result[0].name == "pkg-with-tests"


def test_discover_returns_sorted_list(packages_dir: Path) -> None:
    for name in ["zebra-pkg", "alpha-pkg", "middle-pkg"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()
        (pkg / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')

    result = discover_test_packages(packages_dir.parent)
    assert [p.name for p in result] == ["alpha-pkg", "middle-pkg", "zebra-pkg"]


def test_discover_empty_when_no_packages_dir(tmp_path: Path) -> None:
    result = discover_test_packages(tmp_path)
    assert result == []


# --- Basic CLI tests ---


def test_pytest_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["pytest", "--help"])
    assert result.exit_code == 0
    assert "--package" in result.output
    assert "-p" in result.output


def test_pytest_filters_by_package(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    pkg_a = create_pkg("pkg-a")
    pkg_b = create_pkg("pkg-b")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[pkg_a, pkg_b],
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            runner.invoke(cli, ["pytest", "-p", "pkg-a"])

            # 2 calls: sync + pytest for pkg-a only
            assert mock_run.call_count == 2
            call_args = mock_run.call_args_list[1][0][0]
            assert "pkg-a" in str(call_args)


def test_pytest_errors_on_unknown_package(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    real_pkg = create_pkg("real-pkg")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[real_pkg],
    ):
        result = runner.invoke(cli, ["pytest", "-p", "unknown-pkg"])

        assert result.exit_code == 1
        assert "No packages matched: unknown-pkg" in result.output
        assert "Available: real-pkg" in result.output


@pytest.mark.parametrize(
    "pkg_names,filters,expected_matches",
    [
        (["pkg-foo", "pkg-foo-client", "pkg-bar"], ["foo"], 2),  # substring
        (["foo", "foo-client"], ["foo"], 1),  # exact takes precedence
        (["foo", "foo-client", "bar"], ["foo", "bar"], 2),  # mixed
    ],
    ids=["substring", "exact", "mixed"],
)
def test_pytest_package_filter_matching(
    runner: CliRunner,
    create_pkg: Callable[[str], PackageInfo],
    pkg_names: list[str],
    filters: list[str],
    expected_matches: int,
) -> None:
    pkgs = [create_pkg(n) for n in pkg_names]
    filter_args = [arg for f in filters for arg in ["-p", f]]

    with patch("dev_cli.commands.pytest_cmd.discover_test_packages", return_value=pkgs):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            runner.invoke(cli, ["pytest", *filter_args])
            assert mock_run.call_count == expected_matches * 2  # sync + pytest per pkg


def test_pytest_passes_args_through(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    my_pkg = create_pkg("my-pkg")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[my_pkg],
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            runner.invoke(cli, ["pytest", "--", "-v", "--tb=short", "-k", "test_foo"])

            call_args = mock_run.call_args[0][0]
            assert "-v" in call_args
            assert "--tb=short" in call_args
            assert "-k" in call_args
            assert "test_foo" in call_args


def test_pytest_continues_on_failure(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b", "pkg-c"]]

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = [
                sync_success(),
                Mock(returncode=1, stdout="", stderr=""),
                sync_success(),
                Mock(returncode=0, stdout="", stderr=""),
                sync_success(),
                Mock(returncode=0, stdout="", stderr=""),
            ]
            result = runner.invoke(cli, ["pytest"])

            assert mock_run.call_count == 6
            assert result.exit_code == 1
            assert "FAILED" in result.output
            assert "PASSED" in result.output


def test_pytest_shows_summary(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b"]]

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = runner.invoke(cli, ["pytest"])

            assert "Test Summary" in result.output
            assert "pkg-a" in result.output
            assert "pkg-b" in result.output
            assert "2 passed" in result.output


# --- Quiet/verbose mode tests ---


def test_pytest_quiet_mode_hides_streaming_output(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Quiet mode (default) shows compact progress, not full pytest output."""
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b"]]

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="collected 5 items\ntest_foo.py::test_one PASSED\n",
                stderr="",
            )
            result = runner.invoke(cli, ["pytest"])

            assert "[1/2]" in result.output
            assert "[2/2]" in result.output
            # Pytest's "collected X items" should NOT appear
            assert "collected" not in result.output


def test_pytest_verbose_shows_streaming_output(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Verbose mode shows full pytest output AND aggregate summary."""
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b"]]

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="collected 5 items\ntest_foo.py::test_one PASSED\n",
                stderr="",
            )
            result = runner.invoke(cli, ["pytest", "--verbose"])

            assert "collected 5 items" in result.output
            assert "test_foo.py::test_one PASSED" in result.output
            assert "Test Summary" in result.output
            assert "2 passed" in result.output


def test_pytest_failure_recap_always_shown(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Failure recap is shown in both quiet and verbose modes."""
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b"]]

    failure_output = (
        "collected 3 items\n"
        "test_foo.py::test_failing FAILED\n"
        "FAILURES\n"
        "test_failing - AssertionError: expected True\n"
    )

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        for verbose_flag in [[], ["--verbose"]]:
            with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
                mock_run.side_effect = [
                    sync_success(),
                    Mock(returncode=0, stdout="all tests passed\n", stderr=""),
                    sync_success(),
                    Mock(returncode=1, stdout=failure_output, stderr=""),
                ]
                result = runner.invoke(cli, ["pytest", *verbose_flag])

                assert "FAILURES" in result.output or "FAILED" in result.output
                assert "pkg-b" in result.output
                assert "AssertionError" in result.output


# --- Parallel execution tests ---


def test_pytest_runs_parallel_by_default(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Tests run in parallel by default - verified by concurrent execution."""
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b", "pkg-c"]]

    started_count = 0
    started_lock = threading.Lock()
    all_started = threading.Event()
    call_order: list[str] = []

    def mock_subprocess_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal started_count

        if is_sync_call(cmd):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        pkg_name = "unknown"
        for i, arg in enumerate(cmd):
            if arg == "--directory" and i + 1 < len(cmd):
                pkg_name = Path(cmd[i + 1]).name
                break

        with started_lock:
            started_count += 1
            call_order.append(f"start:{pkg_name}")
            if started_count >= 3:
                all_started.set()

        all_started.wait(timeout=2.0)

        with started_lock:
            call_order.append(f"end:{pkg_name}")

        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch(
            "dev_cli.commands.pytest_cmd.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = runner.invoke(cli, ["pytest"])

            # Verify parallel: at least 2 starts before any end
            start_indices = [
                i for i, x in enumerate(call_order) if x.startswith("start:")
            ]
            end_indices = [i for i, x in enumerate(call_order) if x.startswith("end:")]
            assert len(start_indices) >= 2
            assert max(start_indices[:2]) < min(end_indices)
            assert result.exit_code == 0


def test_pytest_parallel_one_runs_sequentially(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """--parallel 1 runs packages one at a time."""
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b", "pkg-c"]]

    call_order: list[str] = []
    lock = threading.Lock()
    active_count = 0
    max_concurrent = 0

    def mock_subprocess_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal active_count, max_concurrent

        if is_sync_call(cmd):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        pkg_name = "unknown"
        for i, arg in enumerate(cmd):
            if arg == "--directory" and i + 1 < len(cmd):
                pkg_name = Path(cmd[i + 1]).name
                break

        with lock:
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
            call_order.append(f"start:{pkg_name}")

        time.sleep(0.01)

        with lock:
            call_order.append(f"end:{pkg_name}")
            active_count -= 1

        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch(
            "dev_cli.commands.pytest_cmd.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = runner.invoke(cli, ["pytest", "--parallel", "1"])

            assert max_concurrent == 1
            # Verify strict ordering: start, end, start, end...
            for i in range(0, len(call_order) - 1, 2):
                assert call_order[i].startswith("start:")
                assert call_order[i + 1].startswith("end:")
            assert result.exit_code == 0


def test_pytest_parallel_handles_mixed_results(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Parallel execution handles mixed passed/failed results correctly."""
    pkgs = [create_pkg(name) for name in ["pkg-fail", "pkg-pass-1", "pkg-pass-2"]]

    completed: list[str] = []
    lock = threading.Lock()

    def mock_subprocess_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if is_sync_call(cmd):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        pkg_name = "unknown"
        for i, arg in enumerate(cmd):
            if arg == "--directory" and i + 1 < len(cmd):
                pkg_name = Path(cmd[i + 1]).name
                break

        with lock:
            completed.append(pkg_name)

        if pkg_name == "pkg-fail":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="FAILED test_example.py::test_bad\nAssertionError: bad\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch(
            "dev_cli.commands.pytest_cmd.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = runner.invoke(cli, ["pytest"])

            assert set(completed) == {"pkg-fail", "pkg-pass-1", "pkg-pass-2"}
            assert result.exit_code == 1
            assert "2 passed" in result.output
            assert "1 failed" in result.output


# --- extract_failures_section tests ---


def test_extract_failures_returns_failures_block() -> None:
    pytest_output = """
============================= test session starts ==============================
collected 5 items

test_example.py::test_one PASSED
test_example.py::test_two FAILED

=================================== FAILURES ===================================
_________________________ test_two _________________________

    def test_two():
>       assert False
E       AssertionError

test_example.py:10: AssertionError
=========================== short test summary info ============================
FAILED test_example.py::test_two - AssertionError
============================== 1 failed, 1 passed ==============================
"""
    result = extract_failures_section(pytest_output)
    assert result is not None
    assert "FAILURES" in result
    assert "test_two" in result
    assert "short test summary" not in result


def test_extract_failures_returns_none_for_passing() -> None:
    pytest_output = """
============================= test session starts ==============================
collected 2 items

test_example.py::test_one PASSED

============================== 2 passed ==============================
"""
    result = extract_failures_section(pytest_output)
    assert result is None


def test_extract_failures_excludes_warnings() -> None:
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_failing _________________________

AssertionError
=============================== warnings summary ===============================
test_example.py:3: DeprecationWarning: deprecated
=========================== short test summary info ============================
"""
    result = extract_failures_section(pytest_output)
    assert result is not None
    assert "FAILURES" in result
    assert "DeprecationWarning" not in result


def test_extract_failures_handles_multiple() -> None:
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_one _________________________

AssertionError: first failure
_________________________ test_two _________________________

AssertionError: second failure
=========================== short test summary info ============================
"""
    result = extract_failures_section(pytest_output)
    assert result is not None
    assert "first failure" in result
    assert "second failure" in result


# --- extract_failed_test_names tests ---


def test_extract_failed_names_from_short_summary() -> None:
    pytest_output = """
=========================== short test summary info ============================
FAILED tests/test_example.py::test_one - AssertionError
FAILED tests/test_example.py::test_two - ValueError
============================== 2 failed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert len(result) == 2
    test_names = [name for name, _ in result]
    reasons = [reason for _, reason in result]
    assert "tests/test_example.py::test_one" in test_names
    assert "AssertionError" in reasons
    assert "ValueError" in reasons


def test_extract_failed_names_fallback_to_headers() -> None:
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_one _________________________

AssertionError
_________________________ test_two _________________________

ValueError
============================== 2 failed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert len(result) == 2
    test_names = [name for name, _ in result]
    assert "test_one" in test_names
    assert "test_two" in test_names


def test_extract_failed_names_empty_for_passing() -> None:
    pytest_output = """
============================= test session starts ==============================
test_example.py::test_one PASSED
============================== 2 passed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert result == []


def test_extract_failed_names_ignores_separator_lines() -> None:
    """Separator lines like '_ _ _ _ _' should not match as test names."""
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_one _________________________

AssertionError
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

During handling of the above exception...
_________________________ test_two _________________________

AssertionError
"""
    result = extract_failed_test_names(pytest_output)
    test_names = [name for name, _ in result]
    assert len(result) == 2
    assert all("_ _ _" not in name for name in test_names)


def test_extract_failed_names_handles_ansi_codes() -> None:
    pytest_output = """
=========================== short test summary info ============================
\x1b[31mFAILED\x1b[0m tests/test_example.py::test_one - AssertionError
============================== 1 failed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert len(result) == 1
    test_name, reason = result[0]
    assert test_name == "tests/test_example.py::test_one"
    assert reason == "AssertionError"


# --- Output ordering tests ---


def test_pytest_summary_appears_last(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b"]]

    failure_output = (
        "test_foo.py::test_failing FAILED\n"
        "=================================== FAILURES ===================================\n"
        "_________________________ test_failing _________________________\n"
        "AssertionError\n"
        "=========================== short test summary info ============================\n"
    )

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = [
                sync_success(),
                Mock(returncode=0, stdout="all tests passed\n", stderr=""),
                sync_success(),
                Mock(returncode=1, stdout=failure_output, stderr=""),
            ]
            result = runner.invoke(cli, ["pytest"])

            failures_pos = result.output.find("FAILURES")
            summary_pos = result.output.find("Test Summary")
            failed_tests_pos = result.output.find("Failed tests:")
            assert failures_pos < summary_pos < failed_tests_pos


def test_pytest_extracts_failures_not_full_output(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    # Use two packages so single-package auto-verbose doesn't kick in
    pkg_a = create_pkg("pkg-a")
    pkg_b = create_pkg("pkg-b")

    failure_output = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.11.0\n"
        "plugins: asyncio-0.21.0\n"
        "collected 1 item\n"
        "\n"
        "=================================== FAILURES ===================================\n"
        "_________________________ test_failing _________________________\n"
        "AssertionError\n"
        "=============================== warnings summary ===============================\n"
        "DeprecationWarning: old API\n"
        "=========================== short test summary info ============================\n"
    )

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[pkg_a, pkg_b],
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = [
                sync_success(),
                Mock(returncode=1, stdout=failure_output, stderr=""),
                sync_success(),
                Mock(returncode=0, stdout="", stderr=""),
            ]
            result = runner.invoke(cli, ["pytest"])

            assert "AssertionError" in result.output
            # Should NOT include warnings or session info
            assert "DeprecationWarning" not in result.output
            assert "platform linux" not in result.output


def test_pytest_shows_failed_test_names_at_end(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    pkg_a = create_pkg("pkg-a")

    failure_output = (
        "=================================== FAILURES ===================================\n"
        "_________________________ test_failing _________________________\n"
        "AssertionError\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_example.py::test_failing - AssertionError\n"
    )

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[pkg_a],
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout=failure_output, stderr="")
            result = runner.invoke(cli, ["pytest"])

            summary_pos = result.output.find("Test Summary")
            failed_tests_pos = result.output.find("Failed tests:")
            assert failed_tests_pos > summary_pos
            assert "tests/test_example.py::test_failing" in result.output
            assert "[pkg-a]" in result.output


# --- Rich progress display tests ---


def test_render_progress_table_pending(
    packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    from rich.spinner import Spinner

    pkg = create_pkg("pkg-a")
    results: dict[str, dict[str, bool | str | float | None]] = {}
    start_times: dict[str, float] = {}
    spinners: dict[str, Spinner] = {}

    table = _render_progress_table([pkg], results, start_times, spinners)
    assert table.row_count == 1


def test_render_progress_table_running(
    packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    from rich.spinner import Spinner

    pkg = create_pkg("pkg-a")
    results: dict[str, dict[str, bool | str | float | None]] = {}
    start_times = {"pkg-a": time.time() - 5.0}
    spinners = {"pkg-a": Spinner("dots", style="yellow")}

    table = _render_progress_table([pkg], results, start_times, spinners)
    assert table.row_count == 1


def test_render_progress_table_completed(
    packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    from rich.spinner import Spinner

    pkg_a = create_pkg("pkg-a")
    pkg_b = create_pkg("pkg-b")
    results = {
        "pkg-a": {"success": True, "duration": 2.5, "stdout": "", "stderr": ""},
        "pkg-b": {"success": False, "duration": 3.0, "stdout": "", "stderr": ""},
    }
    start_times: dict[str, float] = {}
    spinners: dict[str, Spinner] = {}

    table = _render_progress_table([pkg_a, pkg_b], results, start_times, spinners)
    assert table.row_count == 2


def test_run_tests_with_rich_progress_returns_results(
    packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    pkg = create_pkg("pkg-a")

    with patch("dev_cli.commands.pytest_cmd.run_package_tests") as mock_run:
        mock_run.return_value = {
            "success": True,
            "stdout": "tests passed",
            "stderr": "",
            "duration": 1.5,
        }

        results = run_tests_with_rich_progress([pkg], (), max_workers=1)

        assert "pkg-a" in results
        assert results["pkg-a"]["success"] is True


def test_run_tests_with_rich_progress_multiple(
    packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    packages = [create_pkg(name) for name in ["pkg-a", "pkg-b", "pkg-c"]]
    call_count = 0

    def mock_run(
        pkg: PackageInfo, pytest_args: tuple[str, ...]
    ) -> dict[str, bool | str | float | None]:
        nonlocal call_count
        call_count += 1
        return {
            "success": pkg.name != "pkg-b",
            "stdout": "output",
            "stderr": "",
            "duration": 1.0,
        }

    with patch("dev_cli.commands.pytest_cmd.run_package_tests", side_effect=mock_run):
        results = run_tests_with_rich_progress(packages, (), max_workers=3)

        assert call_count == 3
        assert results["pkg-a"]["success"] is True
        assert results["pkg-b"]["success"] is False
        assert results["pkg-c"]["success"] is True


# --- TTY detection tests ---


def test_pytest_verbose_does_not_use_rich(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """--verbose uses simple output, not rich progress."""
    pkg_a = create_pkg("pkg-a")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[pkg_a],
    ):
        with patch(
            "dev_cli.commands.pytest_cmd.run_tests_with_rich_progress"
        ) as mock_rich:
            with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = Mock(
                    returncode=0, stdout="passed\n", stderr=""
                )
                runner.invoke(cli, ["pytest", "--verbose"])

                mock_rich.assert_not_called()
                mock_subprocess.assert_called()


def test_pytest_skips_rich_when_not_tty(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Rich progress is NOT used when stdout is not a TTY."""
    pkg_a = create_pkg("pkg-a")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[pkg_a],
    ):
        with patch(
            "dev_cli.commands.pytest_cmd.run_tests_with_rich_progress"
        ) as mock_rich:
            with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = Mock(
                    returncode=0, stdout="passed", stderr=""
                )
                runner.invoke(cli, ["pytest"])

                mock_rich.assert_not_called()
                mock_subprocess.assert_called()


# --- extract_test_counts tests ---


def test_extract_test_counts_passed_only() -> None:
    output = "============================== 42 passed in 3.2s =============================="
    assert extract_test_counts(output) == "42 passed"


def test_extract_test_counts_mixed() -> None:
    output = "============================== 1 failed, 42 passed in 3.2s =============================="
    assert extract_test_counts(output) == "1 failed, 42 passed"


def test_extract_test_counts_with_skipped() -> None:
    output = "============================== 10 passed, 2 skipped in 1.5s =============================="
    assert extract_test_counts(output) == "10 passed, 2 skipped"


def test_extract_test_counts_no_summary() -> None:
    assert extract_test_counts("no summary here\n") is None


def test_extract_test_counts_with_ansi() -> None:
    output = "\x1b[32m============================== 5 passed in 0.5s ==============================\x1b[0m"
    assert extract_test_counts(output) == "5 passed"


def test_extract_test_counts_without_timing() -> None:
    output = "============================== 42 passed =============================="
    assert extract_test_counts(output) == "42 passed"


# --- Single-package auto-verbose tests ---


def test_single_package_shows_full_output(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Single package runs show full pytest output without -v flag."""
    pkg_a = create_pkg("pkg-a")

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=[pkg_a],
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="collected 5 items\ntest_foo.py::test_one PASSED\n============================== 5 passed in 1.0s ==============================\n",
                stderr="",
            )
            result = runner.invoke(cli, ["pytest", "-p", "pkg-a"])

            # Should show full pytest output
            assert "collected 5 items" in result.output
            assert "test_foo.py::test_one PASSED" in result.output
            # Should still show summary
            assert "Test Summary" in result.output


# --- Test counts in compact output ---


def test_compact_output_shows_test_counts(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Non-TTY compact progress lines include test counts."""
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b"]]

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = [
                sync_success(),
                Mock(
                    returncode=0,
                    stdout="============================== 10 passed in 1.0s ==============================\n",
                    stderr="",
                ),
                sync_success(),
                Mock(
                    returncode=0,
                    stdout="============================== 5 passed in 0.5s ==============================\n",
                    stderr="",
                ),
            ]
            result = runner.invoke(cli, ["pytest"])

            assert (
                "PASSED (10 passed)" in result.output
                or "PASSED (5 passed)" in result.output
            )
            assert "15 tests total" in result.output


def test_summary_table_shows_test_counts(
    runner: CliRunner, packages_dir: Path, create_pkg: Callable[[str], PackageInfo]
) -> None:
    """Summary table includes test counts per package."""
    pkgs = [create_pkg(name) for name in ["pkg-a", "pkg-b"]]

    with patch(
        "dev_cli.commands.pytest_cmd.discover_test_packages",
        return_value=pkgs,
    ):
        with patch("dev_cli.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = [
                sync_success(),
                Mock(
                    returncode=0,
                    stdout="============================== 10 passed in 1.0s ==============================\n",
                    stderr="",
                ),
                sync_success(),
                Mock(
                    returncode=1,
                    stdout="=================================== FAILURES ===================================\n_________________________ test_bad _________________________\nAssertionError\n============================== 1 failed, 4 passed in 0.5s ==============================\n",
                    stderr="",
                ),
            ]
            result = runner.invoke(cli, ["pytest"])

            assert "PASSED (10 passed)" in result.output
            assert "FAILED (1 failed, 4 passed)" in result.output
            assert "15 tests total" in result.output
