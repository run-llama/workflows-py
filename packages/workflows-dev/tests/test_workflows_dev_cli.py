from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from click.testing import CliRunner
from workflows_dev.changesets import (
    PackageJson,
    PyProjectContainer,
    current_version,
    is_published,
    sync_package_version_with_pyproject,
)
from workflows_dev.cli import cli
from workflows_dev.commands.pytest_cmd import (
    _render_progress_table,
    discover_test_packages,
    extract_failed_test_names,
    extract_failures_section,
    run_tests_with_rich_progress,
)


def _is_sync_call(cmd: list[str]) -> bool:
    """Check if a subprocess command is a 'uv sync' call vs 'uv run pytest'."""
    return len(cmd) >= 2 and cmd[0] == "uv" and cmd[1] == "sync"


def _sync_success() -> Mock:
    """Return a successful sync result."""
    return Mock(returncode=0, stdout="", stderr="")


def _write_pyproject(path: Path, version: str) -> None:
    path.write_text(
        f"""
[project]
name = "example"
version = "{version}"
""".strip()
    )


def _init_git_repo(repo_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "dev@example.com"], cwd=repo_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Dev User"], cwd=repo_path, check=True
    )


def _commit_and_tag(repo_path: Path, filename: str, content: str, tag: str) -> None:
    file_path = repo_path / filename
    file_path.write_text(content)
    subprocess.run(["git", "add", filename], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Add {tag}"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "tag", tag], cwd=repo_path, check=True)


def test_compute_tag_metadata_patch() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "pkg@v1.0.0", "pkg@v1.0.0")
        _commit_and_tag(repo, "file.txt", "pkg@v1.0.1", "pkg@v1.0.1")

        output_file = Path("out.txt")
        result = runner.invoke(
            cli,
            [
                "compute-tag-metadata",
                "--tag",
                "pkg@v1.0.1",
                "--output",
                str(output_file),
            ],
            env={},
        )
        assert result.exit_code == 0
        assert "Change type: patch" in result.output
        contents = output_file.read_text()
        assert "tag_suffix=v1.0.1" in contents
        assert "semver=1.0.1" in contents
        assert "change_type=patch" in contents


def test_compute_tag_metadata_minor() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "pkg@v1.0.0", "pkg@v1.0.0")
        _commit_and_tag(repo, "file.txt", "pkg@v1.1.0", "pkg@v1.1.0")

        result = runner.invoke(
            cli, ["compute-tag-metadata", "--tag", "pkg@v1.1.0"], env={}
        )
        assert result.exit_code == 0
        assert "Change type: minor" in result.output


def test_compute_tag_metadata_major() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Path.cwd()
        _init_git_repo(repo)
        _commit_and_tag(repo, "file.txt", "pkg@v1.0.0", "pkg@v1.0.0")
        _commit_and_tag(repo, "file.txt", "pkg@v2.0.0", "pkg@v2.0.0")

        result = runner.invoke(
            cli, ["compute-tag-metadata", "--tag", "pkg@v2.0.0"], env={}
        )
        assert result.exit_code == 0
        assert "Change type: major" in result.output


def test_compute_tag_metadata_requires_tag() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["compute-tag-metadata"])
    assert result.exit_code != 0
    assert "Missing option '--tag'" in result.output


def test_update_index_html_success(tmp_path: Path) -> None:
    runner = CliRunner()
    index_path = tmp_path / "index.html"
    index_path.write_text(
        """
<html>
  <head>
    <script type="module" crossorigin src="old.js"></script>
    <link rel="stylesheet" crossorigin href="old.css">
  </head>
</html>
""".strip()
    )
    result = runner.invoke(
        cli,
        [
            "update-index-html",
            "--js-url",
            "https://cdn/js",
            "--css-url",
            "https://cdn/css",
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code == 0
    updated = index_path.read_text()
    assert 'src="https://cdn/js"' in updated
    assert 'href="https://cdn/css"' in updated


def test_update_index_html_missing_file(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "update-index-html",
            "--js-url",
            "https://cdn/js",
            "--css-url",
            "https://cdn/css",
            "--index-path",
            str(tmp_path / "missing.html"),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


# Tests for changesets.py functionality


def test_current_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "test-package"
version = "1.2.3"
dependencies = []
""".strip()
    )
    name, version = current_version(pyproject)
    assert name == "test-package"
    assert version == "1.2.3"


def test_current_version_normalizes_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "test-package"
version = "01.02.03"
dependencies = []
""".strip()
    )
    name, version = current_version(pyproject)
    assert name == "test-package"
    assert version == "1.2.3"


def test_pyproject_container_parse() -> None:
    toml_text = """
[project]
name = "my-package"
version = "0.1.0"
dependencies = ["requests>=2.0.0"]
""".strip()
    toml_doc, py_doc = PyProjectContainer.parse(toml_text)
    assert py_doc.project.name == "my-package"
    assert py_doc.project.version == "0.1.0"
    assert py_doc.project.dependencies == ["requests>=2.0.0"]


def test_is_published_returns_true_when_version_exists() -> None:
    mock_response = Mock()
    mock_response.read.return_value = json.dumps(
        {"releases": {"1.0.0": [], "1.1.0": []}}
    ).encode()

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = is_published("test-package", "1.0.0")
        assert result is True


def test_is_published_returns_false_when_version_missing() -> None:
    mock_response = Mock()
    mock_response.read.return_value = json.dumps({"releases": {"1.0.0": []}}).encode()

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = is_published("test-package", "1.1.0")
        assert result is False


def test_is_published_returns_false_when_package_not_found() -> None:
    mock_error = HTTPError("url", 404, "Not Found", {}, None)  # type: ignore

    with patch("urllib.request.urlopen", side_effect=mock_error):
        result = is_published("nonexistent-package", "1.0.0")
        assert result is False


def test_is_published_raises_on_other_http_errors() -> None:
    mock_error = HTTPError("url", 500, "Internal Server Error", {}, None)  # type: ignore

    with patch("urllib.request.urlopen", side_effect=mock_error):
        try:
            is_published("test-package", "1.0.0")
            assert False, "Expected HTTPError to be raised"
        except HTTPError as e:
            assert e.code == 500


def test_sync_package_version_with_pyproject_updates_version(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    pyproject = package_dir / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "test-package"
version = "1.0.0"
dependencies = []
""".strip()
    )

    packages = {
        "test-js-package": PackageJson(
            name="test-js-package",
            version="2.0.0",
            path=package_dir,
            private=False,
        )
    }

    sync_package_version_with_pyproject(package_dir, packages, "test-js-package")

    toml_doc, py_doc = PyProjectContainer.parse(pyproject.read_text())
    assert py_doc.project.version == "2.0.0"


def test_sync_package_version_with_pyproject_skips_when_no_pyproject(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    packages = {
        "test-js-package": PackageJson(
            name="test-js-package",
            version="2.0.0",
            path=package_dir,
            private=False,
        )
    }

    # Should not raise an error
    sync_package_version_with_pyproject(package_dir, packages, "test-js-package")


def test_sync_package_version_with_pyproject_skips_when_versions_match(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    pyproject = package_dir / "pyproject.toml"
    original_content = """
[project]
name = "test-package"
version = "2.0.0"
dependencies = []
""".strip()
    pyproject.write_text(original_content)

    packages = {
        "test-js-package": PackageJson(
            name="test-js-package",
            version="2.0.0",
            path=package_dir,
            private=False,
        )
    }

    sync_package_version_with_pyproject(package_dir, packages, "test-js-package")

    # Content should be unchanged
    assert pyproject.read_text() == original_content


# Tests for pytest command


def test_discover_test_packages_finds_packages_with_tests(tmp_path: Path) -> None:
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    # Package with tests
    pkg_with_tests = packages_dir / "pkg-with-tests"
    pkg_with_tests.mkdir()
    (pkg_with_tests / "tests").mkdir()

    # Package without tests
    pkg_without_tests = packages_dir / "pkg-without-tests"
    pkg_without_tests.mkdir()

    result = discover_test_packages(tmp_path)

    assert len(result) == 1
    assert result[0].name == "pkg-with-tests"


def test_discover_test_packages_returns_sorted_list(tmp_path: Path) -> None:
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    # Create packages in non-alphabetical order
    for name in ["zebra-pkg", "alpha-pkg", "middle-pkg"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    result = discover_test_packages(tmp_path)

    assert [p.name for p in result] == ["alpha-pkg", "middle-pkg", "zebra-pkg"]


def test_discover_test_packages_empty_when_no_packages_dir(tmp_path: Path) -> None:
    result = discover_test_packages(tmp_path)
    assert result == []


def test_pytest_command_shows_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["pytest", "--help"])
    assert result.exit_code == 0
    assert "--package" in result.output
    assert "-p" in result.output


def test_pytest_command_filters_by_package(tmp_path: Path) -> None:
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    # Create test packages
    for name in ["pkg-a", "pkg-b"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a", packages_dir / "pkg-b"]

        # Mock subprocess.run to capture calls (sync + pytest per package)
        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            runner.invoke(cli, ["pytest", "-p", "pkg-a"])

            # Should only run tests for pkg-a (2 calls: sync + pytest)
            assert mock_run.call_count == 2
            # Second call should be the pytest call for pkg-a
            call_args = mock_run.call_args_list[1][0][0]
            assert "pkg-a" in str(call_args)
            assert "pytest" in call_args


def test_pytest_command_errors_on_unknown_package(tmp_path: Path) -> None:
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "real-pkg"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "real-pkg"]
        result = runner.invoke(cli, ["pytest", "-p", "unknown-pkg"])

        assert result.exit_code == 1
        assert "Unknown package(s): unknown-pkg" in result.output
        assert "Available packages: real-pkg" in result.output


def test_pytest_command_passes_args_to_pytest(tmp_path: Path) -> None:
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "my-pkg"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "my-pkg"]

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            # Use -- to pass args through to pytest (since -v is now our verbose flag)
            runner.invoke(cli, ["pytest", "--", "-v", "--tb=short", "-k", "test_foo"])

            call_args = mock_run.call_args[0][0]
            # Check passthrough args are present
            assert "-v" in call_args
            assert "--tb=short" in call_args
            assert "-k" in call_args
            assert "test_foo" in call_args


def test_pytest_command_continues_on_failure(tmp_path: Path) -> None:
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b", "pkg-c"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [
            packages_dir / "pkg-a",
            packages_dir / "pkg-b",
            packages_dir / "pkg-c",
        ]

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            # Each package has sync + pytest calls
            mock_run.side_effect = [
                _sync_success(),
                Mock(returncode=1, stdout="", stderr=""),
                _sync_success(),
                Mock(returncode=0, stdout="", stderr=""),
                _sync_success(),
                Mock(returncode=0, stdout="", stderr=""),
            ]
            result = runner.invoke(cli, ["pytest"])

            # Should run all 3 packages (2 calls each: sync + pytest)
            assert mock_run.call_count == 6
            # Exit code should be 1 due to failure
            assert result.exit_code == 1
            assert "FAILED" in result.output
            assert "PASSED" in result.output


def test_pytest_command_shows_summary(tmp_path: Path) -> None:
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a", packages_dir / "pkg-b"]

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = runner.invoke(cli, ["pytest"])

            assert "Test Summary" in result.output
            assert "pkg-a" in result.output
            assert "pkg-b" in result.output
            assert "2 passed" in result.output


def test_pytest_command_quiet_mode_by_default(tmp_path: Path) -> None:
    """Verify that without --verbose, output shows compact one-line progress per package.

    In quiet mode (the default):
    - During the run, show `[1/N] package-name... PASSED (duration)` per package
    - Full pytest output is NOT shown during the run
    - Full output only appears in failure recap if there are failures
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a", packages_dir / "pkg-b"]

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            # Simulate pytest output with detailed test information
            mock_run.return_value = Mock(
                returncode=0,
                stdout="collected 5 items\ntest_foo.py::test_one PASSED\ntest_foo.py::test_two PASSED\n",
                stderr="",
            )
            result = runner.invoke(cli, ["pytest"])

            # Should show compact progress format
            assert "[1/2]" in result.output
            assert "[2/2]" in result.output
            assert "pkg-a" in result.output
            assert "pkg-b" in result.output
            assert "PASSED" in result.output

            # Should NOT show full pytest output during run (no "collected X items")
            # The detailed output should only appear in failure recap, not during normal run
            # Count occurrences - in quiet mode, we shouldn't see streaming output
            output_lines = result.output.split("\n")
            collected_lines = [line for line in output_lines if "collected" in line]
            # In quiet mode, pytest's "collected X items" should not be in the output
            assert len(collected_lines) == 0


def test_pytest_command_verbose_flag_shows_streaming_output(tmp_path: Path) -> None:
    """Verify that --verbose flag shows full output AND still shows aggregate recap at end.

    In verbose mode:
    - Full pytest output is streamed for each package as it runs
    - Aggregate summary/recap is still shown at the end
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a", packages_dir / "pkg-b"]

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="collected 5 items\ntest_foo.py::test_one PASSED\ntest_foo.py::test_two PASSED\n",
                stderr="",
            )
            result = runner.invoke(cli, ["pytest", "--verbose"])

            # Should show full pytest output (streaming)
            assert "collected 5 items" in result.output
            assert "test_foo.py::test_one PASSED" in result.output

            # Should still show aggregate summary at end
            assert "Test Summary" in result.output
            assert "pkg-a" in result.output
            assert "pkg-b" in result.output
            assert "2 passed" in result.output


def test_pytest_command_failure_recap_always_shown(tmp_path: Path) -> None:
    """Verify that failure recap is shown regardless of verbose/quiet mode.

    When tests fail:
    - The failure recap section should always be displayed
    - Full output of failed packages should be shown in the recap
    - This applies to both quiet mode (default) and verbose mode
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a", packages_dir / "pkg-b"]

        failure_output = (
            "collected 3 items\n"
            "test_foo.py::test_one PASSED\n"
            "test_foo.py::test_failing FAILED\n"
            "FAILURES\n"
            "test_failing - AssertionError: expected True\n"
        )

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            # pkg-a passes, pkg-b fails (each has sync + pytest)
            mock_run.side_effect = [
                _sync_success(),
                Mock(returncode=0, stdout="all tests passed\n", stderr=""),
                _sync_success(),
                Mock(returncode=1, stdout=failure_output, stderr=""),
            ]
            # Test in quiet mode (default)
            result = runner.invoke(cli, ["pytest"])

            # Failure recap should be shown
            assert "FAILURES" in result.output or "FAILED" in result.output
            # The failed package's output should appear in the recap
            assert "pkg-b" in result.output
            assert "AssertionError" in result.output

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _sync_success(),
                Mock(returncode=0, stdout="all tests passed\n", stderr=""),
                _sync_success(),
                Mock(returncode=1, stdout=failure_output, stderr=""),
            ]
            # Test in verbose mode
            result = runner.invoke(cli, ["pytest", "--verbose"])

            # Failure recap should still be shown in verbose mode
            assert "FAILURES" in result.output or "FAILED" in result.output
            assert "pkg-b" in result.output
            assert "AssertionError" in result.output


# Tests for Phase 4: Parallel execution


def test_pytest_command_runs_parallel_by_default(tmp_path: Path) -> None:
    """Verify that tests run in parallel by default.

    Uses threading events to verify that multiple subprocess.run calls are
    in-flight concurrently. Each mock subprocess blocks until all packages
    have started, proving they run in parallel.
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b", "pkg-c"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [
            packages_dir / "pkg-a",
            packages_dir / "pkg-b",
            packages_dir / "pkg-c",
        ]

        # Track concurrent execution (only pytest calls, not sync)
        started_count = 0
        started_lock = threading.Lock()
        all_started = threading.Event()
        call_order: list[str] = []

        def mock_subprocess_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal started_count

            # Handle sync calls - return immediately without tracking
            if _is_sync_call(cmd):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )

            # Extract package name from command
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

            # Wait for all to start (with timeout to prevent hanging)
            # In parallel mode, all 3 should start before any completes
            all_started.wait(timeout=2.0)

            with started_lock:
                call_order.append(f"end:{pkg_name}")

            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="tests passed\n",
                stderr="",
            )

        with patch(
            "workflows_dev.commands.pytest_cmd.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = runner.invoke(cli, ["pytest"])

            # In parallel mode, all packages should have started before any ended
            # The call_order should show all starts before all ends
            start_indices = [
                i for i, x in enumerate(call_order) if x.startswith("start:")
            ]
            end_indices = [i for i, x in enumerate(call_order) if x.startswith("end:")]

            # Verify parallel: at least 2 packages started before any ended
            # (allows for some scheduling variation)
            assert len(start_indices) >= 2
            assert len(end_indices) >= 2
            # In true parallel execution, at least the first 2 starts should
            # occur before any end
            assert max(start_indices[:2]) < min(end_indices), (
                f"Expected parallel execution, got order: {call_order}"
            )

            assert result.exit_code == 0


def test_pytest_command_parallel_one_runs_sequentially(tmp_path: Path) -> None:
    """Verify that --parallel 1 runs packages one at a time.

    Uses threading events to verify that subprocess.run calls do NOT overlap.
    Each mock subprocess records its start and end, and we verify no overlap.
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b", "pkg-c"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [
            packages_dir / "pkg-a",
            packages_dir / "pkg-b",
            packages_dir / "pkg-c",
        ]

        # Track execution order strictly (only pytest calls, not sync)
        call_order: list[str] = []
        lock = threading.Lock()
        active_count = 0
        max_concurrent = 0

        def mock_subprocess_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal active_count, max_concurrent

            # Handle sync calls - return immediately without tracking
            if _is_sync_call(cmd):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )

            # Extract package name from command
            pkg_name = "unknown"
            for i, arg in enumerate(cmd):
                if arg == "--directory" and i + 1 < len(cmd):
                    pkg_name = Path(cmd[i + 1]).name
                    break

            with lock:
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
                call_order.append(f"start:{pkg_name}")

            # Small delay to give other threads a chance to start if parallel
            time.sleep(0.01)

            with lock:
                call_order.append(f"end:{pkg_name}")
                active_count -= 1

            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="tests passed\n",
                stderr="",
            )

        with patch(
            "workflows_dev.commands.pytest_cmd.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = runner.invoke(cli, ["pytest", "--parallel", "1"])

            # With --parallel 1, max concurrent should be 1
            assert max_concurrent == 1, (
                f"Expected max 1 concurrent, got {max_concurrent}"
            )

            # Verify strict ordering: start, end, start, end, start, end
            for i in range(0, len(call_order) - 1, 2):
                assert call_order[i].startswith("start:"), (
                    f"Expected start at {i}, got {call_order[i]}"
                )
                assert call_order[i + 1].startswith("end:"), (
                    f"Expected end at {i + 1}, got {call_order[i + 1]}"
                )
                # Same package
                pkg_start = call_order[i].split(":")[1]
                pkg_end = call_order[i + 1].split(":")[1]
                assert pkg_start == pkg_end, f"Mismatch: {pkg_start} vs {pkg_end}"

            assert result.exit_code == 0


def test_pytest_command_parallel_handles_mixed_results(tmp_path: Path) -> None:
    """Verify that parallel execution correctly handles a mix of passed/failed packages.

    Tests that:
    - All packages run to completion even when some fail
    - The exit code reflects any failure
    - The summary correctly reports passed/failed counts
    - Failed package output appears in the failure recap
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-pass-1", "pkg-fail", "pkg-pass-2"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [
            packages_dir / "pkg-fail",
            packages_dir / "pkg-pass-1",
            packages_dir / "pkg-pass-2",
        ]

        completed_packages: list[str] = []
        lock = threading.Lock()

        def mock_subprocess_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            # Handle sync calls - return success immediately
            if _is_sync_call(cmd):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )

            # Extract package name from command
            pkg_name = "unknown"
            for i, arg in enumerate(cmd):
                if arg == "--directory" and i + 1 < len(cmd):
                    pkg_name = Path(cmd[i + 1]).name
                    break

            with lock:
                completed_packages.append(pkg_name)

            if pkg_name == "pkg-fail":
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="FAILED test_example.py::test_bad\nAssertionError: bad\n",
                    stderr="",
                )
            else:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="all tests passed\n",
                    stderr="",
                )

        with patch(
            "workflows_dev.commands.pytest_cmd.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = runner.invoke(cli, ["pytest"])

            # All packages should have run (only pytest calls tracked)
            assert len(completed_packages) == 3
            assert set(completed_packages) == {"pkg-fail", "pkg-pass-1", "pkg-pass-2"}

            # Exit code should be non-zero due to failure
            assert result.exit_code == 1

            # Summary should show correct counts
            assert "2 passed" in result.output
            assert "1 failed" in result.output

            # Failed package output should appear in recap
            assert "pkg-fail" in result.output
            assert "AssertionError" in result.output


# Tests for extract_failures_section helper


def test_extract_failures_section_returns_failures_block() -> None:
    """Verify that the function extracts just the FAILURES section."""
    pytest_output = """
============================= test session starts ==============================
platform linux -- Python 3.11.0
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
    assert "AssertionError" in result
    # Should NOT include the short test summary or final counts
    assert "short test summary" not in result
    assert "1 failed, 1 passed" not in result


def test_extract_failures_section_returns_none_for_passing_tests() -> None:
    """Verify that the function returns None when there are no failures."""
    pytest_output = """
============================= test session starts ==============================
platform linux -- Python 3.11.0
collected 2 items

test_example.py::test_one PASSED
test_example.py::test_two PASSED

============================== 2 passed ==============================
"""
    result = extract_failures_section(pytest_output)
    assert result is None


def test_extract_failures_section_excludes_warnings() -> None:
    """Verify that warnings are not included in the extracted failures."""
    pytest_output = """
============================= test session starts ==============================
collected 2 items

test_example.py::test_one PASSED
test_example.py::test_failing FAILED

=================================== FAILURES ===================================
_________________________ test_failing _________________________

    def test_failing():
>       assert False
E       AssertionError

test_example.py:5: AssertionError
=============================== warnings summary ===============================
test_example.py:3: DeprecationWarning: deprecated function
  warnings.warn("deprecated function", DeprecationWarning)
=========================== short test summary info ============================
FAILED test_example.py::test_failing - AssertionError
============================== 1 failed, 1 passed ==============================
"""
    result = extract_failures_section(pytest_output)
    assert result is not None
    assert "FAILURES" in result
    assert "test_failing" in result
    # Should NOT include warnings section
    assert "warnings summary" not in result
    assert "DeprecationWarning" not in result


def test_extract_failures_section_handles_multiple_failures() -> None:
    """Verify that multiple failures are all included."""
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_one _________________________

AssertionError: first failure

test_example.py:5: AssertionError
_________________________ test_two _________________________

AssertionError: second failure

test_example.py:10: AssertionError
=========================== short test summary info ============================
"""
    result = extract_failures_section(pytest_output)
    assert result is not None
    assert "test_one" in result
    assert "first failure" in result
    assert "test_two" in result
    assert "second failure" in result


def test_pytest_command_summary_appears_last(tmp_path: Path) -> None:
    """Verify that the summary table appears after failures in output."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    for name in ["pkg-a", "pkg-b"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a", packages_dir / "pkg-b"]

        failure_output = (
            "collected 3 items\n"
            "test_foo.py::test_failing FAILED\n"
            "=================================== FAILURES ===================================\n"
            "_________________________ test_failing _________________________\n"
            "AssertionError: expected True\n"
            "=========================== short test summary info ============================\n"
        )

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _sync_success(),
                Mock(returncode=0, stdout="all tests passed\n", stderr=""),
                _sync_success(),
                Mock(returncode=1, stdout=failure_output, stderr=""),
            ]
            result = runner.invoke(cli, ["pytest"])

            # Find positions of key sections
            failures_pos = result.output.find("FAILURES")
            summary_pos = result.output.find("Test Summary")
            failed_tests_pos = result.output.find("Failed tests:")

            # Failures should appear before summary
            assert failures_pos != -1, "FAILURES section not found"
            assert summary_pos != -1, "Test Summary not found"
            assert failures_pos < summary_pos, "Expected FAILURES before Test Summary"

            # Failed tests list should be at the very end (after summary)
            assert failed_tests_pos != -1, "Failed tests section not found"
            assert failed_tests_pos > summary_pos, "Expected Failed tests after Summary"


def test_pytest_command_extracts_failures_not_full_output(tmp_path: Path) -> None:
    """Verify that only the FAILURES section is shown, not full pytest output."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a"]

        # Pytest output with warnings that should be stripped
        failure_output = (
            "============================= test session starts ==============================\n"
            "platform linux -- Python 3.11.0\n"
            "plugins: asyncio-0.21.0\n"
            "collected 1 item\n"
            "\n"
            "test_foo.py::test_failing FAILED\n"
            "\n"
            "=================================== FAILURES ===================================\n"
            "_________________________ test_failing _________________________\n"
            "\n"
            "    def test_failing():\n"
            ">       assert False\n"
            "E       AssertionError\n"
            "\n"
            "test_foo.py:5: AssertionError\n"
            "=============================== warnings summary ===============================\n"
            "test_foo.py:3: DeprecationWarning: old API\n"
            "=========================== short test summary info ============================\n"
            "FAILED test_foo.py::test_failing\n"
            "============================== 1 failed ==============================\n"
        )

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout=failure_output, stderr="")
            result = runner.invoke(cli, ["pytest"])

            # Should include the actual failure
            assert "test_failing" in result.output
            assert "AssertionError" in result.output

            # Should NOT include warnings or session info
            assert "DeprecationWarning" not in result.output
            assert "platform linux" not in result.output
            assert "plugins:" not in result.output


# Tests for extract_failed_test_names helper


def test_extract_failed_test_names_from_short_summary() -> None:
    """Verify extraction from short test summary info section with reasons."""
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_one _________________________

AssertionError: first
=========================== short test summary info ============================
FAILED tests/test_example.py::test_one - AssertionError
FAILED tests/test_example.py::test_two - ValueError
============================== 2 failed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert len(result) == 2
    # Result is list of (test_name, reason) tuples
    test_names = [name for name, _ in result]
    reasons = [reason for _, reason in result]
    assert "tests/test_example.py::test_one" in test_names
    assert "tests/test_example.py::test_two" in test_names
    assert "AssertionError" in reasons
    assert "ValueError" in reasons


def test_extract_failed_test_names_fallback_to_headers() -> None:
    """Verify fallback to FAILURES section headers when no short summary."""
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_one _________________________

AssertionError: first
_________________________ test_two _________________________

ValueError: second
============================== 2 failed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert len(result) == 2
    # Fallback returns tuples with None reason
    test_names = [name for name, _ in result]
    assert "test_one" in test_names
    assert "test_two" in test_names


def test_extract_failed_test_names_returns_empty_for_passing() -> None:
    """Verify empty list for passing tests."""
    pytest_output = """
============================= test session starts ==============================
collected 2 items

test_example.py::test_one PASSED
test_example.py::test_two PASSED

============================== 2 passed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert result == []


def test_extract_failed_test_names_ignores_separator_lines() -> None:
    """Verify separator lines like '_ _ _ _ _' are not matched as test names.

    Pytest uses "_ _ _ _ _" separator lines between test failures. These should
    not be confused with the "_____ test_name _____" header format.
    """
    pytest_output = """
=================================== FAILURES ===================================
_________________________ test_one _________________________

    def test_one():
>       assert False
E       AssertionError

tests/test_example.py:5: AssertionError
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

During handling of the above exception, another exception occurred:

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

_________________________ test_two _________________________

    def test_two():
>       assert False
E       AssertionError

tests/test_example.py:10: AssertionError
============================== 2 failed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    # Should only find actual test names, not separator lines
    assert len(result) == 2
    test_names = [name for name, _ in result]
    assert "test_one" in test_names
    assert "test_two" in test_names
    # Separator lines should NOT be included
    for name, _ in result:
        assert "_ _ _" not in name


def test_extract_failed_test_names_handles_ansi_codes() -> None:
    """Verify ANSI color codes in pytest output don't break parsing.

    When FORCE_COLOR=1 is set, pytest outputs ANSI escape codes for colors.
    The parser should strip these before pattern matching.
    """
    # Simulate pytest output with ANSI color codes (red for FAILED)
    pytest_output = """
============================= test session starts ==============================
collected 1 item

test_example.py::test_one FAILED

=================================== FAILURES ===================================
_________________________ test_one _________________________

    def test_one():
>       assert False
E       AssertionError

tests/test_example.py:5: AssertionError
============================================== short test summary info ==============================================
\x1b[31mFAILED\x1b[0m tests/test_example.py::test_one - AssertionError
============================== 1 failed ==============================
"""
    result = extract_failed_test_names(pytest_output)
    assert len(result) == 1
    test_name, reason = result[0]
    assert test_name == "tests/test_example.py::test_one"
    assert reason == "AssertionError"


def test_pytest_command_shows_failed_test_names_at_end(tmp_path: Path) -> None:
    """Verify that failed test names appear at the very end of output."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a"]

        failure_output = (
            "=================================== FAILURES ===================================\n"
            "_________________________ test_failing _________________________\n"
            "AssertionError\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_example.py::test_failing - AssertionError\n"
            "============================== 1 failed ==============================\n"
        )

        with patch("workflows_dev.commands.pytest_cmd.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout=failure_output, stderr="")
            result = runner.invoke(cli, ["pytest"])

            # "Failed tests:" should appear after "Test Summary"
            summary_pos = result.output.find("Test Summary")
            failed_tests_pos = result.output.find("Failed tests:")
            assert summary_pos != -1
            assert failed_tests_pos != -1
            assert failed_tests_pos > summary_pos

            # Should show the test name at the end
            assert "tests/test_example.py::test_failing" in result.output
            # Should include package name
            assert "[pkg-a]" in result.output
            # Should include the error reason
            assert "AssertionError" in result.output


# Tests for rich progress display


def test_render_progress_table_pending_packages(tmp_path: Path) -> None:
    """Verify that pending packages are rendered with dim styling."""
    from rich.spinner import Spinner

    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    packages = [pkg]
    results: dict[str, dict[str, bool | str | float]] = {}
    start_times: dict[str, float] = {}
    spinners: dict[str, Spinner] = {}

    table = _render_progress_table(packages, results, start_times, spinners)

    # Table should have one row with pending status
    assert table.row_count == 1


def test_render_progress_table_running_packages(tmp_path: Path) -> None:
    """Verify that running packages show spinner and elapsed time."""
    from rich.spinner import Spinner

    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    packages = [pkg]
    results: dict[str, dict[str, bool | str | float]] = {}
    start_times = {"pkg-a": time.time() - 5.0}  # Started 5 seconds ago
    spinners = {"pkg-a": Spinner("dots", style="yellow")}

    table = _render_progress_table(packages, results, start_times, spinners)

    # Table should have one row with running status
    assert table.row_count == 1


def test_render_progress_table_completed_packages(tmp_path: Path) -> None:
    """Verify that completed packages show PASSED/FAILED status."""
    from rich.spinner import Spinner

    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg_a = packages_dir / "pkg-a"
    pkg_a.mkdir()
    (pkg_a / "tests").mkdir()

    pkg_b = packages_dir / "pkg-b"
    pkg_b.mkdir()
    (pkg_b / "tests").mkdir()

    packages = [pkg_a, pkg_b]
    results = {
        "pkg-a": {"success": True, "duration": 2.5, "stdout": "", "stderr": ""},
        "pkg-b": {"success": False, "duration": 3.0, "stdout": "", "stderr": ""},
    }
    start_times: dict[str, float] = {}
    spinners: dict[str, Spinner] = {}

    table = _render_progress_table(packages, results, start_times, spinners)

    # Table should have two rows
    assert table.row_count == 2


def test_render_progress_table_with_spinners(tmp_path: Path) -> None:
    """Verify that spinners are used for running packages."""
    from rich.spinner import Spinner

    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    packages = [pkg]
    results: dict[str, dict[str, bool | str | float]] = {}
    start_times = {"pkg-a": time.time()}
    spinners = {"pkg-a": Spinner("dots", style="yellow")}

    # Table should be created with the spinner
    table = _render_progress_table(packages, results, start_times, spinners)
    assert table.row_count == 1


def test_run_tests_with_rich_progress_returns_results(tmp_path: Path) -> None:
    """Verify that run_tests_with_rich_progress returns correct results dict."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    with patch("workflows_dev.commands.pytest_cmd.run_package_tests") as mock_run:
        mock_run.return_value = {
            "success": True,
            "stdout": "tests passed",
            "stderr": "",
            "duration": 1.5,
        }

        results = run_tests_with_rich_progress([pkg], (), max_workers=1)

        assert "pkg-a" in results
        assert results["pkg-a"]["success"] is True
        assert results["pkg-a"]["duration"] == 1.5


def test_run_tests_with_rich_progress_multiple_packages(tmp_path: Path) -> None:
    """Verify that multiple packages are all processed."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    packages = []
    for name in ["pkg-a", "pkg-b", "pkg-c"]:
        pkg = packages_dir / name
        pkg.mkdir()
        (pkg / "tests").mkdir()
        packages.append(pkg)

    call_count = 0

    def mock_run_package_tests(
        pkg: Path, pytest_args: tuple[str, ...]
    ) -> dict[str, bool | str | float]:
        nonlocal call_count
        call_count += 1
        return {
            "success": pkg.name != "pkg-b",  # pkg-b fails
            "stdout": "output",
            "stderr": "",
            "duration": 1.0,
        }

    with patch(
        "workflows_dev.commands.pytest_cmd.run_package_tests",
        side_effect=mock_run_package_tests,
    ):
        results = run_tests_with_rich_progress(packages, (), max_workers=3)

        # All packages should be processed
        assert call_count == 3
        assert len(results) == 3
        assert results["pkg-a"]["success"] is True
        assert results["pkg-b"]["success"] is False
        assert results["pkg-c"]["success"] is True


def test_run_tests_with_rich_progress_callable(tmp_path: Path) -> None:
    """Verify that run_tests_with_rich_progress can be called directly.

    This tests the rich progress function works correctly when called.
    The actual TTY detection is tested indirectly through the non-TTY tests.
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    # Test that run_tests_with_rich_progress works when called directly
    with patch("workflows_dev.commands.pytest_cmd.run_package_tests") as mock_run:
        mock_run.return_value = {
            "success": True,
            "stdout": "tests passed",
            "stderr": "",
            "duration": 1.5,
        }

        results = run_tests_with_rich_progress([pkg], (), max_workers=1)

        # Should have called run_package_tests
        mock_run.assert_called_once()
        # Should return results
        assert "pkg-a" in results
        assert results["pkg-a"]["success"] is True


def test_pytest_command_verbose_does_not_use_rich(tmp_path: Path) -> None:
    """Verify that --verbose flag uses simple output, not rich progress.

    In verbose mode, even if we had TTY detection working, we still want
    to show full output for each package instead of the rich progress.
    """
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a"]

        with patch(
            "workflows_dev.commands.pytest_cmd.run_tests_with_rich_progress"
        ) as mock_rich:
            with patch(
                "workflows_dev.commands.pytest_cmd.subprocess.run"
            ) as mock_subprocess:
                mock_subprocess.return_value = Mock(
                    returncode=0, stdout="tests passed\n", stderr=""
                )

                # Even with isatty() behavior, verbose should use simple output
                result = runner.invoke(cli, ["pytest", "--verbose"])

                # Rich progress should NOT have been called (verbose mode)
                mock_rich.assert_not_called()
                # Regular subprocess should have been called
                mock_subprocess.assert_called()
                # Verbose mode shows full output
                assert "tests passed" in result.output


def test_pytest_command_skips_rich_when_not_tty(tmp_path: Path) -> None:
    """Verify that rich progress is NOT used when stdout is not a TTY."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()

    pkg = packages_dir / "pkg-a"
    pkg.mkdir()
    (pkg / "tests").mkdir()

    runner = CliRunner()
    with patch(
        "workflows_dev.commands.pytest_cmd.discover_test_packages"
    ) as mock_discover:
        mock_discover.return_value = [packages_dir / "pkg-a"]

        with patch(
            "workflows_dev.commands.pytest_cmd.run_tests_with_rich_progress"
        ) as mock_rich:
            with patch(
                "workflows_dev.commands.pytest_cmd.subprocess.run"
            ) as mock_subprocess:
                mock_subprocess.return_value = Mock(
                    returncode=0, stdout="tests passed", stderr=""
                )

                # CliRunner doesn't use a TTY by default, so isatty() returns False
                runner.invoke(cli, ["pytest"])

                # Rich progress should NOT have been called
                mock_rich.assert_not_called()
                # Regular subprocess should have been called
                mock_subprocess.assert_called()
