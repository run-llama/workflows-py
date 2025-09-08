"""Tests for scripts in the scripts/ directory."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_validate_version_matching() -> None:
    """Test the script succeeds when versions match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock pyproject.toml
        pyproject_path = Path(tmpdir) / "pyproject.toml"
        pyproject_path.write_text("""
[project]
name = "test-package"
version = "1.2.3"
description = "Test package"
""")

        # Create scripts directory and copy the script
        scripts_dir = Path(tmpdir) / "scripts"
        scripts_dir.mkdir()
        script_path = scripts_dir / "validate_version.py"
        original_script = Path("scripts/validate_version.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/tags/v1.2.3"  # Matches the mock version

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )
        assert result.returncode == 0
        assert "Version validated: 1.2.3" in result.stdout


def test_validate_version_not_matching() -> None:
    """Test the script fails when versions don't match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock pyproject.toml
        pyproject_path = Path(tmpdir) / "pyproject.toml"
        pyproject_path.write_text("""
[project]
name = "test-package"
version = "1.2.3"
description = "Test package"
""")

        # Create scripts directory and copy the script
        scripts_dir = Path(tmpdir) / "scripts"
        scripts_dir.mkdir()
        script_path = scripts_dir / "validate_version.py"
        original_script = Path("scripts/validate_version.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/tags/v9.9.9"  # Doesn't match

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )
        assert result.returncode == 1
        assert "doesn't match pyproject.toml version" in result.stdout


def test_validate_version_not_a_tag() -> None:
    """Test the script exits gracefully when not a tag push."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock pyproject.toml
        pyproject_path = Path(tmpdir) / "pyproject.toml"
        pyproject_path.write_text("""
[project]
name = "test-package"
version = "1.2.3"
""")

        # Create scripts directory and copy the script
        scripts_dir = Path(tmpdir) / "scripts"
        scripts_dir.mkdir()
        script_path = scripts_dir / "validate_version.py"
        original_script = Path("scripts/validate_version.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/heads/main"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )
        assert result.returncode == 1
        assert "Not a tag push" in result.stdout


def test_validate_version_no_version() -> None:
    """Test the script handles missing version gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock pyproject.toml without version
        pyproject_path = Path(tmpdir) / "pyproject.toml"
        pyproject_path.write_text("""
[project]
name = "test-package"
description = "Test package"
""")

        # Create scripts directory and copy the script
        scripts_dir = Path(tmpdir) / "scripts"
        scripts_dir.mkdir()
        script_path = scripts_dir / "validate_version.py"
        original_script = Path("scripts/validate_version.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/tags/v1.0.0"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )
        assert result.returncode == 1
        # The error message will vary based on tomllib vs line parsing


def test_detect_change_type_patch() -> None:
    """Test detecting patch version change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize git repo with tags
        subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=tmpdir, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=tmpdir, check=True
        )

        # Create a dummy file and make commits with tags
        dummy_file = Path(tmpdir) / "dummy.txt"
        dummy_file.write_text("v1.0.0")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v1.0.0"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v1.0.0"], cwd=tmpdir, check=True)

        dummy_file.write_text("v1.0.1")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v1.0.1"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v1.0.1"], cwd=tmpdir, check=True)

        # Copy the script
        script_path = Path(tmpdir) / "detect_change_type.py"
        original_script = Path("scripts/detect_change_type.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/tags/v1.0.1"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert "Current tag: v1.0.1" in result.stdout
        assert "Previous tag: v1.0.0" in result.stdout
        assert "Change type: patch" in result.stdout


def test_detect_change_type_minor() -> None:
    """Test detecting minor version change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize git repo with tags
        subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=tmpdir, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=tmpdir, check=True
        )

        # Create tags
        dummy_file = Path(tmpdir) / "dummy.txt"
        dummy_file.write_text("v1.0.0")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v1.0.0"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v1.0.0"], cwd=tmpdir, check=True)

        dummy_file.write_text("v1.1.0")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v1.1.0"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v1.1.0"], cwd=tmpdir, check=True)

        # Copy the script
        script_path = Path(tmpdir) / "detect_change_type.py"
        original_script = Path("scripts/detect_change_type.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/tags/v1.1.0"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert "Change type: minor" in result.stdout


def test_detect_change_type_major() -> None:
    """Test detecting major version change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize git repo with tags
        subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=tmpdir, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=tmpdir, check=True
        )

        # Create tags
        dummy_file = Path(tmpdir) / "dummy.txt"
        dummy_file.write_text("v1.0.0")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v1.0.0"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v1.0.0"], cwd=tmpdir, check=True)

        dummy_file.write_text("v2.0.0")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v2.0.0"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v2.0.0"], cwd=tmpdir, check=True)

        # Copy the script
        script_path = Path(tmpdir) / "detect_change_type.py"
        original_script = Path("scripts/detect_change_type.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/tags/v2.0.0"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert "Change type: major" in result.stdout


def test_detect_change_type_not_a_tag() -> None:
    """Test the script handles non-tag refs gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "detect_change_type.py"
        original_script = Path("scripts/detect_change_type.py")
        script_path.write_text(original_script.read_text())

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/heads/main"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert "Not a tag push" in result.stdout


def test_detect_change_type_github_output() -> None:
    """Test that the script writes to GITHUB_OUTPUT when set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize git repo with tags
        subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=tmpdir, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=tmpdir, check=True
        )

        dummy_file = Path(tmpdir) / "dummy.txt"
        dummy_file.write_text("v1.0.0")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v1.0.0"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v1.0.0"], cwd=tmpdir, check=True)

        dummy_file.write_text("v1.0.1")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)
        subprocess.run(
            ["git", "commit", "-m", "v1.0.1"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "tag", "v1.0.1"], cwd=tmpdir, check=True)

        # Copy the script
        script_path = Path(tmpdir) / "detect_change_type.py"
        original_script = Path("scripts/detect_change_type.py")
        script_path.write_text(original_script.read_text())

        # Create output file
        output_file = Path(tmpdir) / "github_output.txt"

        env = os.environ.copy()
        env["GITHUB_REF"] = "refs/tags/v1.0.1"
        env["GITHUB_OUTPUT"] = str(output_file)

        subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        # Check output file
        output_content = output_file.read_text()
        assert "change_type=patch" in output_content
