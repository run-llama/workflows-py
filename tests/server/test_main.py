# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from workflows.server.__main__ import run_server


def test_no_file_path_argument(capsys: Any) -> None:
    """Test that the script exits with usage message when no file path is provided."""
    with patch("sys.argv", ["workflows.server"]):
        with pytest.raises(SystemExit) as exc_info:
            run_server()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert (
            "Usage: python -m workflows.server <path_to_server_script>" in captured.err
        )


def test_nonexistent_file(capsys: Any) -> None:
    """Test that the script exits when file doesn't exist."""
    with patch("sys.argv", ["workflows.server", "/nonexistent/file.py"]):
        with pytest.raises(SystemExit) as exc_info:
            run_server()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: File '/nonexistent/file.py' not found" in captured.err


def test_directory_instead_of_file(capsys: Any, tmp_path: Path) -> None:
    """Test that the script exits when a directory is provided instead of a file."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    with patch("sys.argv", ["workflows.server", str(test_dir)]):
        with pytest.raises(SystemExit) as exc_info:
            run_server()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert f"Error: '{test_dir}' is not a file" in captured.err


def test_no_workflow_server_instance(capsys: Any, tmp_path: Path) -> None:
    """Test that the script exits when no WorkflowServer instance is found."""
    # Create a test Python file without a WorkflowServer instance
    test_file = tmp_path / "no_server.py"
    test_file.write_text("""
# A file without WorkflowServer instance
some_variable = "hello"
another_variable = 42
""")

    with patch("sys.argv", ["workflows.server", str(test_file)]):
        with pytest.raises(SystemExit) as exc_info:
            run_server()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert (
            f"Error: No WorkflowServer instance found in '{test_file}'" in captured.err
        )


def test_workflow_server_with_custom_name(tmp_path: Path) -> None:
    """Test that the script finds WorkflowServer instance with any variable name."""
    # Create a test Python file with WorkflowServer instance named differently
    test_file = tmp_path / "custom_server.py"
    test_file.write_text("""
from workflows.server.server import WorkflowServer

# WorkflowServer with custom name
my_app = WorkflowServer()
""")

    with patch("sys.argv", ["workflows.server", str(test_file)]):
        with patch("uvicorn.run") as mock_uvicorn:
            run_server()

            # Verify uvicorn.run was called with the server's app
            mock_uvicorn.assert_called_once()
            args, kwargs = mock_uvicorn.call_args
            assert hasattr(args[0], "routes")  # Check it's a Starlette app
            assert kwargs["host"] == "0.0.0.0"
            assert kwargs["port"] == 8080


def test_multiple_workflow_servers_uses_first(tmp_path: Path) -> None:
    """Test that when multiple WorkflowServer instances exist, the first one is used."""
    test_file = tmp_path / "multiple_servers.py"
    test_file.write_text("""
from workflows.server.server import WorkflowServer

# Multiple WorkflowServer instances
first_server = WorkflowServer()
second_server = WorkflowServer()
""")

    with patch("sys.argv", ["workflows.server", str(test_file)]):
        with patch("uvicorn.run") as mock_uvicorn:
            run_server()

            # Should use the first server found
            mock_uvicorn.assert_called_once()


def test_environment_variables(tmp_path: Path) -> None:
    """Test that environment variables are used for host and port."""
    test_file = tmp_path / "env_test.py"
    test_file.write_text("""
from workflows.server.server import WorkflowServer

server = WorkflowServer()
""")

    test_host = "127.0.0.1"
    test_port = "9000"

    with patch("sys.argv", ["workflows.server", str(test_file)]):
        with patch.dict(
            os.environ,
            {
                "WORKFLOWS_PY_SERVER_HOST": test_host,
                "WORKFLOWS_PY_SERVER_PORT": test_port,
            },
        ):
            with patch("uvicorn.run") as mock_uvicorn:
                run_server()

                mock_uvicorn.assert_called_once()
                args, kwargs = mock_uvicorn.call_args
                assert kwargs["host"] == test_host
                assert kwargs["port"] == int(test_port)


def test_module_loading_error(capsys: Any, tmp_path: Path) -> None:
    """Test that script handles module loading errors gracefully."""
    # Create a file with syntax error
    test_file = tmp_path / "syntax_error.py"
    test_file.write_text("""
from workflows.server.server import WorkflowServer

# Syntax error
def invalid_syntax(
    server = WorkflowServer()
""")

    with patch("sys.argv", ["workflows.server", str(test_file)]):
        with pytest.raises(SystemExit) as exc_info:
            run_server()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error loading or running server:" in captured.err


def test_spec_creation_failure(capsys: Any, tmp_path: Path) -> None:
    """Test handling of spec creation failure."""
    test_file = tmp_path / "test.py"
    test_file.write_text("server = 'test'")

    with patch("sys.argv", ["workflows.server", str(test_file)]):
        with patch("importlib.util.spec_from_file_location", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                run_server()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Unable to get spec from module" in captured.err


def test_non_workflow_server_objects_ignored(tmp_path: Path) -> None:
    """Test that objects that aren't WorkflowServer instances are ignored."""
    test_file = tmp_path / "mixed_objects.py"
    test_file.write_text("""
from workflows.server.server import WorkflowServer

# Various non-WorkflowServer objects
string_var = "not a server"
number_var = 42
list_var = [1, 2, 3]
dict_var = {"key": "value"}

class FakeServer:
    pass

fake_server = FakeServer()

# The actual WorkflowServer
real_server = WorkflowServer()
""")

    with patch("sys.argv", ["workflows.server", str(test_file)]):
        with patch("uvicorn.run") as mock_uvicorn:
            run_server()

            # Should find and use the real WorkflowServer
            mock_uvicorn.assert_called_once()
