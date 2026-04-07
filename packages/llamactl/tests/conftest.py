import os
import shutil
import tempfile

import pytest

"""Test session configuration for isolating per-worker state.

Each xdist worker gets a unique HOME and LLAMACTL_CONFIG_DIR so migrations and
SQLite DBs do not clash across processes.
"""

# Base temp dir for the whole session
_TEST_HOME = tempfile.mkdtemp(prefix="llamactl_test_home_")

# Worker-specific subdir (e.g. gw0, gw1). Falls back to 'main' without xdist.
_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")
_WORKER_HOME = os.path.join(_TEST_HOME, _WORKER_ID)
_WORKER_CONFIG = os.path.join(_WORKER_HOME, ".config", "llamactl")

# Ensure directories exist
os.makedirs(_WORKER_CONFIG, exist_ok=True)

# Isolate HOME and config per worker
os.environ["HOME"] = _WORKER_HOME
os.environ.setdefault("TERM", "xterm")
os.environ["LLAMACTL_CONFIG_DIR"] = _WORKER_CONFIG


def pytest_sessionfinish(session: pytest.Session, exitstatus: pytest.ExitCode) -> None:
    try:
        shutil.rmtree(_TEST_HOME, ignore_errors=True)
    except Exception:
        pass
