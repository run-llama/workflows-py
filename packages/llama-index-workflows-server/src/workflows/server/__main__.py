import argparse
import importlib.util
import os
import sys
from pathlib import Path

import uvicorn

from workflows.server.server import WorkflowServer


def run_server() -> None:
    parser = argparse.ArgumentParser(description="Start the workflows server")
    parser.add_argument("file_path", nargs="?", help="Path to server application")
    args = parser.parse_args()

    if not args.file_path:
        usage = "Usage: python -m workflows.server <path_to_server_script>"
        print(usage, file=sys.stderr)
        sys.exit(1)

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found", file=sys.stderr)
        sys.exit(1)

    if not file_path.is_file():
        print(f"Error: '{file_path}' is not a file", file=sys.stderr)
        sys.exit(1)

    file_path = file_path.resolve()
    module_name = file_path.stem

    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Unable to get spec from module {module_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find a variable of type WorkflowServer
        server = None
        for attr_name in dir(module):
            attr_value = getattr(module, attr_name)
            if isinstance(attr_value, WorkflowServer):
                server = attr_value
                break

        if server is None:
            print(
                f"Error: No WorkflowServer instance found in '{args.file_path}'",
                file=sys.stderr,
            )
            sys.exit(1)

        # At this point, a WorkflowServer instance is guaranteed to exist
        assert server is not None

        host = os.environ.get("WORKFLOWS_PY_SERVER_HOST", "0.0.0.0")
        port = int(os.environ.get("WORKFLOWS_PY_SERVER_PORT", 8080))
        uvicorn.run(server.app, host=host, port=port)

    except Exception as e:
        print(f"Error loading or running server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_server()
