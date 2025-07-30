import argparse
import importlib.util
import os
import sys
from pathlib import Path

import uvicorn


def run_server() -> None:
    parser = argparse.ArgumentParser(description="Start the workflows server")
    parser.add_argument("file_path", nargs="?", help="Path to server application")
    args = parser.parse_args()

    if not args.file_path:
        print("You have to provide a Python script defining a server instance")
        sys.exit(1)

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)

    if not file_path.is_file():
        print(f"Error: '{file_path}' is not a file")
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

        # Check if 'server' variable exists
        if not hasattr(module, "server"):
            print(f"Error: No 'server' variable found in '{args.file_path}'")
            sys.exit(1)

        server = getattr(module, "server")

        host = os.environ.get("WORKFLOWS_PY_SERVER_HOST", "0.0.0.0")
        port = int(os.environ.get("WORKFLOWS_PY_SERVER_PORT", 8080))
        uvicorn.run(server.app, host=host, port=port)

    except Exception as e:
        print(f"Error loading or running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()
