#!/usr/bin/env python
"""
Generate TypeScript SDK from WorkflowServer OpenAPI schema.

This script:
1. Generates OpenAPI schema from WorkflowServer
2. Calls @hey-api/openapi-ts to generate TypeScript client
"""

import json
import subprocess
import sys
from pathlib import Path

# Add parent directory to path to import workflows
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflows.server import WorkflowServer


def main() -> None:
    # Paths
    root_dir = Path(__file__).parent.parent
    openapi_path = root_dir / "openapi.json"
    client_sdk_dir = root_dir / "client-sdk"

    # Step 1: Generate OpenAPI schema
    print("Generating OpenAPI schema...")
    server = WorkflowServer()
    schema = server.openapi_schema()

    # Enhance the schema with better metadata
    schema["info"]["title"] = "LlamaIndex Workflows API"
    schema["info"]["description"] = "TypeScript client for LlamaIndex Workflows server"

    # Save OpenAPI schema
    with open(openapi_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"✅ OpenAPI schema saved to {openapi_path}")

    # Step 2: Install dependencies if needed
    print("\nInstalling pnpm dependencies...")
    subprocess.run(["pnpm", "install"], cwd=client_sdk_dir, check=True)

    # Step 3: Generate TypeScript client
    print("Generating TypeScript client...")
    subprocess.run(["pnpm", "run", "generate"], cwd=client_sdk_dir, check=True)

    # Step 4: Build the SDK
    print("Building TypeScript SDK...")
    subprocess.run(["pnpm", "run", "build"], cwd=client_sdk_dir, check=True)

    print(f"\n✅ SDK generated successfully in {client_sdk_dir}")
    print("   Generated files are in client-sdk/src/generated/")
    print("   Built files are in client-sdk/dist/")


if __name__ == "__main__":
    main()
