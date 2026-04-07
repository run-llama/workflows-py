#!/usr/bin/env python3
"""Convert .env file to a Kubernetes secret YAML manifest.

Handles JSON-encoded values (e.g. PEM keys stored with \\n escapes).
Usage: python3 tilt/env-to-secret.py > secret.yaml
"""

import json
import subprocess
import sys
from pathlib import Path

env_file = Path(__file__).parent.parent.parent / ".env"
if not env_file.exists():
    sys.exit(0)

cmd = [
    "kubectl",
    "create",
    "secret",
    "generic",
    "control-plane-secrets",
    "-n",
    "llama-agents",
    "--dry-run=client",
    "-o",
    "yaml",
]

for line in env_file.read_text().splitlines():
    if "=" not in line or line.startswith("#"):
        continue
    k, v = line.split("=", 1)
    if v.startswith('"') and v.endswith('"'):
        v = json.loads(v)
    cmd.append(f"--from-literal={k}={v}")

sys.stdout.buffer.write(subprocess.check_output(cmd))
