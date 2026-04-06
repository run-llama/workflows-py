from pathlib import Path

from llama_agents.core._compat import load_toml_file


def _get_min_py_version(requires_python: str) -> str:
    min_v = requires_python.split(",")[0].strip()
    return (
        min_v.replace("=", "")
        .replace(">", "")
        .replace("<", "")
        .replace("~", "")
        .strip()
    )


def infer_python_version(config_dir: Path) -> str:
    if (config_dir / ".python-version").exists():
        with open(config_dir / ".python-version", "r") as f:
            content = f.read()
            if content.strip():
                py_version = content.strip()
                return py_version
    with open(config_dir / "pyproject.toml", "rb") as f:
        data = load_toml_file(f)
    return _get_min_py_version(data.get("project", {}).get("requires-python", "3.12"))


def build_dockerfile_content(
    python_version: str | None = None, port: int = 4501
) -> str:
    return f"""
FROM python:{python_version}-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY . /app/

ENV PATH=/root/.local/bin:$PATH

RUN uv sync --locked
RUN uv tool install llamactl

EXPOSE {port}

ENTRYPOINT [ "uv", "run", "llamactl", "serve", "--host", "0.0.0.0", "--port", "{port}" ]
"""
