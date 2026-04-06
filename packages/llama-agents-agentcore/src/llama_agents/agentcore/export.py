from pathlib import Path

agentcore_dir = Path(".agentcore")


def export_generated_entrypoint_code() -> None:
    entrypoint = Path(__file__).parent / "entrypoint.py"
    content = entrypoint.read_text()
    agentcore_dir.mkdir(exist_ok=True)
    with open(agentcore_dir / "entrypoint.py", "w") as f:
        f.write(content)
    return None
