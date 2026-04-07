from pathlib import Path


def validate_path_traversal(
    path: Path, source_root: Path, path_type: str = "path"
) -> None:
    """Validates that a path is within the source root to prevent path traversal attacks.

    Args:
        path: The path to validate
        source_root: The root directory that paths should be relative to
        path_type: Description of the path type for error messages

    Raises:
        DeploymentError: If the path is outside the source root
    """
    resolved_path = (source_root / path).resolve()
    resolved_source_root = source_root.resolve()

    if not resolved_path.is_relative_to(resolved_source_root):
        msg = (
            f"{path_type} {path} is not a subdirectory of the source root {source_root}"
        )
        raise RuntimeError(msg)
