from __future__ import annotations

import re
from pathlib import Path


class IndexHtmlError(RuntimeError):
    """Raised when index.html cannot be updated."""


SCRIPT_PATTERN = re.compile(
    r'<script\s+type="module"\s+crossorigin\s+src="[^"]*"[^>]*></script>'
)
CSS_PATTERN = re.compile(r'<link\s+rel="stylesheet"\s+crossorigin\s+href="[^"]*"[^>]*>')


def default_index_path() -> Path:
    """Return the default path to the debugger index.html file."""
    return (
        Path(__file__).resolve().parents[3]
        / "src"
        / "workflows"
        / "server"
        / "static"
        / "index.html"
    )


def update_index_html(
    js_url: str, css_url: str, index_path: str | Path | None = None
) -> None:
    """Replace the debugger asset URLs in index.html."""
    target = Path(index_path) if index_path is not None else default_index_path()
    if not target.exists():
        raise FileNotFoundError(f"index.html not found at {target}")

    content = target.read_text(encoding="utf-8")

    new_script = f'<script type="module" crossorigin src="{js_url}"></script>'
    updated_content, script_count = SCRIPT_PATTERN.subn(new_script, content)
    if script_count == 0:
        raise IndexHtmlError("Could not find script tag in index.html")

    new_css = f'<link rel="stylesheet" crossorigin href="{css_url}">'
    updated_content, css_count = CSS_PATTERN.subn(new_css, updated_content)
    if css_count == 0:
        raise IndexHtmlError("Could not find link tag in index.html")

    target.write_text(updated_content, encoding="utf-8")
