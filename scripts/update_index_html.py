#!/usr/bin/env python3
"""Update debugger asset URLs in index.html."""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


def update_index_html(
    js_url: str, css_url: str, index_path: Optional[Path] = None
) -> None:
    """Update the script and CSS URLs in index.html.

    Args:
        js_url: New JavaScript file URL
        css_url: New CSS file URL
        index_path: Path to index.html (defaults to src/workflows/server/static/index.html)

    Raises:
        FileNotFoundError: If index.html doesn't exist
        ValueError: If required elements not found in HTML
    """
    if index_path is None:
        index_path = (
            Path(__file__).parent.parent
            / "src"
            / "workflows"
            / "server"
            / "static"
            / "index.html"
        )

    if not index_path.exists():
        raise FileNotFoundError(f"index.html not found at {index_path}")

    # Read the file
    content = index_path.read_text()

    # Update script src
    script_pattern = (
        r'<script\s+type="module"\s+crossorigin\s+src="[^"]*"[^>]*></script>'
    )
    new_script = f'<script type="module" crossorigin src="{js_url}"></script>'

    updated_content, script_count = re.subn(script_pattern, new_script, content)
    if script_count == 0:
        raise ValueError("Could not find script tag in index.html")

    # Update CSS href
    css_pattern = r'<link\s+rel="stylesheet"\s+crossorigin\s+href="[^"]*"[^>]*>'
    new_css = f'<link rel="stylesheet" crossorigin href="{css_url}">'

    updated_content, css_count = re.subn(css_pattern, new_css, updated_content)
    if css_count == 0:
        raise ValueError("Could not find link tag in index.html")

    # Write back
    index_path.write_text(updated_content)

    print("✅ Updated index.html:")
    print(f"   JavaScript: {js_url}")
    print(f"   CSS: {css_url}")


def main() -> None:
    """Main function to update index.html from command line."""
    parser = argparse.ArgumentParser(
        description="Update debugger asset URLs in index.html"
    )
    parser.add_argument("--js-url", required=True, help="URL for the JavaScript file")
    parser.add_argument("--css-url", required=True, help="URL for the CSS file")
    parser.add_argument(
        "--index-path",
        type=Path,
        help="Path to index.html (optional, defaults to src/workflows/server/static/index.html)",
    )

    args = parser.parse_args()

    try:
        update_index_html(args.js_url, args.css_url, args.index_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
