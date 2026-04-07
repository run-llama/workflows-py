"""Utilities for detecting and handling interactive CLI sessions."""

import functools
import os
import sys


@functools.cache
def is_interactive_session() -> bool:
    """
    Detect if the current CLI session is interactive.

    Returns True if the session is interactive (user can be prompted),
    False if it's non-interactive (e.g., CI/CD, scripted environment).

    This function checks multiple indicators:
    - Whether stdin/stdout are connected to a TTY
    - Explicit non-interactive environment variables

    Examples:
        >>> if is_interactive_session():
        ...     user_input = questionary.text("Enter value:").ask()
        ... else:
        ...     raise click.ClickException("Value required in non-interactive mode")
    """

    # Check if stdin and stdout are TTYs
    # This is the most reliable indicator for interactive sessions
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False

    # Additional check for TERM environment variable
    # Some environments set TERM=dumb for non-interactive sessions
    if os.environ.get("TERM") == "dumb":
        return False

    return True
