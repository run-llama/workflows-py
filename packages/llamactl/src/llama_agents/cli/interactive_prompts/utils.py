"""Shared utilities for CLI operations"""

import questionary
from rich.console import Console

from .session_utils import is_interactive_session

console = Console()


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask for confirmation with a consistent interface.

    In non-interactive sessions, returns the default value without prompting.
    """
    if not is_interactive_session():
        return default

    return questionary.confirm(message, default=default).ask() or False
