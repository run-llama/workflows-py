from __future__ import annotations


def redact_api_key(
    token: str | None,
    visible_prefix: int = 6,
    visible_suffix_long: int = 4,
    visible_suffix_short: int = 2,
    long_threshold: int = 10,
    mask: str = "****",
) -> str:
    """Redact an API key for display.

    Shows a prefix and suffix with a mask in the middle. If token is short,
    reduces the suffix length to keep at least two trailing characters visible.

    This mirrors the masking behavior used for profile names.
    """
    if not token:
        return "-"
    cleaned = token.replace(" ", "")
    if len(cleaned) <= 0:
        return "-"
    first = cleaned[:visible_prefix]
    last_len = (
        visible_suffix_long if len(cleaned) > long_threshold else visible_suffix_short
    )
    last = cleaned[-last_len:] if last_len > 0 else ""
    return f"{first}{mask}{last}"
