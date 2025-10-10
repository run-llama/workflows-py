# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import secrets
import string

alphabet = string.ascii_letters + string.digits  # A-Z, a-z, 0-9


def nanoid(size: int = 10) -> str:
    """Returns a unique identifier with the format 'kY2xP9hTnQ'."""
    return "".join(secrets.choice(alphabet) for _ in range(size))
