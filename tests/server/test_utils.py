# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.


from workflows.server.utils import nanoid


def test_nanoid_default_length() -> None:
    """Test nanoid with default length."""
    result = nanoid()
    assert len(result) == 10
    assert isinstance(result, str)


def test_nanoid_custom_length() -> None:
    """Test nanoid with custom length."""
    result = nanoid(3)
    assert len(result) == 3


def test_nanoid_zero_length() -> None:
    """Test nanoid with zero length."""
    result = nanoid(0)
    assert len(result) == 0
    assert result == ""


def test_nanoid_uniqueness() -> None:
    """Test that nanoid generates unique IDs."""
    # Generate multiple IDs and check for uniqueness
    ids = [nanoid() for _ in range(1000)]
    unique_ids = set(ids)

    # Should be very unlikely to have duplicates with 10-char alphanumeric
    # With 62^10 possible combinations, 1000 IDs should be unique
    assert len(ids) == len(unique_ids)


def test_nanoid_negative_length() -> None:
    """Test nanoid behavior with negative length."""
    # Python range() with negative values returns empty range
    result = nanoid(-1)
    assert result == ""

    result = nanoid(-10)
    assert result == ""
