"""Tests for encryption round-trip."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag
from llama_agents.control_plane.backup.encryption import (
    NONCE_LENGTH,
    SALT_LENGTH,
    decrypt,
    encrypt,
)


def test_round_trip() -> None:
    plaintext = b"hello world"
    password = "test-password"
    ciphertext = encrypt(plaintext, password)
    assert decrypt(ciphertext, password) == plaintext


def test_tampered_ciphertext_raises_invalid_tag() -> None:
    ciphertext = encrypt(b"some data", "pw")
    corrupted = bytearray(ciphertext)
    corrupted[-1] ^= 0xFF
    with pytest.raises(InvalidTag):
        decrypt(bytes(corrupted), "pw")


def test_wrong_password_raises_invalid_tag() -> None:
    ciphertext = encrypt(b"secret", "correct-password")
    with pytest.raises(InvalidTag):
        decrypt(ciphertext, "wrong-password")


def test_empty_plaintext_round_trip() -> None:
    ciphertext = encrypt(b"", "pw")
    assert decrypt(ciphertext, "pw") == b""


def test_short_data_raises_value_error() -> None:
    min_length = SALT_LENGTH + NONCE_LENGTH + 16
    too_short = b"\x00" * (min_length - 1)
    with pytest.raises(ValueError, match="too short"):
        decrypt(too_short, "pw")
