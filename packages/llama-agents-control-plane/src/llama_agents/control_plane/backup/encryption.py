"""AES-256-GCM encryption/decryption with PBKDF2 key derivation.

Wire format: [16-byte salt][12-byte nonce][ciphertext + 16-byte GCM auth tag]
"""

import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

SALT_LENGTH = 16
NONCE_LENGTH = 12
KEY_LENGTH = 32  # AES-256
PBKDF2_ITERATIONS = 600_000


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from a password and salt using PBKDF2-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt(plaintext: bytes, password: str) -> bytes:
    """Encrypt plaintext with AES-256-GCM using a password.

    Returns wire format: [16-byte salt][12-byte nonce][ciphertext + 16-byte GCM tag]
    """
    salt = os.urandom(SALT_LENGTH)
    nonce = os.urandom(NONCE_LENGTH)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return salt + nonce + ciphertext


def decrypt(data: bytes, password: str) -> bytes:
    """Decrypt AES-256-GCM encrypted data using a password.

    Expects wire format: [16-byte salt][12-byte nonce][ciphertext + 16-byte GCM tag]

    Raises:
        cryptography.exceptions.InvalidTag: if password is wrong or data is tampered.
        ValueError: if data is too short to contain the header.
    """
    min_length = SALT_LENGTH + NONCE_LENGTH + 16  # at least the auth tag
    if len(data) < min_length:
        raise ValueError(
            f"Encrypted data too short: {len(data)} bytes, minimum {min_length}"
        )
    salt = data[:SALT_LENGTH]
    nonce = data[SALT_LENGTH : SALT_LENGTH + NONCE_LENGTH]
    ciphertext = data[SALT_LENGTH + NONCE_LENGTH :]
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)
