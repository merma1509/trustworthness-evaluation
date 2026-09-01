"""sealing.py
Shared hashing + symmetric sealing primitives for the human-validation experiment
and the reproducible pipeline

The plan calls for GPG-sealed auto labels that can be opened ONLY after all human
annotations are finalised. Because ``gpg`` is not universally available on the
analyst machine, this module provides a drop-in, self-contained equivalent using
``cryptography.Fernet`` with a passphrase-derived key (PBKDF2-HMAC-SHA256)

Semantics preserved:
  * the payload is encrypted so a rater cannot read auto labels in advance,
  * decryption requires the passphrase held separately by the experiment owner,
  * the plaintext hash is committed so integrity can be verified after opening

All hashing is deterministic SHA-256 so the reproduced pipeline can verify that
no sealed artifact changed after it was locked
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, List

# PBKDF2 parameters — fixed for reproducibility (iteration count is explicit)
_ITERATIONS = 390_000
_SALT_BYTES = 16
_KEY_BYTES = 32


def _fernet_key_from_passphrase(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet (AES-128-CBC + HMAC) key from a passphrase + salt"""

    dk = hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, _ITERATIONS, dklen=_KEY_BYTES
    )
    return base64.urlsafe_b64encode(dk)


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of raw bytes"""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of a UTF-8 string."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes (streamed, memory-safe)"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonl(records: List[dict]) -> str:
    """Deterministic hash of a list of JSON records (canonical line serialisation)

    Each record is serialised with ``sort_keys=True`` and encoded as UTF-8 so the
    hash is independent of key insertion order / unicode handling.
    """
    h = hashlib.sha256()
    for rec in records:
        line = json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
        h.update(line)
        h.update(b"\n")
    return h.hexdigest()


def sha256_file_hex16(path: Path) -> str:
    """Return the 16-char shortened hash prefix used in manifests."""
    return sha256_file(path)[:16]


def encrypt_payload(payload: bytes, passphrase: str) -> bytes:
    """Encrypt bytes with a passphrase-derived Fernet key.

    Returns a binary blob whose header carries the salt so it is self-decryptable
    (the passphrase alone is sufficient to reopen it)
    """
    from cryptography.fernet import Fernet

    salt = os.urandom(_SALT_BYTES)
    key = _fernet_key_from_passphrase(passphrase, salt)
    token = Fernet(key).encrypt(payload)
    # Header: 8-byte magic + 4-byte salt length + salt, then the Fernet token.
    return b"SEAL1" + len(salt).to_bytes(4, "big") + salt + token


def decrypt_payload(blob: bytes, passphrase: str) -> bytes:
    """Decrypt a blob produced by :func:`encrypt_payload`.

    Raises:
        ValueError: on a malformed header.
        cryptography.fernet.InvalidToken: on a wrong passphrase / tampered blob
    """
    from cryptography.fernet import Fernet

    prefix = b"SEAL1"
    if not blob.startswith(prefix):
        raise ValueError("Not a SEAL1 payload (bad header)")
    offset = len(prefix)
    salt_len = int.from_bytes(blob[offset : offset + 4], "big")
    salt_start = offset + 4
    salt = blob[salt_start : salt_start + salt_len]
    token = blob[salt_start + salt_len :]
    key = _fernet_key_from_passphrase(passphrase, salt)
    return Fernet(key).decrypt(token)


def encrypt_json_to_file(
    data: Any, out_path: Path, passphrase: str, plaintext_sha: Optional[str] = None
) -> str:
    """Encrypt a JSON value to a sealed file; return the plaintext SHA-256

    The plaintext hash is computed independently and returned so the manifest can
    record it (to verify integrity after the seal is opened) without leaking the
    payload itself.
    """
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    digest = plaintext_sha or sha256_bytes(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encrypt_payload(payload, passphrase))
    return digest


def decrypt_json_file(in_path: Path, passphrase: str) -> Any:
    """Decrypt a sealed JSON file produced by :func:`encrypt_json_to_file`"""
    blob = in_path.read_bytes()
    return json.loads(decrypt_payload(blob, passphrase).decode("utf-8"))


def generate_passphrase() -> str:
    """Generate a random, high-entropy passphrase and return it.

    The returned value is meant to be written by the experiment owner to a
    keeper file (e.g. ``sealed_keeper.txt.enc``) or a password manager
    """
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
