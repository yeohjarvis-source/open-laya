from __future__ import annotations

import hashlib
import hmac
import secrets

API_KEY_PREFIX = "laya_live_"


def generate_api_key() -> tuple[str, str, str]:
    """Return the complete key, its public id, and its secret component."""
    public_id = secrets.token_urlsafe(9).replace("-", "").replace("_", "")
    secret = secrets.token_urlsafe(32)
    return f"{API_KEY_PREFIX}{public_id}.{secret}", public_id, secret


def parse_api_key(value: str) -> tuple[str, str] | None:
    if not value.startswith(API_KEY_PREFIX) or "." not in value:
        return None
    public_id, secret = value[len(API_KEY_PREFIX) :].split(".", 1)
    if not public_id or not secret:
        return None
    return public_id, secret


def hash_api_secret(secret: str, pepper: str) -> str:
    return hmac.new(pepper.encode(), secret.encode(), hashlib.sha256).hexdigest()


def secure_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode(), right.encode())

