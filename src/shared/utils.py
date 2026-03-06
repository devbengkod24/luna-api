"""Utility functions."""
import hashlib
import secrets
import string
from datetime import datetime, timezone
from typing import Optional

from src.shared.constants import API_KEY_PREFIX, API_KEY_LENGTH


def generate_api_key() -> str:
    """Generate a random API key."""
    random_part = secrets.token_urlsafe(API_KEY_LENGTH)
    return f"{API_KEY_PREFIX}{random_part}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def mask_api_key(api_key: str) -> str:
    """
    Mask an API key for display.
    
    Example: sk-luna-abcd...xyz1
    """
    if len(api_key) < len(API_KEY_PREFIX) + 8:
        return api_key
    
    prefix_end = len(API_KEY_PREFIX)
    visible_start = api_key[prefix_end:prefix_end + 4]
    visible_end = api_key[-4:]
    
    return f"{API_KEY_PREFIX}{visible_start}...{visible_end}"


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def to_iso8601(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 (RFC 3339) string, e.g. 2026-01-31T06:32:46Z."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def to_unix_timestamp(dt: Optional[datetime]) -> Optional[int]:
    """Convert datetime to Unix timestamp (seconds since epoch)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())