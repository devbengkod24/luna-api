"""Photobooth photo entities."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StoredObject:
    """One object as the storage backend describes it, without its bytes."""
    path: str
    size_bytes: int
    modified_at: datetime


@dataclass(frozen=True)
class Photo:
    """A photobooth photo.

    No database row backs this. The object in storage IS the record: a photo
    has nothing worth querying beyond its size and when it was taken, and both
    already live on the object. A metadata table would add the one failure a
    face photo must not have — a row deleted while its object survives, so the
    dashboard reports it gone while the picture is still stored.
    """
    id: str
    size_bytes: int
    created_at: datetime
