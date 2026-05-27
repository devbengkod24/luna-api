"""Domain image value object.

This is intentionally opaque: domain and use cases never inspect `.raw`.
Adapters wrap their native representation (e.g. numpy ndarray) into `Image`
when decoding, and unwrap when consuming.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Image:
    """BGR image passed between vision adapters."""

    raw: Any
    width: int
    height: int
