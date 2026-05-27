"""Identifier generation port (replaces inline uuid.uuid4 calls in use cases)."""

from abc import ABC, abstractmethod


class IUuidGenerator(ABC):
    """Generates string identifiers for new entities."""

    @abstractmethod
    def new_id(self) -> str:
        """Return a new globally-unique string identifier."""
        ...
