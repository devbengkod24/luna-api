"""UUID4-based identifier generator (stdlib)."""

import uuid

from src.ports.identifiers import IUuidGenerator


class StdUuidGenerator(IUuidGenerator):
    def new_id(self) -> str:
        return str(uuid.uuid4())
