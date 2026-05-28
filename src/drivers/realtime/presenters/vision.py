"""Presenters for vision results — WS payload + delta diffing."""

from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any

from src.domain.entities.vision import VisionResult


def vision_result_to_dict(result: VisionResult) -> dict[str, Any]:
    return {
        "track_id": result.track_id,
        "name": result.name,
        "nim": result.nim,
        "emotion": result.emotion,
        "bbox": result.bbox.to_list(),
        "confidence": {
            "face": result.confidence.face,
            "emotion": result.confidence.emotion,
        },
        "face_id": result.face_id,
    }


@dataclass
class VisionDeltaCalculator:
    """Compute added/updated/removed tracks against the last emitted snapshot.

    A track is "updated" when any field of its serialized payload differs.
    """

    _last: dict[int, dict] = field(default_factory=dict)

    def diff(self, results: Iterable[VisionResult]) -> dict[str, Any]:
        current: dict[int, dict] = {}
        for r in results:
            if r.track_id is None:
                continue
            current[r.track_id] = vision_result_to_dict(r)

        added: list[dict] = []
        updated: list[dict] = []
        for tid, payload in current.items():
            previous = self._last.get(tid)
            if previous is None:
                added.append(payload)
            elif previous != payload:
                updated.append(payload)

        removed: list[int] = [tid for tid in self._last.keys() if tid not in current]

        self._last = current
        return {"added": added, "updated": updated, "removed": removed}

    def reset(self) -> None:
        self._last.clear()


def full_snapshot(results: Iterable[VisionResult]) -> list[dict]:
    """Backward-compatible full snapshot payload."""
    return [vision_result_to_dict(r) for r in results if r.track_id is not None]
