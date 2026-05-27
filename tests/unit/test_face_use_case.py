"""Unit tests for face registration use case."""

import base64

import pytest

from src.domain.entities.face import RegisterFaceInput
from src.domain.entities.vision import BoundingBox, DetectedFace
from src.use_cases.face_use_case import RegisterFacesUseCase

from .fakes import FakeFaceDetector, FakeFaceService, FakeUuidGenerator


def _b64(payload: bytes = b"image") -> str:
    return base64.b64encode(payload).decode()


def _detection(embedding: list[float]) -> DetectedFace:
    return DetectedFace(
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        embedding=embedding,
        confidence=0.99,
    )


@pytest.mark.asyncio
async def test_register_faces_stores_each_detection():
    detector = FakeFaceDetector(detections=[_detection([0.1] * 512)])
    face_service = FakeFaceService()
    use_case = RegisterFacesUseCase(
        face_detector=detector,
        face_service=face_service,
        uuid_generator=FakeUuidGenerator(ids=["fixed-id"]),
    )

    results = await use_case.execute(
        faces=[RegisterFaceInput(name="Alice", image_base64=_b64(), nim="123")]
    )

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].face_id == "fixed-id"
    assert face_service._by_id["fixed-id"].name == "Alice"


@pytest.mark.asyncio
async def test_register_faces_skips_existing_when_skip_existing_true():
    detector = FakeFaceDetector(detections=[_detection([0.1] * 512)])
    face_service = FakeFaceService()
    use_case = RegisterFacesUseCase(
        face_detector=detector,
        face_service=face_service,
        uuid_generator=FakeUuidGenerator(ids=["id-1", "id-2"]),
    )

    first = await use_case.execute(
        faces=[RegisterFaceInput(name="Alice", image_base64=_b64(), nim="123")]
    )
    second = await use_case.execute(
        faces=[RegisterFaceInput(name="Alice", image_base64=_b64(), nim="123")],
        skip_existing=True,
    )

    assert first[0].success is True
    assert second[0].success is False
    assert "already registered" in (second[0].message or "")


@pytest.mark.asyncio
async def test_register_faces_rejects_invalid_base64():
    detector = FakeFaceDetector(detections=[_detection([0.1] * 512)])
    use_case = RegisterFacesUseCase(
        face_detector=detector,
        face_service=FakeFaceService(),
        uuid_generator=FakeUuidGenerator(),
    )

    [result] = await use_case.execute(
        faces=[RegisterFaceInput(name="Alice", image_base64="@@not-base64@@")]
    )

    assert result.success is False
    assert "Invalid base64" in (result.message or "")


@pytest.mark.asyncio
async def test_register_faces_rejects_when_no_face_detected():
    detector = FakeFaceDetector(detections=[])
    use_case = RegisterFacesUseCase(
        face_detector=detector,
        face_service=FakeFaceService(),
        uuid_generator=FakeUuidGenerator(),
    )

    [result] = await use_case.execute(
        faces=[RegisterFaceInput(name="Alice", image_base64=_b64())]
    )

    assert result.success is False
    assert "No face detected" in (result.message or "")
