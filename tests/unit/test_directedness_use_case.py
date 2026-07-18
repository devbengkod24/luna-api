"""Unit tests for the device-directed speech fusion use case."""

from typing import List

import pytest

from src.use_cases.directedness_use_case import DirectednessUseCase


class FakeActiveSpeaker:
    """Returns a fixed per-frame ASD score list regardless of input."""

    def __init__(self, scores: List[float]):
        self._scores = scores
        self.calls = 0

    async def score(self, face_crops, pcm16, sample_rate, fps=None):
        self.calls += 1
        return list(self._scores)


def _crops(n: int = 5):
    # The use case only checks truthiness/length of crops; contents are opaque.
    return [object() for _ in range(n)]


def _uc(active_speaker=None):
    return DirectednessUseCase(
        active_speaker=active_speaker,
        asd_score_threshold=0.4,
        asd_presence_ratio=0.35,
        facing_threshold=0.5,
    )


@pytest.mark.asyncio
async def test_directed_when_facing_and_speaking():
    asd = FakeActiveSpeaker([0.9, 0.8, 0.7, 0.85, 0.95])
    verdict = await _uc(asd).evaluate(
        pcm16=b"\x00" * 320,
        sample_rate=16000,
        candidate_crops=_crops(),
        facing_scores=[0.8, 0.9, 0.7],
        candidate_track_id=3,
    )
    assert verdict.directed is True
    assert verdict.reason == "directed"
    assert verdict.candidate_track_id == 3
    assert asd.calls == 1


@pytest.mark.asyncio
async def test_not_directed_when_not_facing():
    asd = FakeActiveSpeaker([0.9, 0.9, 0.9])
    verdict = await _uc(asd).evaluate(
        pcm16=b"\x00" * 320,
        sample_rate=16000,
        candidate_crops=_crops(),
        facing_scores=[0.1, 0.2, 0.15],
    )
    assert verdict.directed is False
    assert verdict.reason == "low_facing"


@pytest.mark.asyncio
async def test_not_directed_when_facing_but_silent_lips():
    # Bystander case: person faces the kiosk but isn't the one talking — ASD low.
    asd = FakeActiveSpeaker([0.05, 0.1, 0.0, 0.2, 0.05])
    verdict = await _uc(asd).evaluate(
        pcm16=b"\x00" * 320,
        sample_rate=16000,
        candidate_crops=_crops(),
        facing_scores=[0.9, 0.85, 0.95],
    )
    assert verdict.directed is False
    assert verdict.reason == "low_asd"


@pytest.mark.asyncio
async def test_fail_open_when_asd_disabled():
    # No ASD service wired — facing-only gate, fail-open on the AV dimension.
    verdict = await _uc(None).evaluate(
        pcm16=b"\x00" * 320,
        sample_rate=16000,
        candidate_crops=_crops(),
        facing_scores=[0.9, 0.9],
    )
    assert verdict.directed is True
    assert verdict.reason == "directed"


@pytest.mark.asyncio
async def test_fail_open_when_no_face_crops():
    asd = FakeActiveSpeaker([0.9, 0.9])
    verdict = await _uc(asd).evaluate(
        pcm16=b"\x00" * 320,
        sample_rate=16000,
        candidate_crops=[],
        facing_scores=[],  # no face → no facing samples either
    )
    # No crops to score and no facing evidence → fail open (don't go deaf).
    assert verdict.directed is True
    assert verdict.reason == "fail_open_no_face"
    assert asd.calls == 0


@pytest.mark.asyncio
async def test_fail_open_when_asd_errors():
    class Boom:
        async def score(self, *a, **k):
            raise RuntimeError("model exploded")

    verdict = await _uc(Boom()).evaluate(
        pcm16=b"\x00" * 320,
        sample_rate=16000,
        candidate_crops=_crops(),
        facing_scores=[0.9],
    )
    assert verdict.directed is True
    assert verdict.reason == "fail_open_asd_error"
