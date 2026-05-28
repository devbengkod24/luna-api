"""VAD-driven utterance segmenter.

State machine over per-frame is_speech() results:
  IDLE  --(N speech frames)--> SPEAKING  : emit SPEECH_START + prebuffered audio
  SPEAKING --(M silence frames)--> IDLE  : emit SPEECH_END with utterance audio
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum

from src.ports.vad import IVadService


class SegmentEvent(Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class SegmenterTick:
    event: SegmentEvent = SegmentEvent.NONE
    utterance: bytes | None = None  # filled on SPEECH_END


class VadSegmenter:
    """Stateful per-connection segmenter."""

    def __init__(
        self,
        vad: IVadService,
        speech_start_frames: int = 3,
        speech_end_frames: int = 25,
        prebuffer_frames: int = 7,
        max_utterance_frames: int = 1000,
    ) -> None:
        self.vad = vad
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames
        self.max_utterance_frames = max_utterance_frames

        self._prebuffer: deque[bytes] = deque(maxlen=prebuffer_frames)
        self._utterance: list[bytes] = []
        self._is_speaking = False
        self._speech_streak = 0
        self._silence_streak = 0
        # When the assistant is producing a reply (LLM thinking, TTS streaming),
        # we deliberately stop watching VAD. Barge-in is handled out-of-band
        # via a hotword listener on the client; raw VAD would false-trigger on
        # ambient noise / TTS speaker leak and cancel valid replies.
        self._is_responding = False

    @property
    def frame_bytes(self) -> int:
        return self.vad.frame_bytes

    def reset(self) -> None:
        self._prebuffer.clear()
        self._utterance.clear()
        self._is_speaking = False
        self._speech_streak = 0
        self._silence_streak = 0

    def set_responding(self, value: bool) -> None:
        """Toggle the assistant-responding gate.

        While True, push() short-circuits — no SPEECH_START / SPEECH_END will
        be emitted no matter what the client sends. Resets in-progress state
        so the next listening window starts clean.
        """
        if value and not self._is_responding:
            self._prebuffer.clear()
            self._utterance.clear()
            self._is_speaking = False
            self._speech_streak = 0
            self._silence_streak = 0
        self._is_responding = value

    def push(self, frame: bytes) -> SegmenterTick:
        """Feed one PCM-16 mono frame. Returns segmenter state transitions."""
        if self._is_responding:
            return SegmenterTick()
        if len(frame) != self.frame_bytes:
            return SegmenterTick()

        speech = self.vad.is_speech(frame)

        if not self._is_speaking:
            self._prebuffer.append(frame)
            if speech:
                self._speech_streak += 1
            else:
                self._speech_streak = 0
            if self._speech_streak >= self.speech_start_frames:
                self._is_speaking = True
                self._silence_streak = 0
                # Seed utterance with prebuffer so we don't clip the start.
                self._utterance = list(self._prebuffer)
                self._prebuffer.clear()
                return SegmenterTick(event=SegmentEvent.SPEECH_START)
            return SegmenterTick()

        # SPEAKING state
        self._utterance.append(frame)
        if speech:
            self._silence_streak = 0
        else:
            self._silence_streak += 1

        too_long = len(self._utterance) >= self.max_utterance_frames
        if self._silence_streak >= self.speech_end_frames or too_long:
            audio = b"".join(self._utterance)
            self._utterance.clear()
            self._is_speaking = False
            self._speech_streak = 0
            self._silence_streak = 0
            return SegmenterTick(event=SegmentEvent.SPEECH_END, utterance=audio)

        return SegmenterTick()
