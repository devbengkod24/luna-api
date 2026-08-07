"""VAD-driven utterance segmenter.

State machine over per-frame is_speech() results:
  IDLE  --(N speech frames)--> SPEAKING  : emit SPEECH_START + prebuffered audio
  SPEAKING --(M silence frames)--> IDLE  : emit SPEECH_END with utterance audio
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import ceil

from src.ports.vad import IVadService


class SegmentEvent(Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    # Sustained speech detected while the assistant is talking. Advisory only —
    # the caller fuses it with facing before calling accept_barge_in().
    BARGE_IN = "barge_in"


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
        barge_in_window_frames: int = 0,
        barge_in_ratio: float = 0.6,
        barge_in_settle_frames: int = 0,
    ) -> None:
        self.vad = vad
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames
        self.max_utterance_frames = max_utterance_frames
        # Barge-in fires when at least `barge_in_ratio` of the last
        # `barge_in_window_frames` were speech. A sliding ratio rather than a
        # consecutive streak: human speech has micro-gaps between words, and an
        # echo canceller in double-talk attenuates the near-end talker in bursts,
        # so an unbroken run almost never happened and barge-in mostly failed.
        # Still far stricter than speech_start_frames (~90 ms), because TTS leak
        # is intermittent while a person talking over the reply is dense.
        # 0 disables barge-in detection entirely (hotword-only).
        self.barge_in_window_frames = barge_in_window_frames
        self._barge_hits_needed = max(
            1, ceil(barge_in_window_frames * barge_in_ratio)
        ) if barge_in_window_frames else 0
        # Frames discarded right after a barge-in is honoured. The speaker does
        # not fall silent the instant we decide to stop — the client still has to
        # be told, and the browser's echo canceller needs a moment to re-converge
        # once the far-end signal disappears. Anything captured in that gap is
        # part assistant, and splicing it onto the front of the user's utterance
        # is what wrecked the transcript.
        self.barge_in_settle_frames = barge_in_settle_frames
        self._settle_left = 0

        self._prebuffer: deque[bytes] = deque(maxlen=prebuffer_frames)
        self._utterance: list[bytes] = []
        self._is_speaking = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._is_responding = False
        # Rolling speech/silence verdicts heard while the assistant is talking.
        self._barge_window: deque[bool] = deque(
            maxlen=max(barge_in_window_frames, 1)
        )
        # Highest hit count seen since the current reply started, so the router
        # can log how close a failed barge-in got. Turns "barge-in is hard" into
        # a number to tune against.
        self._barge_peak = 0
        # Barge-in only makes sense once the speaker is actually producing
        # sound. Between SPEECH_END and the first audio chunk the assistant is
        # silently thinking (directedness → STT → LLM), and any sound the mic
        # picks up then is the user's own trailing speech or room noise —
        # letting that cancel the reply restarts the pipeline over and over and
        # the answer never arrives. Armed by arm_barge_in() on first audio.
        self._barge_armed = False

    @property
    def frame_bytes(self) -> int:
        return self.vad.frame_bytes

    def reset(self) -> None:
        self._prebuffer.clear()
        self._utterance.clear()
        self._is_speaking = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._barge_window.clear()

    @property
    def is_responding(self) -> bool:
        return self._is_responding

    @property
    def barge_in_stats(self) -> tuple[int, int, bool]:
        """(peak speech frames in a window, frames needed, was it ever armed).

        Diagnostic for tuning: a peak that sits just under the threshold every
        time means the ratio is too high (or the echo canceller is eating the
        user); a peak near zero means the mic is not hearing them at all.
        """
        return self._barge_peak, self._barge_hits_needed, self._barge_armed

    def set_responding(self, value: bool) -> None:
        """Toggle the assistant-responding gate.

        While True, push() emits no SPEECH_START / SPEECH_END; the only thing
        it can raise is BARGE_IN (and only when a barge-in window is set). Resets
        in-progress state so the next listening window starts clean.
        """
        if value and not self._is_responding:
            self._prebuffer.clear()
            self._utterance.clear()
            self._is_speaking = False
            self._speech_streak = 0
            self._silence_streak = 0
            self._barge_window.clear()
            self._barge_peak = 0
            self._barge_armed = False
            self._settle_left = 0
        self._is_responding = value

    def arm_barge_in(self) -> None:
        """Start watching for barge-in — call when the reply's audio starts.

        Deliberately NOT armed for the whole responding window: during the
        silent think phase there is nothing to talk over, so a trigger there is
        always a false one.
        """
        self._barge_armed = True

    def accept_barge_in(self) -> None:
        """Honour a BARGE_IN: drop everything heard so far and go back to
        listening from scratch.

        The caller decides (BARGE_IN is advisory — it is fused with the facing
        signal first).

        The frames buffered during detection are deliberately THROWN AWAY. They
        were all recorded while the speaker was playing, so they carry the
        assistant's voice as well as the user's; seeding the utterance with them
        put ~600 ms of mixed audio at the very start, which is exactly where
        Whisper anchors — the transcript came back garbage. Losing the user's
        opening words is the better trade: what survives is clean and
        contiguous. Normal VAD (with its own prebuffer) picks the sentence up
        again as the user keeps talking.
        """
        self._is_responding = False
        self._barge_armed = False
        self._settle_left = self.barge_in_settle_frames
        self.reset()

    def push(self, frame: bytes) -> SegmenterTick:
        """Feed one PCM-16 mono frame. Returns segmenter state transitions."""
        if len(frame) != self.frame_bytes:
            return SegmenterTick()

        # Post-barge-in settle: the speaker may still be trailing off and the
        # echo canceller is re-converging. Drop, don't buffer.
        if self._settle_left > 0:
            self._settle_left -= 1
            return SegmenterTick()

        if self._is_responding:
            if not self.barge_in_window_frames or not self._barge_armed:
                return SegmenterTick()
            self._barge_window.append(self.vad.is_speech(frame))
            if len(self._barge_window) < self.barge_in_window_frames:
                return SegmenterTick()
            hits = sum(self._barge_window)
            if hits > self._barge_peak:
                self._barge_peak = hits
            if hits >= self._barge_hits_needed:
                # Clear rather than keep sliding: if the caller rejects this one
                # (not facing the kiosk), a whole fresh window is needed before
                # we ask again, instead of re-firing on every later frame.
                self._barge_window.clear()
                return SegmenterTick(event=SegmentEvent.BARGE_IN)
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
