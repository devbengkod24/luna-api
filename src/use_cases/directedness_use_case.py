"""Device-directed speech detection: fuse facing + active-speaker + VAD.

This use case answers the production question "should the system respond to
this utterance?" once a VAD utterance has been segmented. It combines two
signals over the utterance window:

  - facing  : FE MediaPipe head-pose/gaze score (0..1) for the candidate face.
  - speaking: LR-ASD per-frame score (0..1) — does the candidate's lip motion
              correlate with the audio that was heard?

Both must clear their thresholds for the utterance to count as directed. The
gate is deliberately fail-open: if ASD is unavailable or no face was tracked,
the system must not go deaf — it treats the utterance as directed. Only when we
have positive evidence that the speaker is *not* facing or *not* the visible
talker do we drop the utterance.

Pure orchestration — depends on `ports` + `domain` only.
"""

import logging
from typing import List, Optional

from src.config import settings
from src.domain.entities.vision import DirectednessVerdict
from src.domain.image import Image
from src.ports.vision import IActiveSpeakerService

logger = logging.getLogger(__name__)


class DirectednessUseCase:
    """Fuses facing + active-speaker signals into a directed/not verdict."""

    def __init__(
        self,
        active_speaker: Optional[IActiveSpeakerService] = None,
        asd_score_threshold: Optional[float] = None,
        asd_presence_ratio: Optional[float] = None,
        facing_threshold: Optional[float] = None,
    ) -> None:
        self.active_speaker = active_speaker
        self.asd_score_threshold = (
            asd_score_threshold
            if asd_score_threshold is not None
            else settings.asd_score_threshold
        )
        self.asd_presence_ratio = (
            asd_presence_ratio
            if asd_presence_ratio is not None
            else settings.asd_presence_ratio
        )
        self.facing_threshold = (
            facing_threshold
            if facing_threshold is not None
            else settings.facing_threshold
        )

    async def evaluate(
        self,
        *,
        pcm16: bytes,
        sample_rate: int,
        candidate_crops: List[Image],
        facing_scores: List[float],
        candidate_track_id: Optional[int] = None,
    ) -> DirectednessVerdict:
        """Decide whether a segmented utterance is directed at the system.

        `candidate_crops` and `facing_scores` describe the chosen candidate
        track over the utterance window. Either may be empty (no usable face) —
        in which case the missing dimension fails open.
        """
        facing_score = (
            sum(facing_scores) / len(facing_scores) if facing_scores else 1.0
        )
        facing_ok = facing_score >= self.facing_threshold

        # --- Active-speaker (audio-visual) dimension ---
        # Fail open when ASD can't contribute: disabled/no model, or no crops to
        # score. The facing dimension still applies.
        if self.active_speaker is None or not candidate_crops:
            if not facing_ok:
                reason = "low_facing"
            elif self.active_speaker is None:
                reason = "directed"  # ASD disabled — facing-only gate
            else:
                reason = "fail_open_no_face"  # ASD on, but no usable crops
            return DirectednessVerdict(
                directed=facing_ok,
                facing_score=facing_score,
                candidate_track_id=candidate_track_id,
                reason=reason,
            )

        try:
            asd_scores = await self.active_speaker.score(
                candidate_crops, pcm16, sample_rate
            )
        except Exception:
            logger.exception("ASD scoring failed — failing open")
            return DirectednessVerdict(
                directed=facing_ok,
                facing_score=facing_score,
                candidate_track_id=candidate_track_id,
                reason="fail_open_asd_error" if facing_ok else "low_facing",
            )

        if not asd_scores:
            return DirectednessVerdict(
                directed=facing_ok,
                facing_score=facing_score,
                candidate_track_id=candidate_track_id,
                reason="fail_open_no_asd" if facing_ok else "low_facing",
            )

        mean_asd = sum(asd_scores) / len(asd_scores)
        active = sum(1 for s in asd_scores if s >= self.asd_score_threshold)
        active_ratio = active / len(asd_scores)
        asd_ok = active_ratio >= self.asd_presence_ratio

        directed = facing_ok and asd_ok
        if directed:
            reason = "directed"
        elif not facing_ok:
            reason = "low_facing"
        else:
            reason = "low_asd"

        return DirectednessVerdict(
            directed=directed,
            facing_score=facing_score,
            asd_score=mean_asd,
            asd_active_ratio=active_ratio,
            candidate_track_id=candidate_track_id,
            reason=reason,
        )
