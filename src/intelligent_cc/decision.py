from __future__ import annotations

from .models import AudioEvent, CaptionSuggestion, ReactionSignal


HIGH_IMPACT_LABELS = {
    "alarm",
    "explosion",
    "glass breaking",
    "gunshot",
    "honking",
    "scream",
    "siren",
}


class CaptionDecisionEngine:
    def __init__(
        self,
        audio_weight: float = 0.55,
        reaction_weight: float = 0.45,
        decision_threshold: float = 0.5,
        high_impact_audio_threshold: float = 0.45,
    ) -> None:
        self.audio_weight = audio_weight
        self.reaction_weight = reaction_weight
        self.decision_threshold = decision_threshold
        self.high_impact_audio_threshold = high_impact_audio_threshold

    def decide(
        self,
        event: AudioEvent,
        reaction: ReactionSignal,
        index: int,
    ) -> CaptionSuggestion | None:
        label_boost = 0.12 if event.label in HIGH_IMPACT_LABELS else 0.0
        score = min(
            1.0,
            self.audio_weight * event.confidence
            + self.reaction_weight * reaction.confidence
            + label_boost,
        )
        high_impact_audio = (
            event.label in HIGH_IMPACT_LABELS
            and event.confidence >= self.high_impact_audio_threshold
        )
        if score < self.decision_threshold and not high_impact_audio:
            return None

        reason = "audio+visual"
        if high_impact_audio and reaction.confidence < 0.2:
            reason = "high-impact-audio"

        return CaptionSuggestion(
            index=index,
            label=event.label,
            text=f"[{event.label}]",
            start=event.start,
            end=event.end,
            audio_confidence=event.confidence,
            reaction_confidence=reaction.confidence,
            decision_score=float(score),
            reason=reason,
        )
