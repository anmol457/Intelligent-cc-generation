from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioEvent:
    label: str
    confidence: float
    start: float
    end: float


@dataclass(frozen=True)
class ReactionSignal:
    confidence: float
    motion_score: float
    face_shift_score: float
    frame_count: int


@dataclass(frozen=True)
class CaptionSuggestion:
    index: int
    label: str
    text: str
    start: float
    end: float
    audio_confidence: float
    reaction_confidence: float
    decision_score: float
    reason: str


@dataclass(frozen=True)
class PipelineResult:
    video_path: Path
    audio_events: list[AudioEvent]
    suggestions: list[CaptionSuggestion]
    output_path: Path
