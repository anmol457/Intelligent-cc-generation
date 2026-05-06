from __future__ import annotations

from pathlib import Path

from .audio import AudioEventDetector
from .decision import CaptionDecisionEngine
from .models import PipelineResult
from .output import write_sls, write_srt
from .vision import VisualReactionDetector


class IntelligentCCPipeline:
    def __init__(
        self,
        audio_detector: AudioEventDetector | None = None,
        reaction_detector: VisualReactionDetector | None = None,
        decision_engine: CaptionDecisionEngine | None = None,
    ) -> None:
        self.audio_detector = audio_detector or AudioEventDetector()
        self.reaction_detector = reaction_detector or VisualReactionDetector()
        self.decision_engine = decision_engine or CaptionDecisionEngine()

    def run(self, video_path: Path, output_path: Path, output_format: str = "srt") -> PipelineResult:
        video_path = video_path.resolve()
        output_path = output_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        audio_events = self.audio_detector.detect(video_path)
        suggestions = []
        next_index = 1
        for event in audio_events:
            reaction = self.reaction_detector.score_event(video_path, event)
            suggestion = self.decision_engine.decide(event, reaction, next_index)
            if suggestion is not None:
                suggestions.append(suggestion)
                next_index += 1

        if output_format == "srt":
            write_srt(suggestions, output_path)
        elif output_format == "sls":
            write_sls(suggestions, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return PipelineResult(
            video_path=video_path,
            audio_events=audio_events,
            suggestions=suggestions,
            output_path=output_path,
        )
