from __future__ import annotations

from pathlib import Path

import click

from .audio import AudioEventDetector
from .decision import CaptionDecisionEngine
from .pipeline import IntelligentCCPipeline
from .vision import VisualReactionDetector


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output caption path. Defaults to <video-name>.srt.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["srt", "sls"], case_sensitive=False),
    default="srt",
    show_default=True,
)
@click.option("--audio-threshold", default=0.25, show_default=True, help="YAMNet confidence cutoff.")
@click.option("--decision-threshold", default=0.5, show_default=True, help="CC acceptance cutoff.")
@click.option("--max-events", type=int, default=None, help="Optional cap for quick test runs.")
def main(
    video: Path,
    output: Path | None,
    output_format: str,
    audio_threshold: float,
    decision_threshold: float,
    max_events: int | None,
) -> None:
    """Generate context-aware non-speech CC suggestions for VIDEO."""

    if output is None:
        output = video.with_suffix(f".{output_format}")

    pipeline = IntelligentCCPipeline(
        audio_detector=AudioEventDetector(
            confidence_threshold=audio_threshold,
            max_events=max_events,
        ),
        reaction_detector=VisualReactionDetector(),
        decision_engine=CaptionDecisionEngine(decision_threshold=decision_threshold),
    )
    result = pipeline.run(video, output, output_format=output_format)

    click.echo(f"Detected audio events: {len(result.audio_events)}")
    click.echo(f"Accepted CC suggestions: {len(result.suggestions)}")
    click.echo(f"Wrote: {result.output_path}")
