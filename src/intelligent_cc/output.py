from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import CaptionSuggestion


def write_srt(suggestions: list[CaptionSuggestion], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for suggestion in suggestions:
        lines.extend(
            [
                str(suggestion.index),
                f"{format_timestamp(suggestion.start)} --> {format_timestamp(suggestion.end)}",
                suggestion.text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_sls(suggestions: list[CaptionSuggestion], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(suggestion) for suggestion in suggestions]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
