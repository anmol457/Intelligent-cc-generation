from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import ffmpeg
import imageio_ffmpeg
import librosa
import numpy as np

from .models import AudioEvent


YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000
YAMNET_PATCH_SECONDS = 0.48
YAMNET_HOP_SECONDS = 0.48

NOISE_LABEL_HINTS = {
    "alarm",
    "applause",
    "bang",
    "bell",
    "breaking",
    "cheering",
    "clap",
    "crash",
    "cry",
    "door",
    "explosion",
    "fire",
    "glass",
    "gunshot",
    "horn",
    "laughter",
    "music",
    "scream",
    "shatter",
    "siren",
    "thunder",
    "vehicle",
}

LABEL_NORMALIZATION = {
    "air horn, truck horn": "honking",
    "car horn, truck horn": "honking",
    "horn": "honking",
    "glass": "glass breaking",
    "glass shatter": "glass breaking",
    "gunshot, gunfire": "gunshot",
    "music": "music",
    "applause": "applause",
    "cheering": "crowd cheering",
    "siren": "siren",
    "alarm": "alarm",
}


class AudioEventDetector:
    """YAMNet-based non-speech event detector."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        min_duration: float = 0.35,
        merge_gap: float = 0.6,
        max_events: int | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        self.max_events = max_events
        self._yamnet = None
        self._class_names: list[str] | None = None

    def detect(self, video_path: Path) -> list[AudioEvent]:
        with tempfile.TemporaryDirectory(prefix="intelligent_cc_") as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            self._extract_audio(video_path, wav_path)
            waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

        if waveform.size == 0:
            return []

        scores = self._score_waveform(waveform)
        raw_events = self._scores_to_events(scores)
        events = self._merge_events(raw_events)
        events.sort(key=lambda event: (event.start, -event.confidence))
        if self.max_events is not None:
            events = events[: self.max_events]
        return events

    def _extract_audio(self, video_path: Path, wav_path: Path) -> None:
        try:
            (
                ffmpeg.input(str(video_path))
                .output(str(wav_path), ac=1, ar=SAMPLE_RATE, format="wav", loglevel="error")
                .overwrite_output()
                .run(
                    cmd=imageio_ffmpeg.get_ffmpeg_exe(),
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
        except ffmpeg.Error as exc:
            detail = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
            raise RuntimeError(f"Unable to extract audio with ffmpeg: {detail}") from exc

    def _load_yamnet(self):
        if self._yamnet is None:
            import tensorflow_hub as hub

            self._yamnet = hub.load(YAMNET_URL)
            class_map_path = self._yamnet.class_map_path().numpy().decode("utf-8")
            with open(class_map_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self._class_names = [row["display_name"] for row in reader]
        return self._yamnet, self._class_names or []

    def _score_waveform(self, waveform: np.ndarray) -> np.ndarray:
        yamnet, _ = self._load_yamnet()
        scores, _, _ = yamnet(waveform.astype(np.float32))
        return scores.numpy()

    def _scores_to_events(self, scores: np.ndarray) -> list[AudioEvent]:
        _, class_names = self._load_yamnet()
        events: list[AudioEvent] = []
        for frame_index, frame_scores in enumerate(scores):
            top_indices = np.argsort(frame_scores)[-8:][::-1]
            start = frame_index * YAMNET_HOP_SECONDS
            end = start + YAMNET_PATCH_SECONDS
            for class_index in top_indices:
                label = class_names[int(class_index)]
                confidence = float(frame_scores[class_index])
                if confidence < self.confidence_threshold:
                    continue
                if not self._is_captionable_audio(label):
                    continue
                events.append(
                    AudioEvent(
                        label=self._normalize_label(label),
                        confidence=confidence,
                        start=start,
                        end=end,
                    )
                )
        return events

    def _merge_events(self, events: list[AudioEvent]) -> list[AudioEvent]:
        grouped: dict[str, list[AudioEvent]] = {}
        for event in events:
            grouped.setdefault(event.label, []).append(event)

        merged: list[AudioEvent] = []
        for label, label_events in grouped.items():
            label_events.sort(key=lambda event: event.start)
            current = label_events[0] if label_events else None
            for event in label_events[1:]:
                if current is None:
                    current = event
                    continue
                if event.start <= current.end + self.merge_gap:
                    duration_a = max(current.end - current.start, 0.01)
                    duration_b = max(event.end - event.start, 0.01)
                    weighted_conf = (
                        current.confidence * duration_a + event.confidence * duration_b
                    ) / (duration_a + duration_b)
                    current = AudioEvent(
                        label=label,
                        confidence=float(max(current.confidence, weighted_conf, event.confidence)),
                        start=current.start,
                        end=max(current.end, event.end),
                    )
                else:
                    if current.end - current.start >= self.min_duration:
                        merged.append(current)
                    current = event
            if current is not None and current.end - current.start >= self.min_duration:
                merged.append(current)
        return merged

    def _is_captionable_audio(self, label: str) -> bool:
        lowered = label.lower()
        if "speech" in lowered or "conversation" in lowered or "narration" in lowered:
            return False
        return any(hint in lowered for hint in NOISE_LABEL_HINTS)

    def _normalize_label(self, label: str) -> str:
        lowered = label.lower()
        for needle, replacement in LABEL_NORMALIZATION.items():
            if needle in lowered:
                return replacement
        cleaned = lowered.replace("_", " ").replace("/", " ")
        return " ".join(cleaned.split())
