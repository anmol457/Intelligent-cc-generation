from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import AudioEvent, ReactionSignal


class VisualReactionDetector:
    """Detects visible motion and face/head-position changes around an audio event."""

    def __init__(
        self,
        window_before: float = 0.75,
        window_after: float = 1.25,
        sample_fps: float = 6.0,
    ) -> None:
        self.window_before = window_before
        self.window_after = window_after
        self.sample_fps = sample_fps
        self._mediapipe_detector = None
        self._haar_detector = None

    def score_event(self, video_path: Path, event: AudioEvent) -> ReactionSignal:
        frames = self._sample_frames(
            video_path,
            max(event.start - self.window_before, 0.0),
            event.end + self.window_after,
        )
        if len(frames) < 2:
            return ReactionSignal(0.0, 0.0, 0.0, len(frames))

        motion_score = self._motion_score(frames)
        face_shift_score = self._face_shift_score(frames)
        confidence = min(1.0, 0.65 * motion_score + 0.35 * face_shift_score)
        return ReactionSignal(
            confidence=float(confidence),
            motion_score=float(motion_score),
            face_shift_score=float(face_shift_score),
            frame_count=len(frames),
        )

    def _sample_frames(self, video_path: Path, start: float, end: float) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(source_fps / self.sample_fps)), 1)
        start_frame = int(start * source_fps)
        end_frame = int(end * source_fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        frame_number = start_frame
        while frame_number <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_number - start_frame) % step == 0:
                frames.append(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA))
            frame_number += 1
        capture.release()
        return frames

    def _motion_score(self, frames: list[np.ndarray]) -> float:
        scores: list[float] = []
        previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for frame in frames[1:]:
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                current,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            scores.append(float(np.percentile(magnitude, 90)))
            previous = current
        if not scores:
            return 0.0
        return min(1.0, float(np.mean(scores)) / 3.5)

    def _face_shift_score(self, frames: list[np.ndarray]) -> float:
        centers: list[tuple[float, float]] = []
        for frame in frames:
            center = self._detect_face_center(frame)
            if center is not None:
                centers.append(center)
        if len(centers) < 2:
            return 0.0

        deltas = [
            abs(centers[index][0] - centers[index - 1][0])
            + abs(centers[index][1] - centers[index - 1][1])
            for index in range(1, len(centers))
        ]
        return min(1.0, float(np.percentile(deltas, 80)) * 3.0)

    def _detect_face_center(self, frame: np.ndarray) -> tuple[float, float] | None:
        mediapipe_center = self._detect_mediapipe_face_center(frame)
        if mediapipe_center is not None:
            return mediapipe_center
        return self._detect_haar_face_center(frame)

    def _detect_mediapipe_face_center(self, frame: np.ndarray) -> tuple[float, float] | None:
        try:
            import mediapipe as mp
        except ImportError:
            return None

        if self._mediapipe_detector is None:
            self._mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.45,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mediapipe_detector.process(rgb)
        if not result.detections:
            return None

        box = result.detections[0].location_data.relative_bounding_box
        return (float(box.xmin + box.width / 2.0), float(box.ymin + box.height / 2.0))

    def _detect_haar_face_center(self, frame: np.ndarray) -> tuple[float, float] | None:
        if self._haar_detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_detector = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        frame_height, frame_width = gray.shape[:2]
        return (
            float((x + width / 2.0) / frame_width),
            float((y + height / 2.0) / frame_height),
        )
