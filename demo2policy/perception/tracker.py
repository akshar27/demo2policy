"""Hand trackers behind one interface, so the geometry/retargeting code never sees MediaPipe."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .hand_model import HandObservation

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")
MODEL_PATH = Path(__file__).parent / "assets" / "hand_landmarker.task"


class HandTracker(Protocol):
    def track(self, frame_rgb: np.ndarray, t_ms: int) -> HandObservation | None: ...


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        import ssl

        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:  # python.org builds on macOS ship without system certs; certifi fixes that
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(path, "wb") as f:  # noqa: S310 - fixed https URL
            f.write(r.read())
    return path


class MediaPipeHandTracker:
    """MediaPipe Hand Landmarker (VIDEO mode, one hand). Returns pixel landmarks + metric world landmarks.

    ASSUMPTION (verified against real footage in milestone 6, not here): MediaPipe's world landmarks are metric,
    hand-centred, and share the image's axis convention (x right, y down, z away from the camera).
    """

    def __init__(self, model_path: Path | None = None, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path or MODEL_PATH))),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence)
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def track(self, frame_rgb: np.ndarray, t_ms: int) -> HandObservation | None:
        h, w = frame_rgb.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self._lm.detect_for_video(img, int(t_ms))
        if not res.hand_landmarks:
            return None
        px = np.array([[l.x * w, l.y * h] for l in res.hand_landmarks[0]])
        world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]])
        score = float(res.handedness[0][0].score) if res.handedness else 1.0
        return HandObservation(t=t_ms / 1000.0, image_px=px, world=world, score=score)


def iter_video_frames(path: str | Path, max_side: int = 960) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (timestamp_ms, RGB frame) from a video file, downscaled so the long side is <= max_side."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                bgr = cv2.resize(bgr, (round(w * s), round(h * s)))
            yield int(round(1000.0 * i / fps)), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
