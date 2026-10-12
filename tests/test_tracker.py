import numpy as np
import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")

from demo2policy.perception.tracker import MediaPipeHandTracker, iter_video_frames  # noqa: E402


@pytest.fixture(scope="module")
def tracker():
    try:
        return MediaPipeHandTracker()
    except Exception as e:  # model download needs network on first use
        pytest.skip(f"hand model unavailable: {e}")


def test_blank_frame_has_no_hand(tracker):
    assert tracker.track(np.zeros((240, 320, 3), np.uint8), 0) is None


def test_iter_video_frames_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        next(iter_video_frames(tmp_path / "nope.mp4"))


def test_iter_video_frames_roundtrip(tmp_path):
    import cv2
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    frames = list(iter_video_frames(p))
    assert len(frames) == 5 and frames[0][1].shape == (48, 64, 3)
    assert [t for t, _ in frames] == [0, 100, 200, 300, 400]
