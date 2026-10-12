import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from demo2policy.data.expert_data import generate_expert_demos  # noqa: E402
from demo2policy.pipeline import process_video  # noqa: E402
from demo2policy.qa import write_qa  # noqa: E402
from demo2policy.sim.env import PickPlaceEnv  # noqa: E402
from demo2policy.synth.human import HumanStyle, default_camera, human_observations  # noqa: E402


class ScriptedTracker:
    """Stands in for MediaPipe: returns pre-generated observations frame by frame."""

    def __init__(self, observations):
        self.obs, self.i = observations, 0

    def track(self, frame, t_ms):
        o = self.obs[min(self.i, len(self.obs) - 1)]
        self.i += 1
        return o


def _write_blank_video(path, n, fps=30.0, size=(640, 480)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(n):
        w.write(np.zeros((size[1], size[0], 3), np.uint8))
    w.release()


def test_process_video_end_to_end_with_stub_tracker(tmp_path):
    env = PickPlaceEnv()
    d, _ = generate_expert_demos(env, 1, seed0=300)
    cam = default_camera(640, 480)
    obs = human_observations(d[0], cam, HumanStyle(seed=1, n_dropouts=1, n_glitches=0))
    p = tmp_path / "v.mp4"
    _write_blank_video(p, len(obs))
    # the stub emits landmarks for the camera the synthetic video was "shot" with; process_video assumes hfov 65 so use 70
    demos, report, poses = process_video(p, ScriptedTracker(obs), env, scenes=8, hfov_deg=70.0, cam_pitch_deg=42.7)
    assert report["status"] == "ok" and report["frames"] == len(obs)
    assert len(demos) >= 1 and all(x.success for x in demos)
    qa = write_qa(tmp_path / "qa", report, poses, demos)
    assert (qa / "report.json").exists() and (qa / "tracking.png").exists() and (qa / "retargeted.png").exists()


def test_process_video_reports_rejection_reason(tmp_path):
    env = PickPlaceEnv()
    p = tmp_path / "v.mp4"
    _write_blank_video(p, 40)
    demos, report, _ = process_video(p, ScriptedTracker([None]), env, scenes=2)
    assert demos == [] and report["status"] == "rejected" and report["reason"] == "no_hand"
