"""End-to-end: hand observations (from video or synthesis) -> cleaned trajectory -> retargeted demo validated in the sim."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data.demo import Demo
from .geometry.camera import Camera
from .geometry.lifting import HandPose3D, lift_hand
from .perception.tracker import HandTracker, iter_video_frames
from .retarget.clean import CleanConfig, TrackRejected, clean_track
from .retarget.retarget import RetargetConfig, retarget
from .sim.env import PickPlaceEnv


def poses_from_observations(observations: list, cam: Camera) -> list[HandPose3D | None]:
    return [None if o is None else lift_hand(o, cam) for o in observations]


def poses_from_video(path: str | Path, tracker: HandTracker, cam: Camera | None = None, hfov_deg: float = 65.0):
    """Run the hand tracker over a video file. Returns (poses, camera). Intrinsics are assumed from `hfov_deg` unless a camera is given."""
    poses, observations = [], []
    for t_ms, frame in iter_video_frames(path):
        if cam is None:
            cam = Camera.from_fov(frame.shape[1], frame.shape[0], hfov_deg)
        observations.append(tracker.track(frame, t_ms))
    if cam is None:
        raise ValueError("video has no frames")
    return poses_from_observations(observations, cam), cam


def demo_from_poses(poses: list, env: PickPlaceEnv, source: str, clean_cfg: CleanConfig | None = None,
                    ret_cfg: RetargetConfig | None = None) -> Demo:
    """Clean + retarget a pose sequence, then *replay it in the sim* to record observations and the real outcome.

    Raises TrackRejected (with a reason) when the track cannot become a demo. A demo that replays unsuccessfully is still
    returned (success=False) so the failure is visible in the dataset statistics rather than silently dropped.
    """
    clean_cfg = clean_cfg or CleanConfig(rate_hz=1.0 / env.dt)
    traj = clean_track(poses, clean_cfg)
    rt = retarget(traj, ret_cfg, env.cfg)
    env.reset(seed=0, cube_xy=tuple(rt.init["cube_xy"]), cube_yaw=rt.init["cube_yaw"], target_xy=tuple(rt.init["target_xy"]))
    obs, info = [], None
    for k in range(len(rt.ee_target)):
        obs.append(env.obs())
        _, info = env.step(rt.ee_target[k], rt.gripper[k], rt.yaw[k])
    for _ in range(10):  # settle for the success check
        _, info = env.step(rt.ee_target[-1], 1.0, rt.yaw[-1])
    return Demo(obs=np.array(obs), ee_target=rt.ee_target, gripper=rt.gripper, yaw=rt.yaw, dt=env.dt,
                init={**rt.init, "seed": -1}, source=source, success=bool(info.success),
                quality={**traj.quality, **rt.diagnostics})
