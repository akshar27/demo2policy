"""End-to-end: hand observations (from video or synthesis) -> cleaned trajectory -> retargeted demo validated in the sim."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .data.demo import Demo
from .geometry.camera import Camera, pose_from_pitch
from .geometry.lifting import HandPose3D, lift_hand
from .perception.tracker import HandTracker, iter_video_frames
from .retarget.clean import CleanConfig, TrackRejected, clean_track
from .retarget.retarget import RetargetConfig, retarget
from .sim.env import PickPlaceEnv


def poses_from_observations(observations: list, cam: Camera) -> list[HandPose3D | None]:
    return [None if o is None else lift_hand(o, cam) for o in observations]


def poses_from_video(path: str | Path, tracker: HandTracker, cam: Camera | None = None, hfov_deg: float = 65.0,
                     cam_pitch_deg: float = 0.0):
    """Run the hand tracker over a video file. Returns (poses, camera). Intrinsics are assumed from `hfov_deg` unless a camera is given."""
    observations = []
    for t_ms, frame in iter_video_frames(path):
        if cam is None:
            cam = Camera.from_fov(frame.shape[1], frame.shape[0], hfov_deg, pose_from_pitch(cam_pitch_deg))
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


def demos_from_poses_multiscene(poses: list, env: PickPlaceEnv, n_scenes: int, source: str, seed: int = 0,
                                clean_cfg: CleanConfig | None = None, ret_cfg: RetargetConfig | None = None,
                                keep_failed: bool = False) -> tuple[list[Demo], dict]:
    """One human video -> many robot demos: retarget the same hand motion to `n_scenes` different cube positions.

    Each scene is a fresh (cube, target) layout drawn from the env's randomisation box; the similarity transform of the
    anchor alignment re-fits the human path to it, and the result is replayed in the sim. Only successful replays are kept
    (unless keep_failed). This is the data-multiplication step that makes a handful of videos enough to train on.
    """
    clean_cfg = clean_cfg or CleanConfig(rate_hz=1.0 / env.dt)
    traj = clean_track(poses, clean_cfg)
    rng = np.random.default_rng(seed)
    base = ret_cfg or RetargetConfig()
    cfg = env.cfg
    demos, stats = [], {"attempted": 0, "kept": 0, "rejected": {}, "replay_failed": 0}
    for _ in range(n_scenes):
        cube_xy = (float(rng.uniform(*cfg.cube_x)), float(rng.uniform(*cfg.cube_y)))
        rcfg = replace(base, align="anchor", cube_xy=cube_xy)
        stats["attempted"] += 1
        try:
            rt = retarget(traj, rcfg, cfg)
        except TrackRejected as e:
            stats["rejected"][e.reason] = stats["rejected"].get(e.reason, 0) + 1
            continue
        env.reset(seed=0, cube_xy=tuple(rt.init["cube_xy"]), cube_yaw=rt.init["cube_yaw"], target_xy=tuple(rt.init["target_xy"]))
        obs, info = [], None
        for k in range(len(rt.ee_target)):
            obs.append(env.obs())
            _, info = env.step(rt.ee_target[k], rt.gripper[k], rt.yaw[k])
        for _ in range(10):
            _, info = env.step(rt.ee_target[-1], 1.0, rt.yaw[-1])
        ok = bool(info.success)
        if not ok:
            stats["replay_failed"] += 1
            if not keep_failed:
                continue
        stats["kept"] += 1
        demos.append(Demo(obs=np.array(obs), ee_target=rt.ee_target, gripper=rt.gripper, yaw=rt.yaw, dt=env.dt,
                          init={**rt.init, "seed": -1}, source=source, success=ok, quality={**traj.quality, **rt.diagnostics}))
    return demos, stats


def process_video(path: str | Path, tracker: HandTracker, env: PickPlaceEnv, scenes: int = 20, seed: int = 0,
                  hfov_deg: float = 65.0, cam_pitch_deg: float = 0.0, ret_cfg: RetargetConfig | None = None,
                  clean_cfg: CleanConfig | None = None):
    """Video file -> (demos, report). The report always says what happened, including why a video was rejected."""
    poses, cam = poses_from_video(path, tracker, hfov_deg=hfov_deg, cam_pitch_deg=cam_pitch_deg)
    n_hand = sum(p is not None for p in poses)
    report = {"video": str(path), "frames": len(poses), "frames_with_hand": n_hand,
              "camera": {"w": cam.width, "h": cam.height, "hfov_deg": hfov_deg, "pitch_deg": cam_pitch_deg, "assumed_intrinsics": True}}
    try:
        demos, stats = demos_from_poses_multiscene(poses, env, scenes, f"video:{Path(path).name}", seed, clean_cfg, ret_cfg)
        report.update(status="ok", **stats)
        if demos:
            report["track_quality"] = demos[0].quality
    except TrackRejected as e:
        demos = []
        report.update(status="rejected", reason=e.reason, detail=str(e))
    return demos, report, poses
