"""Synthetic 'human video': turn an end-effector demo into the stream of hand observations a tracker would emit.

This is the ground-truth harness for the whole video pipeline: the expert's motion plays the human, a virtual camera
films it, and dropouts / jitter / glitches / scale error are injected. Whatever the pipeline recovers can be compared with
the known truth - before any real footage is involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..data.demo import Demo
from ..geometry.camera import Camera, look_at
from ..perception.hand_model import hand_shape, synth_observation
from ..sim.ik import down_rotation


@dataclass
class HumanStyle:
    fps: float = 30.0
    px_noise: float = 1.5
    world_noise: float = 0.003
    scale_err: float = 0.0           # tracker metric-scale error (+0.05 = hands look 5% bigger than they are)
    n_dropouts: int = 2
    dropout_s: tuple = (0.08, 0.25)
    n_glitches: int = 1              # frames with one wildly wrong landmark
    open_aperture: float = 0.9
    closed_aperture: float = 0.2
    world_rot: float = 0.35          # human world = robot world rotated/scaled/shifted (the camera pose is unknown to the pipeline)
    world_scale: float = 1.15
    world_shift: tuple = (-0.2, 0.15, 0.1)
    seed: int = 0


def default_camera(width: int = 640, height: int = 480) -> Camera:
    return Camera.from_fov(width, height, 70.0, look_at([1.1, 0.0, 0.7], [0.45, 0.0, 0.1]))


def human_observations(demo: Demo, cam: Camera, style: HumanStyle | None = None) -> list:
    """List of HandObservation | None (None = tracker lost the hand), at style.fps."""
    st = style or HumanStyle()
    rng = np.random.default_rng(st.seed)
    T = len(demo)
    t_src = np.arange(T) * demo.dt
    n = int(t_src[-1] * st.fps) + 1
    t = np.arange(n) / st.fps
    interp = lambda a: np.stack([np.interp(t, t_src, a[:, k]) for k in range(a.shape[1])], axis=1)  # noqa: E731
    pts = interp(demo.ee_target)
    grip = gaussian_filter1d(np.interp(t, t_src, demo.gripper), 0.12 * st.fps, mode="nearest")
    yaw = np.interp(t, t_src, demo.yaw)

    c, s = np.cos(st.world_rot), np.sin(st.world_rot)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = st.world_scale * pts @ Rz.T + np.array(st.world_shift)
    yaw = yaw + st.world_rot

    aperture = st.closed_aperture + grip * (st.open_aperture - st.closed_aperture)
    drop = np.zeros(n, bool)
    for _ in range(st.n_dropouts):
        a = rng.integers(2, max(3, n - 10))
        drop[a:a + int(rng.uniform(*st.dropout_s) * st.fps)] = True
    glitch = set(rng.integers(2, n - 2, st.n_glitches).tolist())

    out = []
    Rcw = cam.T_world_cam[:3, :3].T
    for i in range(n):
        if drop[i]:
            out.append(None)
            continue
        R_w = down_rotation(float(yaw[i]) - np.pi / 2)  # hand x-axis = gripper closing axis (see retarget.py)
        shape = hand_shape(aperture[i])
        pinch_local = 0.5 * (shape[4] + shape[8])
        wrist_w = pts[i] - R_w @ pinch_local
        obs = synth_observation(cam, Rcw @ R_w, cam.world_to_cam(wrist_w), aperture[i], t=float(t[i]),
                                px_noise=st.px_noise, world_noise=st.world_noise, rng=rng)
        obs.world = obs.world * (1.0 + st.scale_err)
        if i in glitch:
            obs.image_px[rng.integers(0, 21)] += rng.choice([-1, 1]) * 150.0
        out.append(obs)
    return out
