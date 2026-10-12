"""Pinhole camera (OpenCV convention: x right, y down, z forward) and its pose in a world/robot frame."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    T_world_cam: np.ndarray = field(default_factory=lambda: np.eye(4))  # camera pose in the world/robot frame

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 65.0, T_world_cam: np.ndarray | None = None) -> "Camera":
        """Assumed intrinsics from a horizontal field of view (typical phone main camera ~65-70 deg). Square pixels."""
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return cls(width, height, f, f, width / 2, height / 2, np.eye(4) if T_world_cam is None else T_world_cam)

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """(...,3) points in the camera frame -> (...,2) pixel coordinates."""
        p = np.asarray(p_cam, dtype=float)
        return np.stack([self.fx * p[..., 0] / p[..., 2] + self.cx, self.fy * p[..., 1] / p[..., 2] + self.cy], axis=-1)

    def normalize(self, px: np.ndarray) -> np.ndarray:
        """(...,2) pixels -> (...,2) normalised image-plane coordinates (x/z, y/z)."""
        px = np.asarray(px, dtype=float)
        return np.stack([(px[..., 0] - self.cx) / self.fx, (px[..., 1] - self.cy) / self.fy], axis=-1)

    def cam_to_world(self, p_cam: np.ndarray) -> np.ndarray:
        p = np.asarray(p_cam, dtype=float)
        return p @ self.T_world_cam[:3, :3].T + self.T_world_cam[:3, 3]

    def world_to_cam(self, p_world: np.ndarray) -> np.ndarray:
        R, t = self.T_world_cam[:3, :3], self.T_world_cam[:3, 3]
        return (np.asarray(p_world, dtype=float) - t) @ R


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera pose (4x4, camera->world) at `eye` looking at `target`, OpenCV axes (z forward, y down)."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def pose_from_pitch(pitch_deg: float, height: float = 0.5) -> np.ndarray:
    """Camera pose (camera->world) for a camera at `height` m looking forward and tilted down by `pitch_deg`, no roll.

    World: x = camera's horizontal forward direction, y = left, z = up. The anchor alignment in retargeting only rotates
    about the world z axis, so this frame must have z vertical: a camera whose tilt is unknown/ignored breaks that
    assumption, which is why the pitch is an explicit input.
    """
    p = np.radians(pitch_deg)
    x = np.array([0.0, -1.0, 0.0])
    z = np.array([np.cos(p), 0.0, -np.sin(p)])
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = [0.0, 0.0, height]
    return T
