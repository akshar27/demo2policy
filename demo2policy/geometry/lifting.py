"""Monocular 3D lifting of a hand from one frame.

A tracker (MediaPipe) gives, per frame, (a) 2D pixel landmarks and (b) metric hand-centred 3D landmarks whose axes are
aligned with the camera. (b) has the hand's shape/orientation/size but not its position; (a) has the position on the image
plane but not its depth. With (b) known up to a translation t and (a) known, the perspective constraints are *linear* in t:

    (X_i + t_x) / (Z_i + t_z) = x_i     (x_i = normalised image coordinate of landmark i)
 => t_x - x_i t_z = x_i Z_i - X_i       (and likewise for y)

so t is a 42x3 least-squares problem (PnP with the rotation already known). Depth accuracy then rests on the metric
scale of (b), which for a learned tracker is itself an estimate: that assumption is exactly what the synthetic tests
perturb (see tests/test_geometry.py and scripts/lifting_error.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perception.hand_model import INDEX_TIP, OPEN_HAND, THUMB_TIP, WRIST, HandObservation, hand_scale, pinch_ratio
from .camera import Camera

# Only the palm (wrist + the four finger-base knuckles) is rigid; fingers/thumb articulate, so they must not drive the
# orientation fit (fitting all 21 points biases the rotation by ~1 degree whenever the hand is not fully open).
PALM = [0, 5, 9, 13, 17]
_TEMPLATE = OPEN_HAND[PALM] - OPEN_HAND[PALM].mean(axis=0)


@dataclass
class HandPose3D:
    t: float
    wrist: np.ndarray        # (3,) wrist position, world/robot frame
    R: np.ndarray            # (3,3) hand-frame -> world rotation (columns: x thumb-side, y fingers, z palm normal)
    pinch: float             # thumb-index tip distance / hand size (0 = touching)
    scale: float             # measured hand size (m) - a diagnostic for the tracker's metric scale
    depth: float             # camera-frame depth of the wrist (m)
    pinch_pt: np.ndarray     # (3,) midpoint of thumb tip and index tip, world frame: where the object is actually held
    closing_axis: np.ndarray  # (3,) unit hand-x axis in the world frame (thumb side): the gripper's closing direction
    reproj_px: float         # RMS reprojection error of the fit (px): a per-frame confidence signal
    score: float = 1.0


def procrustes_rotation(points: np.ndarray, template: np.ndarray = _TEMPLATE) -> np.ndarray:
    """Rotation R minimising || points - template R^T || (orthogonal Procrustes / Kabsch), det(R)=+1."""
    P = points - points.mean(axis=0)
    H = template.T @ P                        # 3x3
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def lift_hand(obs: HandObservation, cam: Camera) -> HandPose3D:
    xy = cam.normalize(obs.image_px)                    # (21,2)
    p = obs.world                                       # (21,3) relative to an unknown centre
    A = np.zeros((42, 3))
    b = np.zeros(42)
    A[0::2, 0], A[0::2, 2], b[0::2] = 1.0, -xy[:, 0], xy[:, 0] * p[:, 2] - p[:, 0]
    A[1::2, 1], A[1::2, 2], b[1::2] = 1.0, -xy[:, 1], xy[:, 1] * p[:, 2] - p[:, 1]
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    pts_cam = p + t
    proj = cam.project(pts_cam)
    reproj = float(np.sqrt(np.mean(np.sum((proj - obs.image_px) ** 2, axis=1))))
    R_cam_hand = procrustes_rotation(p[PALM])
    R_world = cam.T_world_cam[:3, :3] @ R_cam_hand
    return HandPose3D(t=obs.t, wrist=cam.cam_to_world(pts_cam[WRIST]), R=R_world, pinch=pinch_ratio(p),
                      scale=hand_scale(p), depth=float(pts_cam[WRIST, 2]),
                      pinch_pt=cam.cam_to_world(0.5 * (pts_cam[THUMB_TIP] + pts_cam[INDEX_TIP])),
                      closing_axis=R_world[:, 0].copy(), reproj_px=reproj, score=obs.score)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
