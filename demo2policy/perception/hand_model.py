"""Canonical 21-point hand (MediaPipe topology) and a synthetic hand generator with exact ground truth.

Hand frame: origin at the wrist, +y toward the fingertips, +x toward the thumb side, +z out of the palm. Units: metres.
The synthetic generator is how the geometry is tested without any video: pose + aperture in, the same quantities a real
tracker would emit out - with known truth to compare against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17

OPEN_HAND = np.array([
    [0.000, 0.000, 0], [0.025, 0.015, 0], [0.045, 0.040, 0], [0.060, 0.065, 0], [0.070, 0.085, 0],     # thumb
    [0.030, 0.090, 0], [0.032, 0.125, 0], [0.033, 0.148, 0], [0.034, 0.168, 0],                         # index
    [0.010, 0.095, 0], [0.011, 0.135, 0], [0.011, 0.160, 0], [0.011, 0.182, 0],                         # middle
    [-0.010, 0.090, 0], [-0.011, 0.126, 0], [-0.011, 0.150, 0], [-0.011, 0.170, 0],                     # ring
    [-0.028, 0.080, 0], [-0.030, 0.108, 0], [-0.030, 0.127, 0], [-0.030, 0.143, 0],                     # pinky
])
# Pinch: the thumb tip meets the index tip, and the thumb joints follow proportionally.
PINCH_THUMB = OPEN_HAND.copy()
PINCH_THUMB[1:5] = OPEN_HAND[1:5] + np.array([0.0, 0.0, 0.0])
PINCH_THUMB[4] = OPEN_HAND[INDEX_TIP] + np.array([-0.004, -0.003, 0.0])
PINCH_THUMB[3] = 0.5 * (OPEN_HAND[3] + PINCH_THUMB[4]) + np.array([0.0, 0.0, 0.0])
PINCH_THUMB[2] = 0.5 * (OPEN_HAND[2] + PINCH_THUMB[3])


def hand_shape(aperture: float) -> np.ndarray:
    """21x3 landmarks in the hand frame. aperture 1 = thumb/index fully apart, 0 = pinched."""
    a = float(np.clip(aperture, 0.0, 1.0))
    h = OPEN_HAND.copy()
    h[1:5] = PINCH_THUMB[1:5] + a * (OPEN_HAND[1:5] - PINCH_THUMB[1:5])
    return h


def hand_scale(lm: np.ndarray) -> float:
    """Wrist -> middle-finger MCP distance: a stable per-hand size reference."""
    return float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))


def pinch_ratio(lm: np.ndarray) -> float:
    """Thumb-tip to index-tip distance as a fraction of hand size (0 = touching)."""
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]) / hand_scale(lm))


@dataclass
class HandObservation:
    """What a hand tracker reports for one frame (matches MediaPipe's two landmark outputs)."""
    t: float
    image_px: np.ndarray    # (21,2) pixel coordinates
    world: np.ndarray       # (21,3) metric landmarks, hand-centred, axes aligned with the camera (OpenCV convention)
    score: float = 1.0


def synth_observation(camera, R_cam_hand: np.ndarray, wrist_cam: np.ndarray, aperture: float, t: float = 0.0,
                      px_noise: float = 0.0, world_noise: float = 0.0, rng: np.random.Generator | None = None) -> HandObservation:
    """Generate the observation a perfect tracker would give for a hand with the given camera-frame pose, plus noise."""
    rng = rng or np.random.default_rng(0)
    shape = hand_shape(aperture)
    pts_cam = shape @ R_cam_hand.T + wrist_cam
    px = camera.project(pts_cam) + rng.normal(0, px_noise, (21, 2))
    centre = pts_cam.mean(axis=0)
    world = (pts_cam - centre) + rng.normal(0, world_noise, (21, 3))
    return HandObservation(t=t, image_px=px, world=world)
