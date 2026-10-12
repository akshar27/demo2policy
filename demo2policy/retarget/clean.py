"""Turn a raw per-frame track (with dropouts and jitter) into a clean, uniformly sampled trajectory - or reject it, with a reason."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..geometry.lifting import HandPose3D


class TrackRejected(Exception):
    """The track is not good enough to become a demo. `.reason` is a short machine-readable code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class CleanConfig:
    rate_hz: float = 20.0
    max_gap_s: float = 0.35          # longer dropouts cannot be bridged -> reject
    max_speed: float = 4.0           # m/s; faster frame-to-frame jumps are tracker glitches
    max_reproj_px: float = 12.0      # frames whose 3D fit disagrees with the 2D landmarks are dropped
    min_score: float = 0.5
    min_valid_ratio: float = 0.7
    smooth_sigma_s: float = 0.15     # Gaussian smoothing (offline, zero phase); synthetic sweep on a noisy tracker (4px/6mm): 0.06 s -> 64%, 0.12 s -> 89%, 0.2 s -> 96% of accepted tracks replay successfully; 0.15 is a compromise because heavy smoothing blurs gripper timing
    min_duration_s: float = 1.0


@dataclass
class Trajectory:
    t: np.ndarray             # (T,) seconds, uniform at cfg.rate_hz, starting at 0
    pinch_pt: np.ndarray      # (T,3)
    pinch: np.ndarray         # (T,) thumb-index distance / hand size
    axis: np.ndarray          # (T,3) closing axis (unit)
    quality: dict = field(default_factory=dict)


def clean_track(poses: list[HandPose3D | None], cfg: CleanConfig | None = None) -> Trajectory:
    cfg = cfg or CleanConfig()
    n_frames = len(poses)
    if n_frames == 0:
        raise TrackRejected("empty_track")
    good = [p for p in poses if p is not None and p.reproj_px <= cfg.max_reproj_px and p.score >= cfg.min_score]
    n_dropped_quality = sum(p is not None for p in poses) - len(good)
    # glitch rejection: drop a sample that implies an impossible speed from the last kept sample
    kept: list[HandPose3D] = []
    n_glitch = 0
    for p in good:
        if kept:
            dt = max(p.t - kept[-1].t, 1e-3)
            if np.linalg.norm(p.pinch_pt - kept[-1].pinch_pt) / dt > cfg.max_speed:
                n_glitch += 1
                continue
        kept.append(p)
    valid_ratio = len(kept) / n_frames
    if len(kept) < 4:
        raise TrackRejected("no_hand", f"{len(kept)} usable frames of {n_frames}")
    if valid_ratio < cfg.min_valid_ratio:
        raise TrackRejected("low_valid_ratio", f"{valid_ratio:.0%} < {cfg.min_valid_ratio:.0%}")
    t = np.array([p.t for p in kept])
    t = t - t[0]
    if t[-1] < cfg.min_duration_s:
        raise TrackRejected("too_short", f"{t[-1]:.2f}s")
    gaps = np.diff(t)
    if gaps.max() > cfg.max_gap_s:
        raise TrackRejected("gap_too_long", f"{gaps.max():.2f}s > {cfg.max_gap_s}s")

    grid = np.arange(0.0, t[-1], 1.0 / cfg.rate_hz)
    interp = lambda a: np.stack([np.interp(grid, t, a[:, k]) for k in range(a.shape[1])], axis=1)  # noqa: E731
    pts = interp(np.array([p.pinch_pt for p in kept]))
    axis = interp(np.array([p.closing_axis for p in kept]))
    pinch = np.interp(grid, t, np.array([p.pinch for p in kept]))
    sig = cfg.smooth_sigma_s * cfg.rate_hz
    pts = gaussian_filter1d(pts, sig, axis=0, mode="nearest")
    pinch = gaussian_filter1d(pinch, sig, mode="nearest")
    axis = gaussian_filter1d(axis, sig, axis=0, mode="nearest")
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    jitter = float(np.sqrt(np.mean(np.sum(np.diff(pts, 2, axis=0) ** 2, axis=1)))) if len(pts) > 2 else 0.0
    q = {"frames": n_frames, "valid_ratio": round(valid_ratio, 3), "dropped_low_quality": n_dropped_quality,
         "dropped_glitch": n_glitch, "max_gap_s": round(float(gaps.max()), 3), "duration_s": round(float(grid[-1]), 2),
         "accel_rms_m": jitter, "median_reproj_px": float(np.median([p.reproj_px for p in kept])),
         "median_scale_m": float(np.median([p.scale for p in kept]))}
    return Trajectory(grid, pts, pinch, axis, q)
