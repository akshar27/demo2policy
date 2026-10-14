"""Calibrate relative disparity to metric depth, and fuse it with the PnP depth estimate.

Model: 1/z = a * d + b  (d = a depth model's relative disparity at a pixel, z = metric camera depth). (a, b) is fitted per
video by robust least squares against reference depths (here: the lifting step's wrist depth over frames).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class DisparityCalibration:
    a: float
    b: float
    n: int
    residual_rel: float   # median |z_pred - z_ref| / z_ref on the fitting points

    def depth(self, d: np.ndarray | float) -> np.ndarray | float:
        inv = self.a * np.asarray(d, dtype=float) + self.b
        return 1.0 / np.maximum(inv, 1e-3)


def fit_disparity_to_depth(d: np.ndarray, z_ref: np.ndarray, iters: int = 5, huber_k: float = 1.5) -> DisparityCalibration:
    """Robust (IRLS / Huber) affine fit of 1/z_ref on d. Needs >= 3 points with some spread in d."""
    d, z_ref = np.asarray(d, float), np.asarray(z_ref, float)
    ok = np.isfinite(d) & np.isfinite(z_ref) & (z_ref > 0)
    d, y = d[ok], 1.0 / z_ref[ok]
    if len(d) < 3 or np.ptp(d) < 1e-6:
        raise ValueError("not enough spread in disparity to calibrate depth")
    A = np.stack([d, np.ones_like(d)], axis=1)
    w = np.ones_like(d)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        r = y - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.sqrt(np.minimum(1.0, huber_k * s / np.maximum(np.abs(r), 1e-12)))
    cal = DisparityCalibration(float(coef[0]), float(coef[1]), int(len(d)), 0.0)
    cal.residual_rel = float(np.median(np.abs(cal.depth(d) - 1.0 / y) * y))
    return cal


def fuse_depth(z_pnp: float, z_depth_model: float, w_model: float = 0.5) -> float:
    """Weighted fusion in inverse-depth space (errors in disparity are closer to additive than errors in depth)."""
    return float(1.0 / ((1.0 - w_model) / z_pnp + w_model / z_depth_model))


def refine_with_depth(poses: list, disparities: list, cam, w_model: float = 0.5, max_residual: float = 0.15):
    """Fuse a depth model's reading into each pose's pinch point (moved along its camera ray).

    `disparities[i]` is the model's relative disparity at poses[i].pinch_px (None where the hand or the sample is missing).
    The affine disparity->depth map is calibrated against the PnP depths themselves, so this REDUCES PER-FRAME NOISE but
    inherits any systematic scale bias of the PnP depth; it cannot fix a wrong hand-size assumption. If the calibration fits
    badly (median relative residual > max_residual) the poses are returned unchanged. Returns (poses, info).
    """
    idx = [i for i, (p, d) in enumerate(zip(poses, disparities)) if p is not None and d is not None]
    info = {"fusion": "skipped", "n_calibration": len(idx)}
    if len(idx) < 8:
        info["reason"] = "too_few_depth_samples"
        return poses, info
    z_pnp = np.array([cam.world_to_cam(poses[i].pinch_pt)[2] for i in idx])
    d = np.array([disparities[i] for i in idx])
    try:
        cal = fit_disparity_to_depth(d, z_pnp)
    except ValueError as e:
        info["reason"] = str(e)
        return poses, info
    info.update(a=cal.a, b=cal.b, residual_rel=cal.residual_rel)
    if cal.residual_rel > max_residual:
        info["reason"] = f"calibration residual {cal.residual_rel:.0%} > {max_residual:.0%}"
        return poses, info
    out = list(poses)
    for i, z0, di in zip(idx, z_pnp, d):
        p = poses[i]
        z1 = fuse_depth(float(z0), float(cal.depth(di)), w_model)
        P = cam.world_to_cam(p.pinch_pt)
        out[i] = replace(p, pinch_pt=cam.cam_to_world(P * (z1 / P[2])))
    info["fusion"] = "applied"
    return out, info
