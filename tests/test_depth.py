import numpy as np
import pytest

from demo2policy.geometry.camera import Camera
from demo2policy.geometry.depth_fusion import fit_disparity_to_depth, fuse_depth, refine_with_depth
from demo2policy.geometry.lifting import lift_hand
from demo2policy.perception.depth import sample_disparity
from demo2policy.perception.hand_model import synth_observation


def test_affine_calibration_recovers_known_map_with_outliers():
    rng = np.random.default_rng(0)
    z = rng.uniform(0.4, 1.2, 200)
    d = 3.0 / z + 0.5 + rng.normal(0, 0.01, 200)       # true: 1/z = (d - 0.5) / 3
    d[:10] += 2.0                                       # gross outliers
    cal = fit_disparity_to_depth(d, z)
    assert abs(cal.a - 1 / 3) < 0.01 and abs(cal.b + 0.5 / 3) < 0.02
    assert cal.residual_rel < 0.03


def test_calibration_needs_spread():
    with pytest.raises(ValueError):
        fit_disparity_to_depth(np.ones(20), np.linspace(0.5, 1, 20))


def test_fuse_depth_is_between_the_inputs():
    assert 0.5 < fuse_depth(0.5, 0.6) < 0.6
    assert fuse_depth(0.5, 0.6, w_model=0.0) == pytest.approx(0.5)
    assert fuse_depth(0.5, 0.6, w_model=1.0) == pytest.approx(0.6)


def test_sample_disparity_median_and_bounds():
    d = np.arange(100, dtype=float).reshape(10, 10)
    assert sample_disparity(d, np.array([5, 5]), radius=1) == d[4:7, 4:7].mean()
    with pytest.raises(ValueError):
        sample_disparity(d, np.array([50, 50]))


def test_fusion_reduces_per_frame_noise_but_keeps_a_systematic_scale_bias():
    """The honest property: calibrating against PnP depth averages out its noise, but cannot remove a shared scale bias."""
    cam = Camera.from_fov(640, 480, 65.0)
    rng = np.random.default_rng(1)
    R = np.diag([1.0, -1, -1])
    z_true = np.linspace(0.45, 0.9, 80)
    poses, disp = [], []
    for z in z_true:
        obs = synth_observation(cam, R, np.array([0.0, 0.0, z]), 0.3, px_noise=1.0, world_noise=0.0025, rng=rng)
        obs.world = obs.world * 1.10                        # tracker's metric scale is 10% too big -> PnP depth biased
        poses.append(lift_hand(obs, cam))
        disp.append(float(3.0 / z + 0.4 + rng.normal(0, 0.01)))   # depth model: accurate relative disparity
    fused, info = refine_with_depth(poses, disp, cam, w_model=0.7)
    assert info["fusion"] == "applied"
    z_pnp = np.array([cam.world_to_cam(p.pinch_pt)[2] for p in poses])
    z_fus = np.array([cam.world_to_cam(p.pinch_pt)[2] for p in fused])
    rel = lambda z: np.median(np.abs(z - z_true) / z_true)   # noqa: E731
    assert rel(z_fus) < rel(z_pnp)                           # fusion helps ...
    assert np.median(z_fus / z_true) > 1.05                  # ... but the 10% scale bias is still there


def test_refine_skips_when_calibration_is_poor_or_data_missing():
    cam = Camera.from_fov(640, 480, 65.0)
    R = np.diag([1.0, -1, -1])
    obs = synth_observation(cam, R, np.array([0.0, 0.0, 0.6]), 0.5)
    poses = [lift_hand(obs, cam)] * 3
    out, info = refine_with_depth(poses, [1.0, 2.0, 3.0], cam)
    assert info["fusion"] == "skipped" and out is poses
