import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2policy.geometry.camera import Camera, look_at
from demo2policy.geometry.lifting import PALM, lift_hand, procrustes_rotation, rotation_angle_deg
from demo2policy.perception.hand_model import (OPEN_HAND, hand_scale, hand_shape, pinch_ratio, synth_observation)

CAM = Camera.from_fov(640, 480, 65.0)


def _rot(seed):
    return Rotation.random(random_state=seed).as_matrix()


def test_project_normalize_roundtrip():
    p = np.array([[0.1, -0.05, 0.6], [0.0, 0.0, 1.0]])
    n = CAM.normalize(CAM.project(p))
    np.testing.assert_allclose(n, p[:, :2] / p[:, 2:3], atol=1e-9)


def test_world_cam_transform_inverse():
    cam = Camera.from_fov(640, 480, 65.0, look_at([1.0, 0.0, 0.5], [0.5, 0.0, 0.0]))
    p = np.array([[0.4, 0.1, 0.0], [0.5, -0.2, 0.3]])
    np.testing.assert_allclose(cam.cam_to_world(cam.world_to_cam(p)), p, atol=1e-9)
    # the look-at target is on the optical axis
    c = cam.world_to_cam(np.array([0.5, 0.0, 0.0]))
    assert abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9 and c[2] > 0


def test_pinch_ratio_monotonic_in_aperture():
    r = [pinch_ratio(hand_shape(a)) for a in np.linspace(0, 1, 6)]
    assert r[0] < 0.1 < r[-1] and all(np.diff(r) > 0)


def test_procrustes_recovers_rotation_noise_free():
    for s in range(10):
        R = _rot(s)
        palm = OPEN_HAND[PALM] - OPEN_HAND[PALM].mean(0)
        assert rotation_angle_deg(procrustes_rotation(palm @ R.T), R) < 1e-3


def test_lift_exact_with_perfect_tracker():
    rng = np.random.default_rng(1)
    for s in range(20):
        R = Rotation.from_euler("xyz", rng.uniform(-0.6, 0.6, 3)).as_matrix() @ np.diag([1, -1, -1])  # fingers roughly up/away
        wrist = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.4, 0.9)])
        ap = rng.uniform(0, 1)
        obs = synth_observation(CAM, R, wrist, ap)
        pose = lift_hand(obs, CAM)
        assert np.linalg.norm(pose.wrist - wrist) < 1e-6
        assert rotation_angle_deg(pose.R, R) < 1e-4
        assert abs(pose.pinch - pinch_ratio(hand_shape(ap))) < 1e-9
        assert pose.reproj_px < 1e-6


def test_lift_is_noise_robust_at_typical_pixel_noise():
    rng = np.random.default_rng(2)
    errs = []
    for s in range(60):
        R = Rotation.from_euler("xyz", rng.uniform(-0.5, 0.5, 3)).as_matrix() @ np.diag([1, -1, -1])
        wrist = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.6])
        obs = synth_observation(CAM, R, wrist, 0.5, px_noise=2.0, world_noise=0.003, rng=rng)
        errs.append(np.linalg.norm(lift_hand(obs, CAM).wrist - wrist))
    assert np.median(errs) < 0.015  # 2 px landmark noise + 3 mm world-landmark noise -> ~1.5 cm at 0.6 m


def test_wrong_metric_scale_shows_up_as_depth_error_and_in_scale_diagnostic():
    """If the tracker's metric scale is off by k, depth is off by about k - the assumption the pipeline rests on."""
    R = np.diag([1.0, -1, -1])
    wrist = np.array([0.0, 0.0, 0.6])
    obs = synth_observation(CAM, R, wrist, 0.5)
    obs.world = obs.world * 1.2
    pose = lift_hand(obs, CAM)
    assert 0.6 * 1.1 < pose.depth < 0.6 * 1.3
    assert pose.scale == pytest.approx(1.2 * hand_scale(hand_shape(0.5)), rel=1e-6)


def test_reprojection_error_flags_inconsistent_observation():
    R = np.diag([1.0, -1, -1])
    obs = synth_observation(CAM, R, np.array([0.0, 0.0, 0.6]), 0.5)
    good = lift_hand(obs, CAM).reproj_px
    obs.image_px[8] += [40, 0]  # one landmark far off
    assert lift_hand(obs, CAM).reproj_px > good + 3
