import numpy as np
import pytest

from demo2policy.data.expert_data import generate_expert_demos
from demo2policy.geometry.lifting import lift_hand
from demo2policy.pipeline import demo_from_poses, poses_from_observations
from demo2policy.retarget.clean import CleanConfig, TrackRejected, clean_track
from demo2policy.sim.env import PickPlaceEnv
from demo2policy.synth.human import HumanStyle, default_camera, human_observations


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


@pytest.fixture(scope="module")
def expert_demos(env):
    d, _ = generate_expert_demos(env, 6, seed0=100)
    return d


CAM = default_camera()


def _poses(demo, **style):
    return poses_from_observations(human_observations(demo, CAM, HumanStyle(**style)), CAM)


def test_pinch_point_lifted_close_to_truth(expert_demos):
    d = expert_demos[0]
    obs = human_observations(d, CAM, HumanStyle(n_dropouts=0, n_glitches=0, px_noise=0, world_noise=0, world_rot=0,
                                                world_scale=1.0, world_shift=(0, 0, 0)))
    pts = np.array([lift_hand(o, CAM).pinch_pt for o in obs])
    t = np.arange(len(obs)) / 30.0
    truth = np.stack([np.interp(t, np.arange(len(d)) * d.dt, d.ee_target[:, k]) for k in range(3)], axis=1)
    assert np.abs(pts - truth).max() < 1e-4


def test_clean_fills_short_dropouts_and_resamples(expert_demos):
    tr = clean_track(_poses(expert_demos[0], n_dropouts=3), CleanConfig(rate_hz=20))
    assert np.allclose(np.diff(tr.t), 0.05) and tr.quality["valid_ratio"] > 0.8
    assert tr.quality["dropped_glitch"] + tr.quality["dropped_low_quality"] >= 1  # the injected glitch frame was caught


def test_clean_rejects_with_reasons(expert_demos):
    d = expert_demos[0]
    poses = _poses(d)
    with pytest.raises(TrackRejected) as e:
        clean_track([None] * 50)
    assert e.value.reason == "no_hand"
    with pytest.raises(TrackRejected) as e:
        clean_track(poses[:10], CleanConfig(min_duration_s=2.0))
    assert e.value.reason == "too_short"
    holey = list(poses)
    for i in range(30, 60):
        holey[i] = None
    with pytest.raises(TrackRejected) as e:
        clean_track(holey, CleanConfig(max_gap_s=0.35, min_valid_ratio=0.1))
    assert e.value.reason == "gap_too_long"
    with pytest.raises(TrackRejected) as e:
        clean_track(poses, CleanConfig(min_valid_ratio=1.01))
    assert e.value.reason == "low_valid_ratio"


def test_no_visible_grasp_is_rejected(env, expert_demos):
    d = expert_demos[0]
    from copy import copy
    d2 = copy(d)
    d2.gripper = np.ones_like(d.gripper)  # a human who never closes their hand
    with pytest.raises(TrackRejected) as e:
        demo_from_poses(_poses(d2), env, "synthetic")
    assert e.value.reason == "no_grasp_visible"


def test_full_roundtrip_expert_to_synthetic_video_to_successful_robot_demo(env, expert_demos):
    wins = 0
    for d in expert_demos:
        out = demo_from_poses(_poses(d, seed=int(d.init["seed"])), env, "synthetic_video")
        wins += out.success
        assert out.quality["valid_ratio"] > 0.8 and out.quality["frac_ik_infeasible"] == 0
    assert wins >= len(expert_demos) - 1, f"{wins}/{len(expert_demos)} video-derived demos succeeded on replay"
