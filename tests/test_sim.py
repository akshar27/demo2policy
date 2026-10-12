import numpy as np
import pytest

from demo2policy.sim.env import CUBE_HALF, OBS_DIM, EnvConfig, PickPlaceEnv
from demo2policy.sim.expert import rollout_expert
from demo2policy.sim.ik import IKSolver, down_rotation, rotation_error


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv(EnvConfig())


def test_ik_reaches_poses_pointing_down(env):
    ik = IKSolver(env.model)
    q0 = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    for p in [(0.5, 0.0, 0.2), (0.45, -0.2, 0.05), (0.55, 0.25, 0.1)]:
        for yaw in (0.0, 0.8, -0.8):
            R = down_rotation(yaw)
            q, perr, rerr = ik.solve(q0, np.array(p), R)
            full = q0.copy()
            full[:7] = q
            pos, rot = ik.site_pose(full)
            assert np.linalg.norm(pos - p) < 2e-3
            assert np.linalg.norm(rotation_error(rot, R)) < 1e-2


def test_reset_is_deterministic_per_seed(env):
    a = env.reset(seed=7)
    b = env.reset(seed=7)
    c = env.reset(seed=8)
    assert a.shape == (OBS_DIM,)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert not np.allclose(a[4:6], c[4:6])


def test_cube_starts_in_randomisation_box_on_table(env):
    cfg = env.cfg
    for s in range(20):
        env.reset(seed=s)
        c = env.cube_pos
        assert cfg.cube_x[0] - 1e-3 <= c[0] <= cfg.cube_x[1] + 1e-3
        assert cfg.cube_y[0] - 1e-3 <= c[1] <= cfg.cube_y[1] + 1e-3
        assert abs(c[2] - CUBE_HALF) < 2e-3


def test_targets_are_clipped_and_rate_limited(env):
    env.reset(seed=0)
    start = env.ee_pos
    env.step(np.array([5.0, 5.0, 5.0]), 1.0)
    moved = np.linalg.norm(env._ee_cmd - start)
    assert moved <= env.cfg.max_ee_speed * env.dt + 1e-6
    for _ in range(200):
        env.step(np.array([5.0, 5.0, 5.0]), 1.0)
    assert np.all(env._ee_cmd <= np.array(env.cfg.workspace_hi) + 1e-6)


def test_expert_succeeds_on_randomised_cubes(env):
    wins = 0
    n = 30
    for s in range(n):
        env.reset(seed=500 + s)
        wins += rollout_expert(env)["info"].success
    assert wins / n >= 0.95, f"expert success {wins}/{n}"


def test_expert_rollout_is_reproducible(env):
    env.reset(seed=3)
    a = rollout_expert(env)
    env.reset(seed=3)
    b = rollout_expert(env)
    np.testing.assert_allclose(a["ee_target"], b["ee_target"])
    np.testing.assert_allclose(a["obs"], b["obs"], atol=1e-5)


def test_doing_nothing_does_not_succeed(env):
    env.reset(seed=1)
    for _ in range(40):
        _, info = env.step(env.ee_pos, 1.0)
    assert not info.success
