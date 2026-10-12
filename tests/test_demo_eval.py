import numpy as np
import pytest

from demo2policy.data.demo import Demo, load_demoset, save_demoset
from demo2policy.data.expert_data import generate_expert_demos
from demo2policy.eval.runner import evaluate
from demo2policy.eval.stats import wilson_interval
from demo2policy.policies.replay import ReplayPolicy
from demo2policy.sim.env import PickPlaceEnv


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


@pytest.fixture(scope="module")
def demos(env):
    d, stats = generate_expert_demos(env, 4, seed0=0)
    assert stats["kept"] == 4
    return d


def test_demoset_roundtrip(tmp_path, demos):
    p = tmp_path / "d.h5"
    save_demoset(p, demos, {"k": 1})
    back, meta = load_demoset(p)
    assert meta == {"k": 1} and len(back) == len(demos)
    np.testing.assert_allclose(back[0].obs, demos[0].obs, atol=1e-6)
    np.testing.assert_allclose(back[0].actions, demos[0].actions, atol=1e-6)
    assert back[0].init["seed"] == demos[0].init["seed"] and back[0].success is True


def test_demo_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        Demo(obs=np.zeros((5, 11)), ee_target=np.zeros((4, 3)), gripper=np.zeros(5), yaw=np.zeros(5), dt=0.05)


def test_schema_version_mismatch_is_rejected(tmp_path, demos):
    import h5py
    p = tmp_path / "d.h5"
    save_demoset(p, demos)
    with h5py.File(p, "r+") as f:
        f.attrs["schema_version"] = 99
    with pytest.raises(ValueError, match="schema version"):
        load_demoset(p)


def test_replay_in_its_own_scene_succeeds(env, demos):
    """Sanity check: determinism of the whole stack. Replaying demo k on seed k's initial state must succeed."""
    for d in demos[:3]:
        res = evaluate(ReplayPolicy(d), n=1, seed0=d.init["seed"], env=env)
        assert res.successes == 1


def test_replay_fails_when_cube_moves(env, demos):
    res = evaluate(ReplayPolicy(demos[0]), n=20, seed0=20_000, env=env)
    assert res.rate < 0.3  # open-loop replay cannot generalise to new cube positions


def test_wilson_interval_properties():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.8
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson_interval(0, 0) == (0.0, 1.0)
