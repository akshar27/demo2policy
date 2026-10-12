import numpy as np
import pytest

torch = pytest.importorskip("torch")

from demo2policy.data.demo import Demo  # noqa: E402
from demo2policy.policies.bc import BCConfig, BCPolicy, train_bc  # noqa: E402


def _toy_demos(n=6, T=40, seed=0):
    """Synthetic demos where the action is a simple function of the observation (move toward the cube)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cube = np.array([rng.uniform(0.4, 0.6), rng.uniform(-0.2, 0.0), 0.02])
        ee = np.array([0.3, 0.0, 0.4])
        obs, tgt = [], []
        for _t in range(T):
            o = np.concatenate([ee, [0.08], cube, [0.0], [0.5, 0.25, 0.0]]).astype(np.float32)
            nxt = ee + 0.2 * (cube + [0, 0, 0.1] - ee)
            obs.append(o)
            tgt.append(nxt)
            ee = nxt
        out.append(Demo(obs=np.array(obs), ee_target=np.array(tgt), gripper=np.ones(T), yaw=np.full(T, 1.5), dt=0.05))
    return out


def test_bc_fits_simple_expert_and_loss_decreases():
    losses = []
    pol = train_bc(_toy_demos(), BCConfig(epochs=150, hidden=64, layers=2, batch_size=64, lr=3e-3), log=losses)
    assert losses[-1] < 0.2 * losses[0]
    d = _toy_demos(1, seed=5)[0]
    a = pol.act(d.obs[3], 3)
    assert np.linalg.norm(a.ee_target - d.ee_target[3]) < 0.02
    assert 0.0 <= a.gripper <= 1.0


def test_bc_save_load_roundtrip(tmp_path):
    pol = train_bc(_toy_demos(), BCConfig(epochs=5, hidden=32, layers=2))
    pol.save(tmp_path / "m.pt")
    back = BCPolicy.load(tmp_path / "m.pt")
    o = _toy_demos(1)[0].obs[0]
    a, b = pol.act(o, 0), back.act(o, 0)
    np.testing.assert_allclose(a.ee_target, b.ee_target, atol=1e-6)
    assert a.gripper == pytest.approx(b.gripper) and a.yaw == pytest.approx(b.yaw)


def test_training_is_deterministic_for_a_seed():
    a = train_bc(_toy_demos(), BCConfig(epochs=5, hidden=32, layers=2, seed=3))
    b = train_bc(_toy_demos(), BCConfig(epochs=5, hidden=32, layers=2, seed=3))
    o = _toy_demos(1)[0].obs[2]
    np.testing.assert_allclose(a.act(o, 2).ee_target, b.act(o, 2).ee_target, atol=1e-6)
