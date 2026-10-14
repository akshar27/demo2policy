"""How good is Depth Anything V2 on a scene with *known* depth? Renders the simulator scene (MuJoCo gives exact depth),
runs the model, and measures (1) rank agreement with true inverse depth, (2) error after calibrating with only a few
reference points - the situation in the video pipeline, where the reference is the PnP wrist depth.

Caveat that matters: this scene shows a robot arm and a cube, not a human hand, and the model is run on synthetic renders.
It validates the plumbing and the calibration, not hand-depth accuracy on real footage. Writes docs/results/depth_check.md.
"""
from pathlib import Path

import mujoco
import numpy as np
from scipy.stats import spearmanr

from demo2policy.geometry.depth_fusion import fit_disparity_to_depth
from demo2policy.perception.depth import DepthAnythingV2
from demo2policy.sim.env import PickPlaceEnv
from demo2policy.sim.expert import rollout_expert

W, H = 640, 480
env = PickPlaceEnv()
env.reset(seed=4)
frames, orig = [], env.step


def step(*a, **k):
    out = orig(*a, **k)
    if env.t in (30, 60, 90):
        frames.append(None)
        r = mujoco.Renderer(env.model, H, W)
        r.update_scene(env.data, camera="front")
        rgb = r.render().copy()
        r.enable_depth_rendering()
        z = r.render().copy()
        frames[-1] = (rgb, z)
        r.close()
    return out


env.step = step
rollout_expert(env)
model = DepthAnythingV2()
rng = np.random.default_rng(0)
rows = ["| frame | Spearman(pred disparity, true 1/z) | abs-rel depth error, calibrated on 5 random pixels | on 20 pixels | oracle affine fit |", "|---|---|---|---|---|"]
for i, (rgb, z) in enumerate(frames):
    disp = model.disparity(rgb)
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    ys, xs = np.nonzero(valid)
    d_all, z_all = disp[ys, xs], z[ys, xs]
    rho = spearmanr(d_all, 1.0 / z_all).statistic
    errs = {}
    for k in (5, 20):
        e = []
        for _ in range(100):
            idx = rng.choice(len(d_all), k, replace=False)
            try:
                cal = fit_disparity_to_depth(d_all[idx], z_all[idx])
            except ValueError:
                continue
            e.append(np.median(np.abs(cal.depth(d_all) - z_all) / z_all))
        errs[k] = float(np.median(e))
    oracle = fit_disparity_to_depth(d_all, z_all)
    o = float(np.median(np.abs(oracle.depth(d_all) - z_all) / z_all))
    rows.append(f"| {i} | {rho:.3f} | {errs[5]:.1%} | {errs[20]:.1%} | {o:.1%} |")
    print(rows[-1], flush=True)
out = Path("docs/results")
out.mkdir(parents=True, exist_ok=True)
(out / "depth_check.md").write_text("\n".join(rows) + "\n\nDepth Anything V2 Small on 640x480 renders of the simulator scene (arm + cube + floor); true depth from MuJoCo. Median absolute relative depth error over all valid pixels. A robot-arm render is NOT a human hand on real video; see the script docstring.\n")
