"""Monte-Carlo error of monocular hand lifting on synthetic hands (ground truth known). Writes docs/results/lifting_error.md."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from demo2policy.geometry.camera import Camera
from demo2policy.geometry.lifting import lift_hand, rotation_angle_deg
from demo2policy.perception.hand_model import synth_observation

cam = Camera.from_fov(640, 480, 65.0)
rng = np.random.default_rng(0)


def trial(depth, px_noise, world_noise, scale_err):
    R = Rotation.from_euler("xyz", rng.uniform(-0.5, 0.5, 3)).as_matrix() @ np.diag([1, -1, -1])
    wrist = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08), depth])
    obs = synth_observation(cam, R, wrist, rng.uniform(0, 1), px_noise=px_noise, world_noise=world_noise, rng=rng)
    obs.world = obs.world * (1 + scale_err)
    pose = lift_hand(obs, cam)
    return np.linalg.norm(pose.wrist[:2] - wrist[:2]), abs(pose.depth - wrist[2]), rotation_angle_deg(pose.R, R)


rows = ["| depth (m) | landmark noise (px) | world noise (mm) | metric-scale error | lateral err (mm) | depth err (mm) | rotation err (deg) |",
        "|---|---|---|---|---|---|---|"]
for depth in (0.4, 0.6, 0.9):
    for px, wn, se in [(0, 0, 0), (1, 2, 0), (2, 3, 0), (4, 5, 0), (2, 3, 0.05), (2, 3, 0.10), (2, 3, -0.10)]:
        e = np.array([trial(depth, px, wn / 1000, se) for _ in range(300)])
        m = np.median(e, axis=0)
        rows.append(f"| {depth} | {px} | {wn} | {se:+.0%} | {m[0]*1000:.1f} | {m[1]*1000:.1f} | {m[2]:.1f} |")
out = Path("docs/results")
out.mkdir(parents=True, exist_ok=True)
(out / "lifting_error.md").write_text("\n".join(rows) + "\n\nMedians over 300 random synthetic hands per row (VGA camera, 65 deg HFOV, random orientation within +-0.5 rad, random pinch).\n")
print("\n".join(rows))
