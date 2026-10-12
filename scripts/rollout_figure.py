"""Render a filmstrip of one scripted-expert episode to docs/img/expert-rollout.png."""
import sys

import matplotlib.pyplot as plt

from demo2policy.sim.env import PickPlaceEnv
from demo2policy.sim.expert import rollout_expert

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4
env = PickPlaceEnv()
env.reset(seed=seed)
frames, orig = [], env.step


def step(*a, **k):
    out = orig(*a, **k)
    if env.t % 25 == 0:
        frames.append(env.render(400, 300))
    return out


env.step = step
res = rollout_expert(env)
fig, ax = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 2.6))
for a, f in zip(ax, frames):
    a.imshow(f)
    a.axis("off")
plt.tight_layout()
plt.savefig("docs/img/expert-rollout.png", dpi=70)
print("success:", res["info"].success, "frames:", len(frames))
