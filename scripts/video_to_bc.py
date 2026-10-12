"""Does a policy trained on video-derived demos work? Synthetic human videos -> multi-scene retargeting -> BC -> sim eval.

Compares, on identical seeded evaluation episodes: BC on scripted-expert demos (the upper reference) vs BC on demos
derived from a few (synthetic) human videos, each retargeted to many scenes. Writes docs/results/video_to_bc.md.
"""
import sys
import time
from pathlib import Path

from demo2policy.data.demo import load_demoset
from demo2policy.data.expert_data import generate_expert_demos
from demo2policy.eval.runner import evaluate
from demo2policy.pipeline import demos_from_poses_multiscene, poses_from_observations
from demo2policy.policies.bc import BCConfig, train_bc
from demo2policy.retarget.clean import TrackRejected
from demo2policy.sim.env import PickPlaceEnv
from demo2policy.synth.human import HumanStyle, default_camera, human_observations

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
env = PickPlaceEnv()
cam = default_camera()
bc = dict(obs_noise=0.05)
rows = ["| training data | demos used | BC success (3 seeds x %d eval episodes) |" % EPISODES, "|---|---|---|"]


def run(label, demos):
    wins = tot = 0
    per = []
    for s in range(3):
        res = evaluate(train_bc(demos, BCConfig(seed=s, **bc)), n=EPISODES, seed0=10_000, env=env)
        wins += res.successes
        tot += res.n
        per.append(res.successes / res.n)
    rows.append(f"| {label} | {len(demos)} | {wins / tot:.1%} ({' / '.join(f'{x:.0%}' for x in per)}) |")
    print(rows[-1], flush=True)


expert, _ = load_demoset("data/expert.h5")
run("scripted expert, 100 demos (reference)", expert[:100])
run("scripted expert, 10 demos", expert[:10])

# a pool of "human videos": expert motions filmed by the virtual camera at typical tracker noise
src, _ = generate_expert_demos(env, 24, seed0=5000)
video_poses = []
for k, d in enumerate(src):
    obs = human_observations(d, cam, HumanStyle(seed=k, world_rot=0.1 * (k % 5 - 2), world_scale=1.0 + 0.05 * (k % 3)))
    video_poses.append(poses_from_observations(obs, cam))

for n_videos, scenes in [(1, 100), (10, 10), (20, 5)]:
    demos, rejected = [], 0
    t0 = time.time()
    for k in range(n_videos):
        try:
            d, st = demos_from_poses_multiscene(video_poses[k], env, scenes, "synthetic_video", seed=k)
            demos += d
        except TrackRejected:
            rejected += 1
    print(f"  {n_videos} videos x {scenes} scenes -> {len(demos)} validated demos ({rejected} videos rejected, {time.time()-t0:.0f}s)", flush=True)
    run(f"{n_videos} synthetic human video(s) x {scenes} scenes", demos)

out = Path("docs/results")
out.mkdir(parents=True, exist_ok=True)
(out / "video_to_bc.md").write_text("\n".join(rows) + "\n\nAll rows: BC with 0.05 observation-noise augmentation, same seeded evaluation episodes. 'Human videos' are synthetic (expert motion filmed by a virtual camera, 1.5 px / 3 mm noise, dropouts, one glitch frame); real footage is milestone 6b.\n")
