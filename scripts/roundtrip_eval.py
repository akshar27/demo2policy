"""Synthetic video -> robot demo round trip under increasing tracker degradation. Writes docs/results/roundtrip.md."""
import collections
from pathlib import Path

from demo2policy.data.expert_data import generate_expert_demos
from demo2policy.pipeline import demo_from_poses, poses_from_observations
from demo2policy.retarget.clean import TrackRejected
from demo2policy.sim.env import PickPlaceEnv
from demo2policy.synth.human import HumanStyle, default_camera, human_observations

env = PickPlaceEnv()
cam = default_camera()
demos, _ = generate_expert_demos(env, 30, seed0=2000)
conds = [
    ("clean tracker", dict(px_noise=0, world_noise=0, n_dropouts=0, n_glitches=0)),
    ("typical (1.5px, 3mm, 2 dropouts, 1 glitch)", dict()),
    ("noisy (4px, 6mm)", dict(px_noise=4, world_noise=0.006)),
    ("very noisy (8px, 10mm)", dict(px_noise=8, world_noise=0.010)),
    ("many dropouts (6)", dict(n_dropouts=6)),
    ("metric scale +5%", dict(scale_err=0.05)),
    ("metric scale +15%", dict(scale_err=0.15)),
    ("metric scale -15%", dict(scale_err=-0.15)),
]
rows = ["| condition | replay success | rejected (reasons) | median max-gap (s) |", "|---|---|---|---|"]
for name, kw in conds:
    ok = rej = n = 0
    reasons = collections.Counter()
    for k, d in enumerate(demos):
        n += 1
        obs = human_observations(d, cam, HumanStyle(seed=k, **kw))
        try:
            out = demo_from_poses(poses_from_observations(obs, cam), env, "synthetic_video")
            ok += out.success
        except TrackRejected as e:
            rej += 1
            reasons[e.reason] += 1
    rows.append(f"| {name} | {ok}/{n} ({ok/n:.0%}) | {rej} ({', '.join(f'{k}:{v}' for k, v in reasons.items()) or '-'}) | - |")
    print(rows[-1], flush=True)
out = Path("docs/results")
out.mkdir(parents=True, exist_ok=True)
(out / "roundtrip.md").write_text("\n".join(rows[:2] + [r.rsplit("|", 2)[0] + "|" for r in rows[2:]]).replace(" | median max-gap (s) |", " |").replace("|---|---|---|---|", "|---|---|---|") + "\n\n30 scripted-expert demos per condition act as the 'human'; each is filmed by a virtual camera (unknown human-world rotation 0.35 rad / scale 1.15 / shift), tracked, lifted, cleaned, retargeted with anchor alignment, and replayed in the simulator. Success = the replayed video-derived demo places the cube in the target zone.\n")
