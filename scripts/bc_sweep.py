"""BC ablations on scripted-expert demos: number of demos, time feature, observation noise. Writes docs/results/bc_sweep.{json,md}."""
import json
import sys
import time
from pathlib import Path

from demo2policy.data.demo import load_demoset
from demo2policy.eval.runner import evaluate
from demo2policy.policies.bc import BCConfig, train_bc
from demo2policy.sim.env import PickPlaceEnv

demos, _ = load_demoset("data/expert.h5")
env = PickPlaceEnv()
episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
seeds = [0, 1, 2]
grid = [  # (label, n_demos, cfg kwargs)
    ("n=10", 10, {}), ("n=25", 25, {}), ("n=50", 50, {}), ("n=100", 100, {}),
    ("n=100 +time", 100, {"use_time": True}),
    ("n=100 +obs_noise 0.05", 100, {"obs_noise": 0.05}),
    ("n=100 +obs_noise 0.1", 100, {"obs_noise": 0.1}),
]
rows = []
for label, n, kw in grid:
    wins = tot = 0
    per_seed = []
    for s in seeds:
        t0 = time.time()
        pol = train_bc(demos[:n], BCConfig(seed=s, **kw))
        res = evaluate(pol, n=episodes, seed0=10_000, env=env)
        wins += res.successes
        tot += res.n
        per_seed.append(res.successes / res.n)
        print(f"{label} seed {s}: {res.summary()} ({time.time() - t0:.0f}s)", flush=True)
    rows.append({"label": label, "n_demos": n, **kw, "per_seed": per_seed, "mean": wins / tot})
out = Path("docs/results")
out.mkdir(parents=True, exist_ok=True)
(out / "bc_sweep.json").write_text(json.dumps(rows, indent=2))
md = ["| config | success per training seed | mean |", "|---|---|---|"]
for r in rows:
    md.append(f"| {r['label']} | {' / '.join(f'{x:.0%}' for x in r['per_seed'])} | {r['mean']:.1%} |")
(out / "bc_sweep.md").write_text("\n".join(md) + f"\n\n{episodes} eval episodes per training seed, same seeded initial states for every row.\n")
print("\n".join(md))
