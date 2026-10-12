# demo2policy

Learn robot manipulation policies from human videos: video -> hand tracking -> retargeted robot demonstrations ->
imitation-learned policy -> evaluated in MuJoCo simulation.

**Status:** milestones 1-3 of 7 done (simulator + scripted expert, demo format + replay baseline + eval harness, behavior cloning). See `docs/system-design.md` and `docs/milestones.md`.

![Scripted expert pick-and-place](docs/img/expert-rollout.png)

## Run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/rollout_figure.py 4     # render one expert episode
```

## Results so far (simulated expert demos, 95% CIs from `d2p eval`)

| policy | success on randomised cube positions |
|---|---|
| open-loop replay of one demo | 4% (2/50) |
| BC, 100 demos | 45% |
| BC, 100 demos + obs noise 0.05 | **92.7%** (mean of 3 training seeds, 150 episodes) |
| scripted expert (privileged state) | ~100% |

```bash
d2p gen-expert --n 100 --out data/expert.h5
d2p train-bc --n-demos 100 --obs-noise 0.05 --out runs/bc
d2p eval bc:runs/bc/model.pt --episodes 100
```
Next: the video side (hand tracking -> retargeting) so these demos can come from human video instead of the expert.
