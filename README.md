# demo2policy

Learn robot manipulation policies from human videos: video -> hand tracking -> retargeted robot demonstrations ->
imitation-learned policy -> evaluated in MuJoCo simulation.

**Status:** milestone 1 of 7 done (simulator + scripted expert). See `docs/system-design.md` and `docs/milestones.md`.

![Scripted expert pick-and-place](docs/img/expert-rollout.png)

## Run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/rollout_figure.py 4     # render one expert episode
```
