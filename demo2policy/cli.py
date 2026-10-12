"""`d2p` command line: generate demos, evaluate policies."""
from __future__ import annotations

from pathlib import Path

import typer

from .data.demo import load_demoset, save_demoset
from .data.expert_data import generate_expert_demos
from .eval.runner import evaluate, save_result
from .policies.replay import ReplayPolicy
from .sim.env import EnvConfig, PickPlaceEnv

app = typer.Typer(no_args_is_help=True, add_completion=False)


def make_policy(spec: str):
    """Policy spec: `replay:<demoset.h5>[:<index>]`  (more kinds are added by later milestones)."""
    kind, _, rest = spec.partition(":")
    if kind == "replay":
        path, _, idx = rest.partition(":")
        demos, _ = load_demoset(path)
        return ReplayPolicy(demos[int(idx or 0)])
    if kind == "bc":
        from .policies.bc import BCPolicy  # imported lazily: needs torch
        return BCPolicy.load(rest)
    raise typer.BadParameter(f"unknown policy spec '{spec}'")


@app.command("gen-expert")
def gen_expert(n: int = 50, seed0: int = 0, out: Path = Path("data/expert.h5")):
    """Roll out the scripted expert and save the successful demos."""
    env = PickPlaceEnv(EnvConfig())
    demos, stats = generate_expert_demos(env, n, seed0)
    save_demoset(out, demos, {"generator": "scripted_expert", **stats})
    typer.echo(f"saved {len(demos)} demos to {out}  ({stats})")


@app.command("eval")
def eval_cmd(policy: str, episodes: int = 50, seed0: int = 10_000, out: Path | None = None):
    """Evaluate a policy on seeded randomised episodes (95% Wilson CI)."""
    res = evaluate(make_policy(policy), n=episodes, seed0=seed0)
    typer.echo(res.summary())
    if out:
        typer.echo(f"wrote {save_result(res, out)}")


if __name__ == "__main__":
    app()
