"""`d2p` command line: generate demos, evaluate policies."""
from __future__ import annotations

import json
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


@app.command("train-bc")
def train_bc_cmd(demos: Path = Path("data/expert.h5"), n_demos: int = 0, epochs: int = 300, use_time: bool = False,
                 obs_noise: float = 0.0, seed: int = 0, out: Path = Path("runs/bc")):
    """Train a behavior-cloning policy on the first `n_demos` demos (0 = all). Writes <out>/model.pt."""
    from .policies.bc import BCConfig, train_bc
    ds, _ = load_demoset(demos)
    ds = ds[:n_demos] if n_demos else ds
    losses: list = []
    pol = train_bc(ds, BCConfig(epochs=epochs, use_time=use_time, obs_noise=obs_noise, seed=seed), log=losses)
    pol.save(out / "model.pt")
    (out / "train.json").write_text(json.dumps({"n_demos": len(ds), "loss": losses}))
    typer.echo(f"trained on {len(ds)} demos, final loss {losses[-1]:.4f} -> {out / 'model.pt'}")


@app.command("video-to-demo")
def video_to_demo(video: Path, scenes: int = 20, out: Path = Path("data/video_demos.h5"), hfov: float = 65.0,
                  cam_pitch: float = 0.0, depth: bool = False, seed: int = 0, qa: Path | None = Path("runs/video_qa")):
    """Track a human video (camera pitched down by --cam-pitch degrees, horizontal FOV --hfov), (--depth fuses Depth Anything V2 into the hand depth), retarget it to `scenes` cube/target layouts, validate each in the sim, save the demos."""
    from .perception.depth import DepthAnythingV2
    from .perception.tracker import MediaPipeHandTracker
    from .pipeline import process_video
    from .qa import write_qa

    env = PickPlaceEnv(EnvConfig())
    demos, report, poses = process_video(video, MediaPipeHandTracker(), env, scenes, seed, hfov, cam_pitch,
                                          depth=DepthAnythingV2() if depth else None)
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "track_quality"}, indent=2))
    if qa:
        typer.echo(f"QA written to {write_qa(qa, report, poses, demos)}")
    if not demos:
        typer.echo("no usable demos produced - see the QA report for why", err=True)
        raise typer.Exit(1)
    save_demoset(out, demos, {"generator": "video", **{k: v for k, v in report.items() if k not in ("track_quality",)}})
    typer.echo(f"saved {len(demos)} validated demos to {out}")
