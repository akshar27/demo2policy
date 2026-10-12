from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..policies.base import Policy
from ..sim.env import EnvConfig, PickPlaceEnv
from .stats import wilson_interval


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    steps: int
    cube_in_zone: bool
    cube_lifted: bool
    final_cube_xy: list


@dataclass
class EvalResult:
    policy: str
    n: int
    successes: int
    rate: float
    ci_low: float
    ci_high: float
    seed0: int
    seconds: float
    episodes: list = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.policy}: {self.successes}/{self.n} = {self.rate:.1%}  (95% CI {self.ci_low:.1%}-{self.ci_high:.1%})"


def evaluate(policy: Policy, n: int = 50, seed0: int = 10_000, env: PickPlaceEnv | None = None,
             env_cfg: EnvConfig | None = None, max_steps: int | None = None) -> EvalResult:
    """Closed-loop rollouts on seeded, randomised initial states. Same seeds -> identical initial states across policies,
    so results are directly (and pairwise) comparable."""
    env = env or PickPlaceEnv(env_cfg)
    max_steps = max_steps or env.cfg.max_steps
    t0 = time.time()
    eps = []
    for k in range(n):
        seed = seed0 + k
        obs = env.reset(seed=seed)
        policy.reset()
        info = None
        lifted = False
        for t in range(max_steps):
            a = policy.act(obs, t)
            obs, info = env.step(a.ee_target, a.gripper, a.yaw)
            lifted = lifted or info.cube_lifted
            if info.success:
                break
        eps.append(EpisodeResult(seed, bool(info.success), info.t, info.cube_in_zone, lifted,
                                 env.cube_pos[:2].round(4).tolist()))
    wins = sum(e.success for e in eps)
    lo, hi = wilson_interval(wins, n)
    return EvalResult(policy.name, n, wins, wins / n, lo, hi, seed0, time.time() - t0, [asdict(e) for e in eps])


def save_result(res: EvalResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "metrics.json"
    p.write_text(json.dumps(asdict(res), indent=2))
    return p
