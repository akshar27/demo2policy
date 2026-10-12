from __future__ import annotations

from ..sim.env import PickPlaceEnv
from ..sim.expert import ExpertConfig, rollout_expert
from .demo import Demo


def generate_expert_demos(env: PickPlaceEnv, n: int, seed0: int = 0, cfg: ExpertConfig | None = None,
                          max_attempts: int | None = None) -> tuple[list[Demo], dict]:
    """Roll out the scripted expert on seeds seed0, seed0+1, ... keeping only successful episodes."""
    demos, tried = [], 0
    limit = max_attempts or n * 3
    seed = seed0
    while len(demos) < n and tried < limit:
        env.reset(seed=seed)
        init = {"cube_xy": env.cube_pos[:2].tolist(), "cube_yaw": env.cube_yaw, "target_xy": env.target[:2].tolist(), "seed": seed}
        r = rollout_expert(env, cfg)
        tried += 1
        if r["info"].success:
            demos.append(Demo(obs=r["obs"], ee_target=r["ee_target"], gripper=r["gripper"], yaw=r["yaw"],
                              dt=env.dt, init=init, source="scripted", success=True))
        seed += 1
    return demos, {"attempted": tried, "kept": len(demos), "seed0": seed0, "expert_success_rate": len(demos) / max(tried, 1)}
