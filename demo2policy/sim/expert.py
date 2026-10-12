"""Scripted pick-and-place expert. Uses privileged state (true cube/target pose); it is the ground-truth demonstrator
for validating the learning stack before any video is involved."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import CUBE_HALF, PickPlaceEnv


@dataclass
class ExpertConfig:
    hover_z: float = 0.22
    grasp_z: float = CUBE_HALF          # grip site height at the grasp (cube centre)
    place_z: float = CUBE_HALF + 0.012  # slightly above the table so the cube is not pressed down
    speed: float = 0.3                  # m/s between waypoints (0.5 drops the cube ~15% of the time)
    close_steps: int = 12
    open_steps: int = 8


def _wrap_pi2(a: float) -> float:
    """Cube is symmetric every 90 degrees; choose the equivalent grasp yaw closest to zero."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def rollout_expert(env: PickPlaceEnv, cfg: ExpertConfig | None = None, record: bool = True) -> dict:
    """Run one expert episode from the env's current (reset) state. Returns the trajectory and the final StepInfo."""
    cfg = cfg or ExpertConfig()
    cube, tgt = env.cube_pos, env.target
    # fingers close along site-y = (sin yaw, -cos yaw); align that with a cube face normal
    grasp_yaw = _wrap_pi2(env.cube_yaw) + np.pi / 2
    place_yaw = grasp_yaw

    legs = [  # (target xyz, gripper, yaw)
        ((cube[0], cube[1], cfg.hover_z), 1.0, grasp_yaw),
        ((cube[0], cube[1], cfg.grasp_z), 1.0, grasp_yaw),
        ("close", 0.0, grasp_yaw),
        ((cube[0], cube[1], cfg.hover_z), 0.0, grasp_yaw),
        ((tgt[0], tgt[1], cfg.hover_z), 0.0, place_yaw),
        ((tgt[0], tgt[1], cfg.place_z), 0.0, place_yaw),
        ("open", 1.0, place_yaw),
        ((tgt[0], tgt[1], cfg.hover_z), 1.0, place_yaw),
    ]
    obs_l, tgt_l, grip_l, yaw_l, info = [], [], [], [], None
    cur = env.ee_pos
    for goal, grip, yaw in legs:
        if isinstance(goal, str):
            n = cfg.close_steps if goal == "close" else cfg.open_steps
            path = [cur] * n
        else:
            goal = np.array(goal, dtype=float)
            n = max(2, int(np.ceil(np.linalg.norm(goal - cur) / (cfg.speed * env.dt))))
            path = [cur + (goal - cur) * (i + 1) / n for i in range(n)]
            cur = goal
        for p in path:
            if record:
                obs_l.append(env.obs())
            _, info = env.step(p, grip, yaw)
            if record:
                tgt_l.append(p)
                grip_l.append(grip)
                yaw_l.append(yaw)
    for _ in range(10):  # let the cube settle for the success check
        _, info = env.step(cur, 1.0, place_yaw)
    return {"obs": np.array(obs_l), "ee_target": np.array(tgt_l), "gripper": np.array(grip_l),
            "yaw": np.array(yaw_l), "info": info}
