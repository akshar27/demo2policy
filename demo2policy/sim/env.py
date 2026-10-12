"""Tabletop pick-and-place environment: Franka Panda, one cube, one target zone.

Action = absolute end-effector position target (robot base frame) + gripper command in [0 closed, 1 open]; the
end-effector is held pointing down at a fixed yaw. Everything downstream (expert, retargeted video demos, learned
policies) speaks this same interface, so demos from different sources are interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .ik import IKSolver, down_rotation

SCENE = Path(__file__).parent / "assets" / "panda" / "d2p_scene.xml"
HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
OBS_DIM = 10  # ee(3) gripper_opening(1) cube(3) target(3)
CUBE_HALF = 0.02


@dataclass
class EnvConfig:
    control_hz: float = 20.0
    max_ee_speed: float = 0.8            # m/s; commanded targets are rate-limited to this
    workspace_lo: tuple = (0.25, -0.45, 0.015)
    workspace_hi: tuple = (0.75, 0.45, 0.45)
    cube_x: tuple = (0.42, 0.58)         # randomisation box for the cube centre
    cube_y: tuple = (-0.25, -0.05)
    cube_yaw: float = np.pi / 4          # +- radians
    target_xy: tuple = (0.5, 0.25)
    target_jitter: float = 0.0           # +- m, uniform, applied to the target if > 0
    success_radius: float = 0.03
    max_steps: int = 300
    seed: int = 0


@dataclass
class StepInfo:
    success: bool
    cube_in_zone: bool
    cube_lifted: bool
    t: int
    ik_pos_err: float = 0.0
    extra: dict = field(default_factory=dict)


class PickPlaceEnv:
    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.ik = IKSolver(self.model)
        self.n_sub = max(1, round(1.0 / (self.cfg.control_hz * self.model.opt.timestep)))
        self.dt = self.n_sub * self.model.opt.timestep
        self.cube_qadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")]
        self.cube_vadr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")]
        self.target_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.rng = np.random.default_rng(self.cfg.seed)
        self.yaw = 0.0
        self.target = np.array([*self.cfg.target_xy, 0.0])
        self.t = 0
        self._ee_cmd = np.zeros(3)
        self._lo, self._hi = np.array(self.cfg.workspace_lo), np.array(self.cfg.workspace_hi)

    # ------------------------------------------------------------------ state accessors
    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ik.site_id].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    @property
    def cube_yaw(self) -> float:
        w, x, y, z = self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def gripper_opening(self) -> float:
        return float(self.data.qpos[7] + self.data.qpos[8])

    def obs(self) -> np.ndarray:
        return np.concatenate([self.ee_pos, [self.gripper_opening], self.cube_pos, self.target]).astype(np.float32)

    # ------------------------------------------------------------------ gym-ish API
    def reset(self, seed: int | None = None, cube_xy: tuple | None = None, cube_yaw: float | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, r = self.cfg, self.rng
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = HOME_QPOS
        self.data.ctrl[:7] = HOME_QPOS[:7]
        self.data.ctrl[7] = 255
        xy = np.array(cube_xy) if cube_xy is not None else np.array([r.uniform(*c.cube_x), r.uniform(*c.cube_y)])
        yaw = cube_yaw if cube_yaw is not None else r.uniform(-c.cube_yaw, c.cube_yaw)
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = [xy[0], xy[1], CUBE_HALF]
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        tgt = np.array(c.target_xy) + (r.uniform(-c.target_jitter, c.target_jitter, 2) if c.target_jitter > 0 else 0)
        self.target = np.array([tgt[0], tgt[1], 0.0])
        self.model.site_pos[self.target_site] = [tgt[0], tgt[1], 0.001]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(0.3 / self.model.opt.timestep)):  # settle
            mujoco.mj_step(self.model, self.data)
        self._ee_cmd = self.ee_pos
        self.yaw = 0.0
        self.t = 0
        return self.obs()

    def step(self, ee_target: np.ndarray, gripper: float, yaw: float | None = None) -> tuple[np.ndarray, StepInfo]:
        """Apply an absolute EE target (rate-limited + clipped to the workspace) and a gripper command for one control step."""
        if yaw is not None:
            self.yaw = yaw
        tgt = np.clip(np.asarray(ee_target, dtype=float), self._lo, self._hi)
        delta = tgt - self._ee_cmd
        max_d = self.cfg.max_ee_speed * self.dt
        n = np.linalg.norm(delta)
        self._ee_cmd = self._ee_cmd + (delta * (max_d / n) if n > max_d else delta)
        q, perr, _ = self.ik.solve(self.data.qpos, self._ee_cmd, down_rotation(self.yaw))
        self.data.ctrl[:7] = q
        self.data.ctrl[7] = 255.0 * float(np.clip(gripper, 0.0, 1.0))
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        return self.obs(), self._info(perr)

    def _info(self, ik_err: float = 0.0) -> StepInfo:
        cube = self.cube_pos
        in_zone = bool(np.linalg.norm(cube[:2] - self.target[:2]) < self.cfg.success_radius and cube[2] < CUBE_HALF + 0.01)
        speed = float(np.linalg.norm(self.data.qvel[self.cube_vadr:self.cube_vadr + 3]))
        released = bool(self.ee_pos[2] > cube[2] + 0.06)
        return StepInfo(success=in_zone and speed < 0.03 and released, cube_in_zone=in_zone,
                        cube_lifted=bool(cube[2] > CUBE_HALF + 0.03), t=self.t, ik_pos_err=ik_err)

    def render(self, width: int = 480, height: int = 360, camera: str = "front") -> np.ndarray:
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()
