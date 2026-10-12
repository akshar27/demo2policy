"""Damped-least-squares inverse kinematics for the Panda arm, solved on a scratch MjData copy."""
from __future__ import annotations

import mujoco
import numpy as np

ARM_DOFS = 7


def down_rotation(yaw: float) -> np.ndarray:
    """Gripper pointing straight down; `yaw` rotates the hand about the world z axis.

    Columns are the site's x/y/z axes in the world frame. z = (0,0,-1). The fingers close along the site y axis,
    which is (sin yaw, -cos yaw, 0).
    """
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rotation_error(r_cur: np.ndarray, r_des: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector (world frame) that takes r_cur toward r_des."""
    return 0.5 * sum(np.cross(r_cur[:, i], r_des[:, i]) for i in range(3))


class IKSolver:
    def __init__(self, model: mujoco.MjModel, site: str = "grip_site", damping: float = 1e-2,
                 iters: int = 40, max_dq: float = 0.2, tol_pos: float = 1e-4, tol_rot: float = 1e-3):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site '{site}' not found in model")
        self.data = mujoco.MjData(model)
        self.damping, self.iters, self.max_dq = damping, iters, max_dq
        self.tol_pos, self.tol_rot = tol_pos, tol_rot
        self.lo = model.jnt_range[:ARM_DOFS, 0].copy()
        self.hi = model.jnt_range[:ARM_DOFS, 1].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def site_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:len(qpos)] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(self, qpos: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return (arm joint targets, position error m, rotation error rad). `qpos` is the full model qpos."""
        d = self.data
        d.qpos[:len(qpos)] = qpos
        q = d.qpos[:ARM_DOFS].copy()
        pos_err = rot_err = np.inf
        for _ in range(self.iters):
            d.qpos[:ARM_DOFS] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos = d.site_xpos[self.site_id]
            rot = d.site_xmat[self.site_id].reshape(3, 3)
            e = np.concatenate([target_pos - pos, rotation_error(rot, target_rot)])
            pos_err, rot_err = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if pos_err < self.tol_pos and rot_err < self.tol_rot:
                break
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, :ARM_DOFS], self._jacr[:, :ARM_DOFS]])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), e)
            n = np.abs(dq).max()
            if n > self.max_dq:
                dq *= self.max_dq / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q, pos_err, rot_err
