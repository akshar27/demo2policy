"""Hand trajectory -> robot demonstration (end-effector target, gripper, yaw) in the simulator's frame.

Three decisions worth defending:
 * The end-effector follows the *pinch point* (midpoint of thumb and index tips), not the wrist: that is where the
   object is held, and it is what a parallel-jaw gripper's tool-centre-point corresponds to.
 * The gripper command comes from the *normalised* thumb-index distance (this video's own open/closed percentiles) with
   hysteresis, because a hand holding a 4 cm cube never closes to zero.
 * Task-space alignment ('anchor'): the camera's pose relative to the robot is rarely known, so a similarity transform
   (rotation about z + uniform scale + translation) is fitted so the human's grasp point lands on the cube and the release
   point on the target. 'extrinsics' mode trusts the known camera pose instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.env import CUBE_HALF, EnvConfig
from ..sim.ik import IKSolver, down_rotation
from .clean import TrackRejected, Trajectory


@dataclass
class RetargetConfig:
    align: str = "anchor"            # "anchor" | "extrinsics"
    close_thr: float = 0.35          # normalised pinch below this -> grasp
    open_thr: float = 0.65           # ... above this -> release (hysteresis)
    min_open_close_range: float = 0.08   # (p90 - p10) of pinch ratio; below this the video has no visible grasp
    scale_limits: tuple = (0.5, 2.0)
    max_infeasible_frac: float = 0.05
    cube_xy: tuple = (0.5, -0.15)    # where the cube is placed in the sim for anchor alignment
    target_xy: tuple = (0.5, 0.25)
    anchor_frames: int = 4           # frames averaged for the grasp / release anchor points


@dataclass
class Retargeted:
    ee_target: np.ndarray     # (T,3)
    gripper: np.ndarray       # (T,)
    yaw: np.ndarray           # (T,)
    init: dict                # cube_xy, cube_yaw, target_xy
    diagnostics: dict = field(default_factory=dict)


def _hysteresis(g: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Open (1) / closed (0) command with a dead band; starts open."""
    out = np.ones_like(g)
    state = 1.0
    for i, v in enumerate(g):
        if state == 1.0 and v < lo:
            state = 0.0
        elif state == 0.0 and v > hi:
            state = 1.0
        out[i] = state
    return out


def _yaw_from_axis(axis: np.ndarray) -> np.ndarray:
    """Gripper yaw whose closing direction (sin yaw, -cos yaw) matches the hand's horizontal closing axis (mod pi)."""
    psi = np.mod(np.arctan2(axis[:, 0], -axis[:, 1]), np.pi)          # in [0, pi)
    # the axis is sign-ambiguous; unwrap with period pi and re-centre on the median so the yaw stays near pi/2
    psi = np.unwrap(psi * 2) / 2
    return psi - np.round((np.median(psi) - np.pi / 2) / np.pi) * np.pi


def _fit_similarity_xy(src_a, src_b, dst_a, dst_b) -> tuple[float, float, np.ndarray]:
    """Rotation about z, uniform scale, translation taking src_a->dst_a and src_b->dst_b exactly (2 points determine it)."""
    vs, vd = np.asarray(src_b) - np.asarray(src_a), np.asarray(dst_b) - np.asarray(dst_a)
    s = np.linalg.norm(vd) / max(np.linalg.norm(vs), 1e-9)
    th = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])
    c, sn = np.cos(th), np.sin(th)
    R = np.array([[c, -sn], [sn, c]])
    t = np.asarray(dst_a) - s * R @ np.asarray(src_a)
    return th, s, t


def retarget(traj: Trajectory, cfg: RetargetConfig | None = None, env_cfg: EnvConfig | None = None) -> Retargeted:
    cfg, env_cfg = cfg or RetargetConfig(), env_cfg or EnvConfig()
    p10, p90 = np.percentile(traj.pinch, [10, 90])
    if p90 - p10 < cfg.min_open_close_range:
        raise TrackRejected("no_grasp_visible", f"pinch range {p90 - p10:.3f} < {cfg.min_open_close_range}")
    g_norm = np.clip((traj.pinch - p10) / (p90 - p10), 0.0, 1.0)
    grip = _hysteresis(g_norm, cfg.close_thr, cfg.open_thr)
    closed = np.where(grip == 0)[0]
    if len(closed) == 0:
        raise TrackRejected("no_grasp_visible", "gripper never closes")
    i_grasp = int(closed[0])
    reopen = np.where(grip[i_grasp:] == 1)[0]
    if len(reopen) == 0:
        raise TrackRejected("no_release", "gripper never reopens after the grasp")
    i_rel = i_grasp + int(reopen[0])

    pts = traj.pinch_pt.copy()
    yaw_h = _yaw_from_axis(traj.axis)
    diag: dict = {"i_grasp": i_grasp, "i_release": i_rel, "pinch_p10": float(p10), "pinch_p90": float(p90)}
    if cfg.align == "anchor":
        # Anchors are averaged over a few frames while the hand is still closed (a single noisy frame would shift the whole
        # path, and the thumb-index midpoint is only meaningful while the fingers are together).
        a_g = pts[i_grasp:i_grasp + cfg.anchor_frames].mean(axis=0)
        a_r = pts[max(i_rel - cfg.anchor_frames, i_grasp + 1):i_rel].mean(axis=0)
        th, s, t = _fit_similarity_xy(a_g[:2], a_r[:2], cfg.cube_xy, cfg.target_xy)
        if not (cfg.scale_limits[0] <= s <= cfg.scale_limits[1]):
            raise TrackRejected("bad_scale", f"alignment scale {s:.2f} outside {cfg.scale_limits}")
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts[:, :2] = (s * pts[:, :2] @ R.T) + t
        pts[:, 2] = pts[:, 2] - a_g[2] + CUBE_HALF
        yaw_r = yaw_h + th
        diag.update(align_rot_deg=float(np.degrees(th)), align_scale=float(s))
        cube_xy, target_xy = np.array(cfg.cube_xy), np.array(cfg.target_xy)
    elif cfg.align == "extrinsics":
        yaw_r = yaw_h
        cube_xy = pts[i_grasp:i_grasp + cfg.anchor_frames, :2].mean(axis=0)
        target_xy = pts[max(i_rel - cfg.anchor_frames, i_grasp + 1):i_rel, :2].mean(axis=0)
    else:
        raise ValueError(f"unknown align mode {cfg.align}")

    # the cube is placed so the grasp yaw is square to its faces (symmetric every 90 deg)
    cube_yaw = float(np.mod(yaw_r[i_grasp] - np.pi / 2 + np.pi / 4, np.pi / 2) - np.pi / 4)
    lo, hi = np.array(env_cfg.workspace_lo), np.array(env_cfg.workspace_hi)
    clipped = np.any((pts < lo - 1e-9) | (pts > hi + 1e-9), axis=1)
    pts = np.clip(pts, lo, hi)

    # IK feasibility along the whole path (warm-started), with the top-down orientation the sim uses
    from ..sim.env import HOME_QPOS, SCENE
    import mujoco
    ik = IKSolver(mujoco.MjModel.from_xml_path(str(SCENE)))
    q = HOME_QPOS.copy()
    bad = 0
    for i in range(len(pts)):
        sol, perr, _ = ik.solve(q, pts[i], down_rotation(float(yaw_r[i])))
        q[:7] = sol
        bad += perr > 5e-3
    infeasible = bad / len(pts)
    diag.update(frac_workspace_clipped=float(clipped.mean()), frac_ik_infeasible=float(infeasible))
    if infeasible > cfg.max_infeasible_frac:
        raise TrackRejected("ik_infeasible", f"{infeasible:.0%} of waypoints unreachable")
    return Retargeted(pts, grip, yaw_r, {"cube_xy": cube_xy.tolist(), "cube_yaw": cube_yaw, "target_xy": target_xy.tolist()}, diag)
