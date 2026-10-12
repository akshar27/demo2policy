"""Demonstration format shared by every demo source (scripted expert, video retargeting) and every learner.

A `Demo` is one episode in the env's action space: per control step the observation, the absolute end-effector target,
the gripper command and the gripper yaw. A `DemoSet` is an HDF5 file holding many demos plus provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Demo:
    obs: np.ndarray           # (T, OBS_DIM) observation *before* each action
    ee_target: np.ndarray     # (T, 3) absolute EE position target, robot base frame
    gripper: np.ndarray       # (T,) 0 closed .. 1 open
    yaw: np.ndarray           # (T,) gripper yaw about world z
    dt: float                 # control period in seconds
    init: dict = field(default_factory=dict)      # cube_xy, cube_yaw, target_xy, seed
    source: str = "unknown"                       # "scripted" | "video:<file>"
    success: bool | None = None                   # outcome when replayed/generated, if known
    quality: dict = field(default_factory=dict)   # tracker/cleaning/retargeting diagnostics

    def __post_init__(self):
        n = len(self.obs)
        for name in ("ee_target", "gripper", "yaw"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {n}")

    def __len__(self) -> int:
        return len(self.obs)

    @property
    def actions(self) -> np.ndarray:
        """(T, 5) policy-space action: absolute ee_target (3), gripper, yaw."""
        return np.concatenate([self.ee_target, self.gripper[:, None], self.yaw[:, None]], axis=1).astype(np.float32)


def save_demoset(path: str | Path, demos: list[Demo], meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["meta"] = json.dumps(meta or {})
        f.attrs["n_demos"] = len(demos)
        for i, d in enumerate(demos):
            g = f.create_group(f"demo_{i:04d}")
            for name in ("obs", "ee_target", "gripper", "yaw"):
                g.create_dataset(name, data=getattr(d, name).astype(np.float32), compression="gzip")
            g.attrs["dt"] = d.dt
            g.attrs["source"] = d.source
            g.attrs["success"] = -1 if d.success is None else int(d.success)
            g.attrs["init"] = json.dumps(d.init, default=_jsonable)
            g.attrs["quality"] = json.dumps(d.quality, default=_jsonable)


def load_demoset(path: str | Path) -> tuple[list[Demo], dict]:
    demos = []
    with h5py.File(path, "r") as f:
        ver = int(f.attrs["schema_version"])
        if ver != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema version {ver}, this build reads {SCHEMA_VERSION}")
        for key in sorted(k for k in f.keys() if k.startswith("demo_")):
            g = f[key]
            succ = int(g.attrs["success"])
            demos.append(Demo(
                obs=g["obs"][:], ee_target=g["ee_target"][:], gripper=g["gripper"][:], yaw=g["yaw"][:],
                dt=float(g.attrs["dt"]), init=json.loads(g.attrs["init"]), source=str(g.attrs["source"]),
                success=None if succ < 0 else bool(succ), quality=json.loads(g.attrs["quality"])))
        meta = json.loads(f.attrs["meta"])
    return demos, meta


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
