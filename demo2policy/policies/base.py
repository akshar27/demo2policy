from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Action:
    ee_target: np.ndarray  # absolute, robot base frame
    gripper: float         # 0 closed .. 1 open
    yaw: float


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, t: int) -> Action: ...
