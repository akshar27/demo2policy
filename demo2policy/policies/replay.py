from __future__ import annotations

import numpy as np

from ..data.demo import Demo
from .base import Action


class ReplayPolicy:
    """Open-loop baseline: re-issue a recorded demo's action sequence, ignoring observations.

    It can only succeed if the scene matches the one the demo was recorded in - which is exactly why it is the
    baseline a closed-loop learned policy must beat once the cube moves.
    """

    def __init__(self, demo: Demo):
        self.demo = demo
        self.name = f"replay[{demo.source}:{demo.init.get('seed', '?')}]"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, t: int) -> Action:
        i = min(t, len(self.demo) - 1)
        return Action(self.demo.ee_target[i].astype(float), float(self.demo.gripper[i]), float(self.demo.yaw[i]))
