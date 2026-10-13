"""Minimal TriFingerPro-v0 gymnasium stub.

Provides the correct observation/action spaces so make_env() can be called
without requiring trifinger-simulation on every node.

Observation: pos(9) + vel(9) + torque(9) + tip_force(3) = 30-dim
Action:      joint torques(9), normalised to [-1, 1]
"""

import numpy as np
import gymnasium as gym


class TriFingerProStubEnv(gym.Env):
    """Lightweight TriFingerPro env that only exposes spaces, no simulation.

    Shared by both TriFingerPro-v0 (torque control) and
    TriFingerPro-position-v0 (position control) — same obs/action dims.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(30, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(30, dtype=np.float32), 0.0, False, False, {}

    def render(self):
        pass
