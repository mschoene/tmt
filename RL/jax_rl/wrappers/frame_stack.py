import collections

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stack_axis=-1):
        super().__init__(env)
        self._num_stack = num_stack
        self._stack_axis = stack_axis

        self._frames = collections.deque([], maxlen=num_stack)

        low = np.repeat(self.observation_space.low, num_stack, axis=stack_axis)
        high = np.repeat(self.observation_space.high,
                         num_stack,
                         axis=stack_axis)
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=self.observation_space.dtype)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            obs, info = ret
        else:
            obs = ret
            info = {}
            
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        ret = self.env.step(action)
        # Handle Gymnasium's 5-tuple return (obs, reward, terminated, truncated, info)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = terminated or truncated
        else:
            obs, reward, done, info = ret

        self._frames.append(obs)
        # Return 4 values to maintain compatibility with pomujoco.py and legacy code
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._num_stack
        return np.concatenate(list(self._frames), axis=self._stack_axis)