from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction

from jax_rl import wrappers
from jax_rl.wrappers import VideoRecorder

# Try to import PixelObservationWrapper from gymnasium (optional)
try:
    from gymnasium.experimental.wrappers import PixelObservationWrapper  # gymnasium >=0.29
except Exception:
    try:
        from gymnasium.wrappers import PixelObservationWrapper  # fallback if available
    except Exception:
        PixelObservationWrapper = None

class GymnasiumCompatibility(gym.Wrapper):
    """Convert Gymnasium (new API) to old Gym API expected by legacy wrappers."""
    def reset(self, seed=None, options=None):
        obs, _info = self.env.reset(seed=seed, options=options)
        return obs
    def step(self, action):
        # obs, reward, terminated, truncated, info = self.env.step(action)
        ret = self.env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = terminated or truncated
        else:
            obs, reward, done, info = ret

        done = terminated or truncated
        return obs, reward, done, info

def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             image_size: int = 84) -> gym.Env:
    # Set numpy seed for reproducibility
    np.random.seed(seed)
    
    # Variant suffixes that map to the same underlying gym env.
    _GYM_ALIASES = {'Swimmer-v5-angle': 'Swimmer-v5'}
    gym_env_name = _GYM_ALIASES.get(env_name, env_name)

    # Check if the env is in gymnasium.
    env_ids = list(gym.envs.registry.keys())

    if gym_env_name not in env_ids:
        raise ValueError(
            f"Unknown environment id '{env_name}'. It must be registered with "
            f"gymnasium (see jax_rl/envs/simpleEnvs.py for the custom envs this "
            f"codebase registers, e.g. by importing jax_rl.envs)."
        )
    env_name = gym_env_name  # use the real gym id for creation
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except TypeError:
        env = gym.make(env_name)
    env.reset(seed=seed)

    # Apply Gymnasium wrappers (new API) BEFORE conversion
    if isinstance(env.action_space, gym.spaces.box.Box):
        env = RescaleAction(env, -1.0, 1.0)

    if from_pixels:
        if PixelObservationWrapper is None:
            raise ImportError("PixelObservationWrapper not available in this gymnasium version")
        camera_id = 2 if 'domain_name' in locals() and domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(
            env,
            render_kwargs={
                'pixels': {
                    'height': image_size,
                    'width': image_size,
                    'camera_id': camera_id
                }
            },
        )
    else:
        # Ensure observations are float32 using the Gymnasium-style wrapper before conversion
        env = wrappers.SinglePrecision(env)
        # pass #TODO check if we need double since we got very small equivar errors at times

    # Convert to old Gym API for legacy wrappers
    env = GymnasiumCompatibility(env)

    # Legacy wrappers (expect old API) BELOW
    env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)
    
    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    if from_pixels:
        # After conversion, use legacy TakeKey to select 'pixels'
        env = wrappers.TakeKey(env, take_key='pixels')

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    return env
