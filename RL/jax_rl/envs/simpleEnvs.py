"""
Registration and utilities for simple symmetric environments
Import this module to register all simple environments with gymnasium
"""

from gymnasium.envs.registration import register

# 2D circular environment with SO(2) symmetry
register(
    id='ParticleCircle2D-v0',
    entry_point='jax_rl.envs.particleCircle2D:ParticleCircle2DEnv',
    max_episode_steps=200,
)

register(
    id='POParticleCircle2D-v0',
    entry_point='jax_rl.envs.particleCircle2D:POParticleCircle2DEnv',
    max_episode_steps=200,
)

register(
    id='POStackedParticleCircle2D-v0',
    entry_point='jax_rl.envs.particleCircle2D:POStackedParticleCircle2DEnv',
    max_episode_steps=200,
    kwargs={'stack_size': 4},
)


# 3D spherical environment (no control, no goal)
register(
    id='ParticleCircle3D-v0',
    entry_point='jax_rl.envs.particleCircle3D:ParticleCircle3DEnv',
    max_episode_steps=200,
)

register(
    id='POParticleCircle3D-v0',
    entry_point='jax_rl.envs.particleCircle3D:POParticleCircle3DEnv',
    max_episode_steps=200,
)

register(
    id='POStackedParticleCircle3D-v0',
    entry_point='jax_rl.envs.particleCircle3D:POStackedParticleCircle3DEnv',
    max_episode_steps=200,
    kwargs={'stack_size': 4},
)

# Reacher2D
register(
    id='Reacher2D-v0',
    entry_point='jax_rl.envs.reacher2D:Reacher2DEnv',
    max_episode_steps=200,
)

register(
    id='POReacher2D-v0',
    entry_point='jax_rl.envs.reacher2D:POReacher2DEnv',
    max_episode_steps=200,
)

register(
    id='POStackedReacher2D-v0',
    entry_point='jax_rl.envs.reacher2D:POStackedReacher2DEnv',
    max_episode_steps=200,
    kwargs={'stack_size': 4},
)

# TriFingerPro-v0 stub: obs=pos(9)+vel(9)+torque(9)+tip_force(3)=30, act=9.
# Does not require trifinger-simulation; dims are confirmed from dataset.
# The existing state_dim/action_dim fallback in train scripts handles any mismatch.
register(
    id='TriFingerPro-v0',
    entry_point='jax_rl.envs.trifinger_stub:TriFingerProStubEnv',
)

register(
    id='TriFingerPro-position-v0',
    entry_point='jax_rl.envs.trifinger_stub:TriFingerProStubEnv',
)


# Export environment classes and utility functions
from jax_rl.envs.particleCircle2D import (
    ParticleCircle2DEnv,
    POParticleCircle2DEnv,
    POStackedParticleCircle2DEnv,
)
from jax_rl.envs.particleCircle3D import (
    ParticleCircle3DEnv,
    POParticleCircle3DEnv,
    POStackedParticleCircle3DEnv,
)

from jax_rl.envs.reacher2D import (
    Reacher2DEnv,
    POReacher2DEnv,
    POStackedReacher2DEnv,
)


__all__ = [
    'ParticleCircle2DEnv',
    'POParticleCircle2DEnv',
    'POStackedParticleCircle2DEnv',
    'ParticleCircle3DEnv',
    'POParticleCircle3DEnv',
    'POStackedParticleCircle3DEnv',
    'Reacher2DEnv',
    'POReacher2DEnv',
    'POStackedReacher2DEnv',
]
