"""
ParticleCircle3D: A 3D particle confined to a spherical region.

Differences vs ParticleCircle2D:
  - 3D state:  (x, y, z, vx, vy, vz)
  - 3D action: (ax, ay, az) but **we drop the force/control part**:
      action is accepted (for API compatibility) but ignored.
  - **No goal / no reward shaping**: reward is always 0.0, terminated is always False.
  - Keep everything else: dt integration, speed clipping, elastic reflection at boundary,
    truncation at max_episode_steps, and rendering (simple 3D matplotlib).

Observation conventions:
  - Full env obs: (x, y, z, vx, vy, vz)
  - POMDP env obs: position only (x, y, z)
  - Stacked POMDP env obs: last `stack_size` positions, shape = (stack_size*3,)

Symmetry: SO(3) covariance of dynamics (with control removed, it's purely kinematic + reflection).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ParticleCircle3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        radius: float = 1.0,
        dt: float = 0.05,
        max_speed: float = 2.0,
        max_accel: float = 1.0,
        max_episode_steps: int = 200,
        drag: float = 0.0,
        seed: int = 0,
        render_mode: str | None = None,
        target_radius: float = 0.0001,
        no_goal: bool = False,
        max_action_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.radius = float(radius)
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.max_action_norm = float(max_action_norm)
        self.max_episode_steps = int(max_episode_steps)
        self.drag = float(drag)
        self.target_radius = float(target_radius)
        self.no_goal = bool(no_goal)
        self.target = np.zeros(3, dtype=np.float32)

        self._step = 0
        self.state = np.zeros(6, dtype=np.float32)  # (x, y, z, vx, vy, vz)

        self.action_space = spaces.Box(low=-max_accel, high=max_accel, shape=(3,), dtype=np.float32)

        # Full obs: (x, y, z, vx, vy, vz)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.np_random = np.random.default_rng(seed)

        # Rendering
        self.fig = None
        self.ax = None
        self.render_mode = render_mode

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clip_action_l2(action: np.ndarray, max_norm: float) -> np.ndarray:
        """Project action onto L2 ball of radius max_norm (SO(3)-covariant)."""
        a = np.asarray(action, dtype=np.float32).reshape(3,)
        n = float(np.linalg.norm(a))
        if n <= max_norm or n == 0.0:
            return a
        return (a * (max_norm / (n + 1e-8))).astype(np.float32)

    def _at_target(self) -> bool:
        return float(np.linalg.norm(self.state[:3] - self.target)) <= self.target_radius
    
    def _random_point_in_sphere(self, min_r: float = 0.1) -> np.ndarray:
        """Sample uniformly from the spherical shell [min_r, radius]."""
        v = self.np_random.normal(size=(3,))
        v = v / (np.linalg.norm(v) + 1e-8)
        u = self.np_random.uniform(min_r**3, self.radius**3)
        r = u ** (1.0 / 3.0)
        return (r * v).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self.state.copy()

    # ── Core gym interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._step = 0

        # Spawn away from origin
        # self.state[:3] = self._random_point_in_sphere(min_r=0.2 * self.radius)

        # Spawn away from origin
        self.state[:3] = self._random_point_in_sphere(min_r=2 * self.target_radius)

        # Small random initial velocity (uniform direction)
        speed = self.np_random.uniform(0.0, self.max_speed * 0.3)
        v = self.np_random.normal(size=(3,))
        v = v / (np.linalg.norm(v) + 1e-8)
        self.state[3:] = (speed * v).astype(np.float32)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_accel, self.max_accel).astype(np.float32)
        action = self._clip_action_l2(action, self.max_action_norm)

        x, y, z, vx, vy, vz = self.state

        ax = action[0] - self.drag * vx
        ay = action[1] - self.drag * vy
        az = action[2] - self.drag * vz

        # Symplectic Euler
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        vz_new = vz + az * self.dt

        # Speed clipping (direction-preserving)
        speed = float(np.sqrt(vx_new**2 + vy_new**2 + vz_new**2))
        if speed > self.max_speed:
            scale = self.max_speed / (speed + 1e-8)
            vx_new *= scale
            vy_new *= scale
            vz_new *= scale

        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        z_new = z + vz_new * self.dt

        # ── Elastic reflection off spherical boundary ─────────────────────────
        pos_new = np.array([x_new, y_new, z_new], dtype=np.float32)
        dist = float(np.linalg.norm(pos_new))

        if dist > self.radius:
            normal = -pos_new / (dist + 1e-8)  # inward-pointing
            v_new = np.array([vx_new, vy_new, vz_new], dtype=np.float32)
            v_new = v_new - 2.0 * float(np.dot(v_new, normal)) * normal
            vx_new, vy_new, vz_new = float(v_new[0]), float(v_new[1]), float(v_new[2])

            pos_new = pos_new * (self.radius / (dist + 1e-8))
            x_new, y_new, z_new = float(pos_new[0]), float(pos_new[1]), float(pos_new[2])

        self.state = np.array([x_new, y_new, z_new, vx_new, vy_new, vz_new], dtype=np.float32)
        self._step += 1

        # reward is distance to target 
        dist_to_target = float(np.linalg.norm(self.state[:3] - self.target))

        if self.no_goal:
            reward = 0.0
            terminated = False
        else:
            reward = -dist_to_target
            if self._at_target():
                reward += 10.0
            terminated = bool(self._at_target())

        truncated = bool(self._step >= self.max_episode_steps)

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return None

        if self.fig is None:
            self.fig = plt.figure(figsize=(6.4, 6.4))
            self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.clear()

        # Draw boundary as wireframe sphere (coarse)
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        xs = self.radius * np.outer(np.cos(u), np.sin(v))
        ys = self.radius * np.outer(np.sin(u), np.sin(v))
        zs = self.radius * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, color="black", linewidth=0.5, alpha=0.5)

        x, y, z, vx, vy, vz = self.state
        self.ax.scatter([x], [y], [z], s=40, c="blue")
        self.ax.quiver(x, y, z, vx, vy, vz, length=0.2, normalize=False, color="red", linewidth=1.5)

        lim = self.radius * 1.05
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        self.ax.set_title(f"ParticleCircle3D (step {self._step})")

        self.fig.canvas.draw()

        if self.render_mode == "rgb_array":
            img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = self.fig.canvas.get_width_height()
            img = img.reshape((h, w, 4))[:, :, :3].copy()
            return img

        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class POParticleCircle3DEnv(ParticleCircle3DEnv):
    """POMDP version: observe only position (x, y, z)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return self.state[:3].copy()


class POStackedParticleCircle3DEnv(ParticleCircle3DEnv):
    """POMDP + frame stacking: observe last `stack_size` positions → shape (stack_size*3,)."""

    def __init__(self, stack_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.stack_size = int(stack_size)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stack_size * 3,),
            dtype=np.float32,
        )
        self.frame_stack = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        pos = obs[:3].copy()
        self.frame_stack = np.tile(pos, self.stack_size)
        return self.frame_stack.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        pos = obs[:3].copy()
        self.frame_stack = np.roll(self.frame_stack, -len(pos))
        self.frame_stack[-len(pos):] = pos
        return self.frame_stack.copy(), reward, terminated, truncated, info