"""
Reacher2D: planar 2-link arm reaching a 2D goal.

State:
  q1, q2, dq1, dq2

Action:
  tau1, tau2 (joint torques, clipped)

Observation conventions (matched to ParticleCircle* style):
  - Full env obs: end-effector position only (x_ee, y_ee)
  - POMDP env obs: same as full (kept for symmetry with other envs)
  - Stacked POMDP env obs: last `stack_size` ee positions, shape = (stack_size*2,)

Goal:
  sampled (optional) uniformly in an annulus within reachable workspace.

Reward (dense):
  -||ee_pos - goal|| + success bonus.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Reacher2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        link_lengths=(0.6, 0.4),
        dt: float = 0.05,
        max_torque: float = 1.0,
        max_speed: float = 8.0,
        damping: float = 0.1,
        inertia=(1.0, 1.0),
        max_episode_steps: int = 200,
        goal_radius: float = 0.005,
        random_goal: bool = True,
        goal_min_r: float = 0.05,
        seed: int = 0,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.l1 = float(link_lengths[0])
        self.l2 = float(link_lengths[1])
        self.dt = float(dt)

        self.max_torque = float(max_torque)
        self.max_speed = float(max_speed)
        self.damping = float(damping)
        self.I1 = float(inertia[0])
        self.I2 = float(inertia[1])

        self.max_episode_steps = int(max_episode_steps)

        self.goal_radius = float(goal_radius)
        self.random_goal = bool(random_goal)
        self.goal_min_r = float(goal_min_r)

        self._step = 0
        self.state = np.zeros(4, dtype=np.float32)  # (q1, q2, dq1, dq2)
        self.goal = np.zeros(2, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(2,),
            dtype=np.float32,
        )

        # Full obs is just end-effector position (x, y)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            # shape=(4,), # endeffector position and goal position (x_ee, y_ee, x_goal, y_goal)
            shape=(8,),  # [cos(q1), sin(q1), cos(q1+q2), sin(q1+q2), dq1, dq2, x_goal, y_goal]
            dtype=np.float32,
        )

        self.np_random = np.random.default_rng(seed)

        # Rendering
        self.fig = None
        self.ax = None
        self.render_mode = render_mode

    # ── Kinematics / helpers ─────────────────────────────────────────────────

    def _ee_pos(self, q1: float, q2: float) -> np.ndarray:
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return np.array([x, y], dtype=np.float32)

    def _sample_goal(self) -> np.ndarray:
        r_max = self.l1 + self.l2
        angle = self.np_random.uniform(0, 2 * np.pi)
        r = np.sqrt(self.np_random.uniform(self.goal_min_r**2, r_max**2))
        return np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)

    # def _get_obs(self) -> np.ndarray:
    #     q1, q2, _, _ = self.state
    #     return self._ee_pos(q1, q2).copy()
    def _get_obs(self) -> np.ndarray:
        q1, q2, dq1, dq2 = self.state
        return np.array(
            [
                np.cos(q1),
                np.sin(q1),
                np.cos(q1 + q2),
                np.sin(q1 + q2),
                dq1,
                dq2,
                self.goal[0],
                self.goal[1],
            ],
            dtype=np.float32,
        )
    
    def _at_goal(self) -> bool:
        q1, q2, _, _ = self.state
        ee = self._ee_pos(q1, q2)
        return float(np.linalg.norm(ee - self.goal)) <= self.goal_radius

    # ── Core gym interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._step = 0

        self.state[0] = float(self.np_random.uniform(-np.pi, np.pi))
        self.state[1] = float(self.np_random.uniform(-np.pi, np.pi))
        self.state[2] = float(self.np_random.uniform(-0.5, 0.5))
        self.state[3] = float(self.np_random.uniform(-0.5, 0.5))

        if self.random_goal:
            self.goal = self._sample_goal()
        else:
            self.goal = np.array([self.l1 + self.l2 - 0.05, 0.0], dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_torque, self.max_torque).astype(np.float32)

        q1, q2, dq1, dq2 = self.state
        tau1, tau2 = float(action[0]), float(action[1])

        ddq1 = (tau1 - self.damping * dq1) / self.I1
        ddq2 = (tau2 - self.damping * dq2) / self.I2

        dq1_new = dq1 + ddq1 * self.dt
        dq2_new = dq2 + ddq2 * self.dt
        dq1_new = float(np.clip(dq1_new, -self.max_speed, self.max_speed))
        dq2_new = float(np.clip(dq2_new, -self.max_speed, self.max_speed))

        q1_new = q1 + dq1_new * self.dt
        q2_new = q2 + dq2_new * self.dt
        q1_new = float((q1_new + np.pi) % (2 * np.pi) - np.pi)
        q2_new = float((q2_new + np.pi) % (2 * np.pi) - np.pi)

        self.state = np.array([q1_new, q2_new, dq1_new, dq2_new], dtype=np.float32)
        self._step += 1

        ee = self._ee_pos(q1_new, q2_new)
        dist = float(np.linalg.norm(ee - self.goal))
        reward = -dist
        if self._at_goal():
            reward += 10.0

        terminated = bool(self._at_goal())
        truncated = bool(self._step >= self.max_episode_steps)

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return None

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6.4, 6.4))

        self.ax.clear()

        q1, q2, _, _ = self.state
        p0 = np.array([0.0, 0.0], dtype=np.float32)
        p1 = np.array([self.l1 * np.cos(q1), self.l1 * np.sin(q1)], dtype=np.float32)
        p2 = self._ee_pos(q1, q2)

        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", linewidth=3)
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", linewidth=3)
        self.ax.scatter([p0[0]], [p0[1]], c="black", s=30)
        self.ax.scatter([p1[0]], [p1[1]], c="black", s=30)
        self.ax.scatter([p2[0]], [p2[1]], c="blue", s=40)

        self.ax.scatter([self.goal[0]], [self.goal[1]], c="red", s=40)
        goal_circle = plt.Circle((self.goal[0], self.goal[1]), self.goal_radius, fill=False, color="red", linewidth=2)
        self.ax.add_patch(goal_circle)

        reach = self.l1 + self.l2
        lim = reach * 1.1
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"Reacher2D (step {self._step})")

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


class POReacher2DEnv(Reacher2DEnv):
    """
    POMDP version: observe only end-effector position (x_ee, y_ee).
    This matches the "position-only" convention used for PO* envs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),  # [x_ee, y_ee, x_goal, y_goal]
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        q1, q2, _, _ = self.state
        return np.concatenate([self._ee_pos(q1, q2), self.goal]).astype(np.float32)


class POStackedReacher2DEnv(Reacher2DEnv):
    """POMDP + frame stacking: observe last `stack_size` ee positions → shape (stack_size*2,)."""

    def __init__(self, stack_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.stack_size = int(stack_size)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stack_size * 4,),
            dtype=np.float32,
        )
        self.frame_stack = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.frame_stack = np.tile(obs, self.stack_size)
        return self.frame_stack.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.frame_stack = np.roll(self.frame_stack, -len(obs))
        self.frame_stack[-len(obs):] = obs
        return self.frame_stack.copy(), reward, terminated, truncated, info