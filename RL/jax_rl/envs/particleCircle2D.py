"""
ParticleCircle2D: A 2D particle confined to a circular region.

Symmetries (exact, target fixed at origin):
  - SO(2): continuous rotation of (x, y, vx, vy) → (R·x, R·y, R·vx, R·vy)
            with action transforming as (ax, ay) → R·(ax, ay)
  - This is the key difference from the square box (only D4 there).

State:  (x, y, vx, vy)   — position + velocity
Action: (ax, ay)          — acceleration (clipped)
Obs:    (x, y, vx, vy)   — fully observable (target fixed at origin, not included)

POMDP version: observe only (x, y) — no velocity.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ParticleCircle2DEnv(gym.Env):
    """
    2D particle in a circular boundary, task = reach origin.

    SO(2) symmetry is exact:
      R_θ · obs → same dynamics → R_θ · action
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        radius: float = 1.0,
        dt: float = 0.05,
        max_speed: float = 2.0,
        max_accel: float = 1.0,
        max_episode_steps: int = 200,
        target_radius: float = 0.001,
        # drag: float = 0.05,
        drag: float = 0.0,
        seed: int = 0,
        # kept for API compatibility but ignored — target is always origin
        obs_type: str = 'no_target',
        reward_type: str = 'dense',
        random_target: bool = False,
        render_mode: str = None,   # ← ADD: gymnasium passes this automatically
        no_goal: bool = False,       # ← ADD: if True, reward is always 0 (for testing symmetries without reward shaping)
        max_action_norm: float = 1.0,  # NEW: limit ||action||_2
        **kwargs,                  # ← ADD: absorb any other future gymnasium kwargs
    ):
        super().__init__()

        self.radius = radius
        self.dt = dt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_episode_steps = max_episode_steps
        self.max_action_norm = float(max_action_norm)  # NEW
        self.target_radius = target_radius
        self.drag = drag
        self.target = np.zeros(2, dtype=np.float32)  # always origin

        self.no_goal = no_goal

        self._step = 0
        self.state = np.zeros(4, dtype=np.float32)  # (x, y, vx, vy)

        self.action_space = spaces.Box(
            low=-max_accel,
            high=max_accel,
            shape=(2,),
            dtype=np.float32,
        )
        # Full obs: (x, y, vx, vy)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.np_random = np.random.default_rng(seed)
        
        # For rendering
        self.fig = None
        self.ax = None
        self.render_mode = render_mode

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clip_action_l2(action: np.ndarray, max_norm: float) -> np.ndarray:
        """Project action onto the L2 ball of radius max_norm (SO(2)-covariant)."""
        a = np.asarray(action, dtype=np.float32).reshape(2,)
        n = float(np.linalg.norm(a))
        if n <= max_norm or n == 0.0:
            return a
        return (a * (max_norm / (n + 1e-8))).astype(np.float32)

    def _random_point_in_circle(self, min_r: float = 0.1) -> np.ndarray:
        """Sample uniformly from the annulus [min_r, radius]."""
        angle = self.np_random.uniform(0, 2 * np.pi)
        r = np.sqrt(self.np_random.uniform(min_r**2, self.radius**2))
        return np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return self.state.copy()

    def _at_target(self) -> bool:
        return float(np.linalg.norm(self.state[:2])) <= self.target_radius

    # ── Core gym interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._step = 0

        # Spawn away from origin so it's not trivially solved
        self.state[:2] = self._random_point_in_circle(min_r=self.target_radius * 2)

        # Small random initial velocity
        speed = self.np_random.uniform(0, self.max_speed * 0.3)
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.state[2] = float(speed * np.cos(angle))
        self.state[3] = float(speed * np.sin(angle))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_accel, self.max_accel).astype(np.float32)
        action = self._clip_action_l2(action, self.max_action_norm)

        x, y, vx, vy = self.state

        # Net acceleration: control + drag
        # ax = action[0] - self.drag * vx
        # ay = action[1] - self.drag * vy
        v_norm = np.sqrt(vx**2 + vy**2)
        ax = action[0] - self.drag * vx * v_norm
        ay = action[1] - self.drag * vy * v_norm

        # Symplectic Euler integration
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt

        # Speed clipping (preserves direction → SO(2)-covariant)
        speed = float(np.sqrt(vx_new**2 + vy_new**2))
        if speed > self.max_speed:
            vx_new = vx_new * self.max_speed / speed
            vy_new = vy_new * self.max_speed / speed

        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt

        # ── Elastic reflection off circle boundary ────────────────────────────
        # Reflection is SO(2)-covariant: rotating the scene then reflecting =
        # reflecting then rotating the result.
        pos_new = np.array([x_new, y_new])
        dist = float(np.linalg.norm(pos_new))

        if dist > self.radius:
            # Inward-pointing normal
            normal = -pos_new / (dist + 1e-8)
            v_new = np.array([vx_new, vy_new])
            # Specular reflection: v → v - 2(v·n̂)n̂
            v_new = v_new - 2.0 * float(np.dot(v_new, normal)) * normal
            vx_new, vy_new = float(v_new[0]), float(v_new[1])
            pos_new = pos_new * self.radius / (dist + 1e-8)
            x_new, y_new = float(pos_new[0]), float(pos_new[1])

        self.state = np.array([x_new, y_new, vx_new, vy_new], dtype=np.float32)
        self._step += 1

        # ── Reward ────────────────────────────────────────────────────────────
        dist_to_target = float(np.linalg.norm(self.state[:2]))
        reward = -dist_to_target
        # Reward — zero in no_goal mode
        if self.no_goal:
            reward = 0.0
            terminated = False
        else:
            dist_to_target = float(np.linalg.norm(self.state[:2]))
            reward = -dist_to_target
            if self._at_target():
                reward += 10.0
            terminated = bool(self._at_target())

        terminated = bool(self._at_target())
        truncated = bool(self._step >= self.max_episode_steps)

        return self._get_obs(), reward, terminated, truncated, {}


    def render(self):
        if self.render_mode is None:
            return None
        
        if self.fig is None:
            # 6.4 * 100 dpi = 640 pixels (divisible by 16)
            self.fig, self.ax = plt.subplots(figsize=(6.4, 6.4))
        self.ax.clear()
        
        # Draw circle boundary
        self.ax.add_patch(Circle((0, 0), self.radius, fill=False, edgecolor='black', linewidth=2))        

        # Draw particle
        x, y, vx, vy = self.state
        self.ax.add_patch(Circle((x, y), 0.03, color='blue'))
        
        # Draw velocity vector
        # arrow = FancyArrowPatch((x, y), (x + vx, y + vy),
        arrow = FancyArrowPatch((x, y), (x + vx*0.2, y + vy*0.2),
                               arrowstyle='->', mutation_scale=10,
                               linewidth=2, color='red', alpha=0.7)
        self.ax.add_patch(arrow)
        
        self.ax.set_xlim(-1.01, 1.01)
        self.ax.set_ylim(-1.01, 1.01)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'ParticleCircle2D (step {self._step})', fontweight='bold')

        # add dashed lines for x and y axes
        self.ax.plot([-1, 1], [0, 0], 'k--', linewidth=1)
        self.ax.plot([0, 0], [-1, 1], 'k--', linewidth=1)
        
        self.fig.canvas.draw()
        
        if self.render_mode == 'rgb_array':
            # Ensure dimensions are divisible by 16 for video encoding
            width, height = self.fig.canvas.get_width_height()
            # Round up to nearest multiple of 16
            width = ((width + 15) // 16) * 16
            height = ((height + 15) // 16) * 16
            self.fig.set_size_inches(width/100, height/100)
            
            img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, :3].copy()  # IMPORTANT: .copy() creates a new array!
            
            return img
        
        plt.pause(0.01)
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


    # ── Symmetry utilities (for verification / data augmentation) ────────────

    @staticmethod
    def rotation_matrix(theta: float) -> np.ndarray:
        """2×2 SO(2) rotation matrix."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=np.float32)

    def apply_so2(self, obs: np.ndarray, action: np.ndarray, theta: float):
        """
        Apply SO(2) rotation by theta to (obs, action).
        obs layout: (x, y, vx, vy)
        action layout: (ax, ay)
        Returns: (rotated_obs, rotated_action)
        """
        R = self.rotation_matrix(theta)
        obs_r = np.concatenate([R @ obs[:2], R @ obs[2:4]])
        act_r = R @ action
        return obs_r.astype(np.float32), act_r.astype(np.float32)

    def verify_so2(self, n_samples: int = 100, theta: float = np.pi / 3) -> dict:
        """
        Empirically verify SO(2) symmetry of the dynamics.
        Checks that rotating obs+action and stepping = stepping then rotating.
        Returns dict of max errors (should all be ~0).
        """
        obs_errors = []
        act_consistency = []

        for _ in range(n_samples):
            obs, _ = self.reset()
            action = self.action_space.sample()

            # Path 1: rotate obs, step, get next_obs_1
            obs_r, act_r = self.apply_so2(obs, action, theta)
            self.state = obs_r.copy()
            next_obs_r, _, _, _, _ = self.step(act_r)

            # Path 2: step with original, rotate result to get next_obs_2
            self.state = obs.copy()
            next_obs, _, _, _, _ = self.step(action)
            next_obs_rot, _ = self.apply_so2(next_obs, action, theta)

            obs_errors.append(np.linalg.norm(next_obs_r - next_obs_rot))

        return {
            'max_obs_error': float(np.max(obs_errors)),
            'mean_obs_error': float(np.mean(obs_errors)),
            'symmetry_holds': float(np.max(obs_errors)) < 1e-4,
        }


class POParticleCircle2DEnv(ParticleCircle2DEnv):
    """
    POMDP version: observe only position (x, y), not velocity.

    SO(2) symmetry still holds on the observable:
      R_θ · (x, y) → same partial obs → R_θ · (ax, ay)
    (though the policy must implicitly infer velocity from history)
    """

    def __init__(self,  **kwargs):  # ← same fix
        super().__init__(**kwargs)
        # Override observation space: only (x, y)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        return self.state[:2].copy()


class POStackedParticleCircle2DEnv(ParticleCircle2DEnv):
    """
    POMDP + frame stacking: observe last `stack_size` positions.
    Obs shape: (stack_size * 2,) = stacked (x, y) pairs.

    SO(2) symmetry holds on each (x_t, y_t) pair independently,
    so the full stacked obs transforms as a direct sum of stack_size
    copies of the 2D SO(2) representation.
    Recommended lie_repr: '({stack_size}, 1, 2)'
    """

    def __init__(self, stack_size: int = 4, **kwargs):  # ← same fix
        super().__init__(**kwargs)
        self.stack_size = stack_size
        single_obs_shape = (1,) if hasattr(self, 'poidx') and len(self.poidx) == 1 else self.observation_space.shape
        stacked_shape = (single_obs_shape[0] * stack_size,)
    
        self._obs_buffer = np.zeros((stack_size, 2), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(stack_size * 2,),
            dtype=np.float32,
        )
        self.frame_stack = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Initialize stack with repeated first observation
        self.frame_stack = np.tile(obs, self.stack_size)
        return self.frame_stack.copy(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Shift stack and add new observation
        self.frame_stack = np.roll(self.frame_stack, -len(obs))
        self.frame_stack[-len(obs):] = obs
        return self.frame_stack.copy(), reward, terminated, truncated, info