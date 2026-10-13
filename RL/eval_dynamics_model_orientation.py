"""Evaluate a trained forward-dynamics model f(s,a)->delta_s at held-out
orientations phi, using RotatedReacherWrapper to co-rotate the arm-start and
target on reset. Two metrics:
  - One-step prediction MSE at each phi (headline: level AND slope together).
  - Open-loop multi-step rollout MSE for context (same action sequence fed to
    both the true env and the model's own autoregressive delta-chaining).
"""
import argparse
import hashlib
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp

from train_dynamics_model import DynamicsMLP


class RotatedReacherWrapper(gym.Wrapper):
    """Adds a fixed extra rotation phi (radians) to the arm-base angle and
    target position on every reset, on top of the standard reset_model()."""

    def __init__(self, env, phi):
        super().__init__(env)
        self.phi = phi

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.phi == 0.0:
            return obs, info
        unwrapped = self.env.unwrapped
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        qpos[0] += self.phi
        c, s = np.cos(self.phi), np.sin(self.phi)
        tx, ty = qpos[2], qpos[3]
        qpos[2] = c * tx - s * ty
        qpos[3] = s * tx + c * ty
        unwrapped.set_state(qpos, qvel)
        return unwrapped._get_obs(), info


def load_model(model_path):
    with open(model_path, 'rb') as f:
        d = pickle.load(f)
    model_def = DynamicsMLP(d['hidden_dims'], d['obs_dim'])
    params = jax.tree_util.tree_unflatten(d['params_tree_def'], [np.asarray(p) for p in d['params_arrays']])
    return model_def, params


def load_expert_action_episodes(dataset_path):
    with open(dataset_path, 'rb') as f:
        d = pickle.load(f)
    return [np.asarray(ep['actions'], dtype=np.float32) for ep in d['episodes']]


# Actions here are ALWAYS replayed from real expert episodes (never uniform-random):
# the true symmetry has G_a == 0 (joint-space torques aren't rotated by this SO(2)),
# so replaying unrotated expert action sequences at a rotated state keeps actions
# in-distribution while still testing a genuinely novel orientation.
def evaluate_onestep(model_def, params, phi_deg, num_transitions, seed, action_episodes):
    base_env = gym.make('Reacher-v5')
    env = RotatedReacherWrapper(base_env, np.radians(phi_deg))
    rng = np.random.default_rng(seed)
    sq_errs = []
    collected = 0
    while collected < num_transitions:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_actions = action_episodes[rng.integers(0, len(action_episodes))]
        for t in range(len(ep_actions)):
            action = ep_actions[t]
            next_obs, _, terminated, truncated, _ = env.step(action)
            ds_true = next_obs - obs
            pred = model_def.apply({'params': params}, jnp.asarray(obs[None]), jnp.asarray(action[None]))
            sq_err = float(jnp.sum((pred[0] - ds_true) ** 2))
            sq_errs.append(sq_err)
            collected += 1
            obs = next_obs
            if terminated or truncated or collected >= num_transitions:
                break
    env.close()
    return float(np.mean(sq_errs)), float(np.std(sq_errs))


def load_fixed_rollout_data(path):
    """Fixed (reset_seed, action_seq) pairs, IDENTICAL across every condition/model
    evaluated -- no policy in the loop, no per-checkpoint resampling. This is what
    makes rollout-MSE differences attributable purely to each model's own error
    compounding, not to which action sequence happened to get sampled."""
    d = np.load(path)
    action_seqs = d['action_seqs']
    reset_seeds = d['reset_seeds']
    h = hashlib.sha256(action_seqs.tobytes()).hexdigest()[:16]
    print(f"[fixed rollout data] {path}  action_seqs={action_seqs.shape}  hash={h}")
    return action_seqs, reset_seeds


def evaluate_rollout(model_def, params, phi_deg, rollout_len, action_seqs, reset_seeds):
    base_env = gym.make('Reacher-v5')
    env = RotatedReacherWrapper(base_env, np.radians(phi_deg))
    final_sq_errs = []
    cumulative_sq_errs = []
    for ep in range(len(action_seqs)):
        obs_true, _ = env.reset(seed=int(reset_seeds[ep]))
        obs_model = obs_true.copy()
        cum_err = 0.0
        ep_actions = action_seqs[ep]
        n_steps = min(rollout_len, len(ep_actions))
        for t in range(n_steps):
            action = ep_actions[t]
            next_obs_true, _, terminated, truncated, _ = env.step(action)
            pred = model_def.apply({'params': params}, jnp.asarray(obs_model[None]), jnp.asarray(action[None]))
            obs_model = obs_model + np.asarray(pred[0])
            cum_err += float(np.sum((obs_model - next_obs_true) ** 2))
            obs_true = next_obs_true
            if terminated or truncated:
                break
        final_sq_errs.append(float(np.sum((obs_model - obs_true) ** 2)))
        cumulative_sq_errs.append(cum_err)
    env.close()
    return float(np.mean(final_sq_errs)), float(np.mean(cumulative_sq_errs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default='./datasets/reacher_expert.pkl')
    parser.add_argument('--fixed_rollout_data', type=str,
                         default='./datasets/orbit_aug_experiment/fixed_rollout_eval_data.npz')
    parser.add_argument('--num_transitions', type=int, default=2000)
    parser.add_argument('--rollout_len', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    model_def, params = load_model(args.model_path)
    action_episodes = load_expert_action_episodes(args.dataset_path)
    action_seqs, reset_seeds = load_fixed_rollout_data(args.fixed_rollout_data)

    phis = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    print(f"{'phi(deg)':>10} {'onestep_MSE':>14} {'onestep_std':>14} {'rollout_final_MSE':>20} {'rollout_cum_MSE':>18}")
    for phi in phis:
        mse, std = evaluate_onestep(model_def, params, phi, args.num_transitions, args.seed, action_episodes)
        rf, rc = evaluate_rollout(model_def, params, phi, args.rollout_len, action_seqs, reset_seeds)
        print(f"{phi:>10d} {mse:>14.6f} {std:>14.6f} {rf:>20.6f} {rc:>18.6f}")
