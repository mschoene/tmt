"""
Generate a random-action dataset from ParticleCircle2DEnv in no_goal mode.
No reward signal, no termination at origin — pure dynamics data.

Saved as a .pkl file matching the format of create_dataset.py:
  episodes: list of dicts with keys:
    observations  : (T+1, 4)  — (x, y, vx, vy), includes initial obs
    actions       : (T, 2)    — (ax, ay)
    rewards       : (T,)      — all zeros (no goal)
    terminals     : (T,)      — bool
    truncations   : (T,)      — bool
    infos         : list of dicts
  metadata: dict with dataset info
"""

import argparse
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))
from jax_rl.envs.particleCircle2D import ParticleCircle2DEnv
from jax_rl.envs.particleCircle3D import ParticleCircle3DEnv


def generate(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_random_uniform.pkl',
    drag: float = 0.0,
):
    env = ParticleCircle2DEnv(
        no_goal=True,
        max_episode_steps=max_episode_steps,
        seed=seed,
        drag=drag,
    )

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps...")

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            # Use env.np_random (reseeded per episode) so same seed → same dataset
            action = env.np_random.uniform(-env.max_accel, env.max_accel, size=2).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps collected "
                  f"({episodes_collected} episodes)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': episodes_collected,
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random',
            'drag': drag,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")
    print(f"  Drag:        {drag}")

    return dataset


def generate_drag_sweep(
    drag_values: list = None,
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_dir: str = './datasets',
):
    """Generate one dataset per drag value for non-linearity sweep."""
    if drag_values is None:
        drag_values = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        # drag_values = [5.0, 10.0]

    import shutil

    canonical_unbiased = os.path.join(output_dir, 'particle_circle_2d_random_uniform.pkl')

    paths = []
    for drag in drag_values:
        out = os.path.join(output_dir, f'particle_circle_2d_drag{drag:.2f}.pkl')
        if drag == 0.0 and os.path.exists(canonical_unbiased):
            # drag=0 is identical to the unbiased dataset — reuse it to guarantee same data
            print(f"\n=== drag=0.0: copying canonical unbiased dataset → {out} ===")
            shutil.copy2(canonical_unbiased, out)
        else:
            print(f"\n=== Generating drag={drag:.2f} → {out} ===")
            generate(
                num_steps=num_steps,
                max_episode_steps=max_episode_steps,
                seed=seed,
                output_path=out,
                drag=drag,
            )
        paths.append(out)

    print(f"\nDrag sweep complete. {len(paths)} datasets written:")
    for p in paths:
        print(f"  {p}")
    return paths


def generate_3d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_3d_random_uniform.pkl',
):
    env = ParticleCircle3DEnv(
        no_goal=True,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (3D)...")

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            n_act = env.action_space.shape[0]
            action = env.np_random.uniform(-env.max_accel, env.max_accel, size=n_act).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps collected "
                  f"({episodes_collected} episodes)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle3D-v0',
            'seed': seed,
            'num_episodes': episodes_collected,
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")

    return dataset


def generate_state_biased_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_state_positive.pkl',
):
    """Collect data where x >= 0 and y >= 0 throughout. Episodes truncate on boundary exit."""
    env = ParticleCircle2DEnv(
        no_goal=True,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (state-biased: x>=0, y>=0)...")

    while steps_collected < num_steps:
        # Resample until initial position falls in positive quadrant
        for attempt in range(10_000):
            obs, _ = env.reset(seed=seed + episodes_collected * 997 + attempt)
            if obs[0] >= 0.0 and obs[1] >= 0.0:
                break
        else:
            raise RuntimeError("Could not find positive-quadrant initial state after 10k attempts")

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            action = env.np_random.uniform(-env.max_accel, env.max_accel, size=2).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if next_obs[0] < 0.0 or next_obs[1] < 0.0:
                # State left positive quadrant — truncate without recording this step
                done = True
                break

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        if len(ep_actions) > 0:
            episodes.append({
                'observations': np.array(ep_observations, dtype=np.float32),
                'actions':      np.array(ep_actions,      dtype=np.float32),
                'rewards':      np.array(ep_rewards,       dtype=np.float32),
                'terminals':    np.array(ep_terminals,     dtype=bool),
                'truncations':  np.array(ep_truncations,   dtype=bool),
                'infos':        ep_infos,
            })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps collected "
                  f"({episodes_collected} episodes, {len(episodes)} non-empty)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': len(episodes),
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random',
            'bias': 'state_positive_quadrant',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")

    return dataset


def generate_state_action_biased_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_state_action_positive.pkl',
):
    """Collect data where x>=0, y>=0 AND ax>=0, ay>=0 throughout."""
    env = ParticleCircle2DEnv(
        no_goal=True,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 2)

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (state+action biased: x>=0, y>=0, ax>=0, ay>=0)...")

    while steps_collected < num_steps:
        for attempt in range(10_000):
            obs, _ = env.reset(seed=seed + episodes_collected * 997 + attempt)
            if obs[0] >= 0.0 and obs[1] >= 0.0:
                break
        else:
            raise RuntimeError("Could not find positive-quadrant initial state after 10k attempts")

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            action = rng.uniform(0.0, env.max_accel, size=2).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if next_obs[0] < 0.0 or next_obs[1] < 0.0:
                done = True
                break

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        if len(ep_actions) > 0:
            episodes.append({
                'observations': np.array(ep_observations, dtype=np.float32),
                'actions':      np.array(ep_actions,      dtype=np.float32),
                'rewards':      np.array(ep_rewards,       dtype=np.float32),
                'terminals':    np.array(ep_terminals,     dtype=bool),
                'truncations':  np.array(ep_truncations,   dtype=bool),
                'infos':        ep_infos,
            })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps collected "
                  f"({episodes_collected} episodes, {len(episodes)} non-empty)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': len(episodes),
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random_positive_quadrant',
            'bias': 'state_and_action_positive_quadrant',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")

    return dataset


def generate_action_biased_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_action_positive.pkl',
):
    """Collect data with actions restricted to ax >= 0, ay >= 0 (positive quadrant)."""
    env = ParticleCircle2DEnv(
        no_goal=True,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (action-biased: ax>=0, ay>=0)...")

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            action = rng.uniform(0.0, env.max_accel, size=2).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps collected "
                  f"({episodes_collected} episodes)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': episodes_collected,
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random_positive_quadrant',
            'bias': 'action_positive_quadrant',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")

    return dataset


def generate_action_positive_flat(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_action_positive_flat.pkl',
):
    """
    Episode-based dataset with actions restricted to ax>=0, ay>=0.

    Same structure as generate_action_biased_2d but records the actual action
    applied by the env (after its internal L2 clip) instead of the pre-clip sample.
    No state-positivity constraint — episodes run to natural termination.
    """
    env = ParticleCircle2DEnv(
        no_goal=True,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (action-positive, episode format)...")

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            a_raw = rng.uniform(0.0, env.max_accel, size=2).astype(np.float32)

            # Replicate env's clipping to record the actual applied action
            a_clipped = np.clip(a_raw, -env.max_accel, env.max_accel)
            norm = float(np.linalg.norm(a_clipped))
            if norm > env.max_action_norm and norm > 0:
                a_clipped = (a_clipped * env.max_action_norm / norm).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(a_raw)

            ep_actions.append(a_clipped)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps collected "
                  f"({episodes_collected} episodes)")

    env.close()

    all_actions = np.concatenate([ep['actions'] for ep in episodes])
    assert np.all(all_actions >= 0), "BUG: some recorded actions are negative"

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': episodes_collected,
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random_positive_quadrant',
            'bias': 'action_positive_quadrant',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {episodes_collected}")
    print(f"  Total steps: {steps_collected}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")
    print(f"  Action min:  {all_actions.min(axis=0)}")
    print(f"  Action max:  {all_actions.max(axis=0)}")

    return dataset


def generate_state_negative_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_state_negative.pkl',
):
    """Collect data where state stays OUTSIDE the positive quadrant (x<0 OR y<0).

    Episodes start from a non-positive-quadrant initial state and are truncated
    the moment the state would enter the positive quadrant (x>=0 AND y>=0).
    Actions are unrestricted.  This is the complement of generate_state_biased_2d.
    """
    env = ParticleCircle2DEnv(no_goal=True, max_episode_steps=max_episode_steps, seed=seed)

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (state-negative: x<0 OR y<0)...")

    while steps_collected < num_steps:
        # Resample until initial state is outside positive quadrant
        for attempt in range(10_000):
            obs, _ = env.reset(seed=seed + episodes_collected * 997 + attempt)
            if obs[0] < 0.0 or obs[1] < 0.0:
                break
        else:
            raise RuntimeError("Could not find non-positive-quadrant initial state after 10k attempts")

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            action = env.np_random.uniform(-env.max_accel, env.max_accel, size=2).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if next_obs[0] >= 0.0 and next_obs[1] >= 0.0:
                # State entered positive quadrant — truncate without recording
                done = True
                break

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        if len(ep_actions) > 0:
            episodes.append({
                'observations': np.array(ep_observations, dtype=np.float32),
                'actions':      np.array(ep_actions,      dtype=np.float32),
                'rewards':      np.array(ep_rewards,       dtype=np.float32),
                'terminals':    np.array(ep_terminals,     dtype=bool),
                'truncations':  np.array(ep_truncations,   dtype=bool),
                'infos':        ep_infos,
            })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes, {len(episodes)} non-empty)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': len(episodes),
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random',
            'bias': 'state_negative_quadrant',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    return dataset


def generate_action_negative_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_action_negative.pkl',
):
    """Collect data where every action has ax<0 OR ay<0 (outside positive-action quadrant).

    Actions are rejection-sampled from [-max_accel, max_accel]^2 excluding the
    positive-action quadrant (ax>=0 AND ay>=0).  States are unrestricted.
    This is the complement of generate_action_biased_2d.
    """
    env = ParticleCircle2DEnv(no_goal=True, max_episode_steps=max_episode_steps, seed=seed)
    rng = np.random.default_rng(seed + 11)

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (action-negative: ax<0 OR ay<0)...")

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            # Rejection-sample: keep only actions NOT in positive quadrant
            for _ in range(10_000):
                a = rng.uniform(-env.max_accel, env.max_accel, size=2).astype(np.float32)
                if not (a[0] >= 0.0 and a[1] >= 0.0):
                    break
            norm = float(np.linalg.norm(a))
            if norm > env.max_action_norm:
                a = (a * (env.max_action_norm / norm)).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(a)

            ep_actions.append(a)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': episodes_collected,
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random_negative_action_quadrant',
            'bias': 'action_negative_quadrant',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    return dataset


def generate_state_action_negative_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_state_action_negative.pkl',
):
    """Collect data where state is outside positive quadrant AND action is outside positive quadrant.

    Every (s, a, s') triple satisfies:
      - state:  x<0 OR y<0   (NOT in positive-state quadrant)
      - action: ax<0 OR ay<0 (NOT in positive-action quadrant)

    This is the complement of generate_state_action_biased_2d and guarantees no
    overlap with that dataset.
    """
    env = ParticleCircle2DEnv(no_goal=True, max_episode_steps=max_episode_steps, seed=seed)
    rng = np.random.default_rng(seed + 22)

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps (state-action-negative: (x<0 OR y<0) AND (ax<0 OR ay<0))...")

    while steps_collected < num_steps:
        # Start in non-positive-state quadrant
        for attempt in range(10_000):
            obs, _ = env.reset(seed=seed + episodes_collected * 997 + attempt)
            if obs[0] < 0.0 or obs[1] < 0.0:
                break
        else:
            raise RuntimeError("Could not find non-positive-quadrant initial state after 10k attempts")

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            # Rejection-sample action NOT in positive quadrant
            for _ in range(10_000):
                a = rng.uniform(-env.max_accel, env.max_accel, size=2).astype(np.float32)
                if not (a[0] >= 0.0 and a[1] >= 0.0):
                    break
            norm = float(np.linalg.norm(a))
            if norm > env.max_action_norm:
                a = (a * (env.max_action_norm / norm)).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(a)

            if next_obs[0] >= 0.0 and next_obs[1] >= 0.0:
                # State entered positive quadrant — truncate
                done = True
                break

            ep_actions.append(a)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        if len(ep_actions) > 0:
            episodes.append({
                'observations': np.array(ep_observations, dtype=np.float32),
                'actions':      np.array(ep_actions,      dtype=np.float32),
                'rewards':      np.array(ep_rewards,       dtype=np.float32),
                'terminals':    np.array(ep_terminals,     dtype=bool),
                'truncations':  np.array(ep_truncations,   dtype=bool),
                'infos':        ep_infos,
            })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes, {len(episodes)} non-empty)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'ParticleCircle2D-v0',
            'seed': seed,
            'num_episodes': len(episodes),
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'no_goal': True,
            'policy': 'random_negative_quadrant',
            'bias': 'state_and_action_negative_quadrant',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {dataset['metadata']['num_episodes']}")
    print(f"  Total steps: {dataset['metadata']['total_steps']}")
    return dataset


def generate_trifinger(
    num_steps: int = 100_000,
    max_episode_steps: int = 300,
    seed: int = 42,
    output_path: str = './datasets/trifinger_random_uniform.pkl',
):
    """Generate a random-torque dataset from TriFingerPlatform (trifinger-simulation).

    State: pos(9) + vel(9) + torque(9) + tip_force(3) = 30 dims.
    Action: torque(9), normalised to [-1, 1] by dividing by max_torque (0.397 Nm).
    """
    try:
        from trifinger_simulation import TriFingerPlatform, trifingerpro_limits
    except ImportError as e:
        raise ImportError(
            "trifinger-simulation is not installed. "
            "Run: pip install trifinger-simulation"
        ) from e

    rng = np.random.default_rng(seed)
    max_torque = trifingerpro_limits.max_torque_Nm  # 0.397

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps from TriFingerPlatform (random torques)...")

    while steps_collected < num_steps:
        platform = TriFingerPlatform(visualization=False)

        ep_observations = []
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        # Initial observation (zero-torque step)
        t = platform.append_desired_action(
            platform.Action(torque=np.zeros(9), position=np.full(9, np.nan))
        )
        robot_obs = platform.get_robot_observation(t)
        obs = np.concatenate([
            robot_obs.position, robot_obs.velocity,
            robot_obs.torque, robot_obs.tip_force,
        ]).astype(np.float32)
        ep_observations.append(obs)

        ep_steps = 0
        while ep_steps < max_episode_steps and steps_collected < num_steps:
            torques = rng.uniform(-max_torque, max_torque, size=9).astype(np.float32)
            action_normalised = (torques / max_torque).astype(np.float32)

            t = platform.append_desired_action(
                platform.Action(torque=torques, position=np.full(9, np.nan))
            )
            robot_obs = platform.get_robot_observation(t)
            next_obs = np.concatenate([
                robot_obs.position, robot_obs.velocity,
                robot_obs.torque, robot_obs.tip_force,
            ]).astype(np.float32)

            ep_observations.append(next_obs)
            ep_actions.append(action_normalised)
            ep_rewards.append(0.0)
            ep_terminals.append(False)
            ep_truncations.append(ep_steps == max_episode_steps - 1)
            ep_infos.append({})

            ep_steps += 1
            steps_collected += 1

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes)")

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'TriFingerPro-v0',
            'seed': seed,
            'num_episodes': episodes_collected,
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'policy': 'random',
            'state_dim': 30,
            'action_dim': 9,
            'action_normalisation': f'torque / {max_torque:.4f} Nm',
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {episodes_collected}")
    print(f"  Total steps: {steps_collected}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")

    return dataset


def generate_trifinger_position(
    num_steps: int = 100_000,
    max_episode_steps: int = 300,
    seed: int = 42,
    output_path: str = './datasets/trifinger_position_random_uniform.pkl',
):
    """Generate a random-position-command dataset from TriFingerPlatform.

    State: pos(9) + vel(9) + torque_measured(9) + tip_force(3) = 30 dims.
    Action: target joint positions (9-dim), normalised to [-1, 1] per joint
    using (pos - center) / half_range where center/half_range are derived from
    the per-joint position limits.  Limits are identical across fingers, so the
    Z(3) symmetry is preserved.
    """
    try:
        from trifinger_simulation import TriFingerPlatform, trifingerpro_limits
    except ImportError as e:
        raise ImportError(
            "trifinger-simulation is not installed. "
            "Run: pip install trifinger-simulation"
        ) from e

    rng = np.random.default_rng(seed)
    pos_lo  = trifingerpro_limits.robot_position.low.astype(np.float32)   # (9,)
    pos_hi  = trifingerpro_limits.robot_position.high.astype(np.float32)
    pos_center     = (pos_lo + pos_hi) / 2.0
    pos_half_range = (pos_hi - pos_lo) / 2.0

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps from TriFingerPlatform (random positions)...")

    while steps_collected < num_steps:
        platform = TriFingerPlatform(visualization=False)

        ep_observations = []
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        # Initial observation at rest position
        t = platform.append_desired_action(
            platform.Action(torque=np.zeros(9), position=pos_center)
        )
        robot_obs = platform.get_robot_observation(t)
        obs = np.concatenate([
            robot_obs.position, robot_obs.velocity,
            robot_obs.torque, robot_obs.tip_force,
        ]).astype(np.float32)
        ep_observations.append(obs)

        ep_steps = 0
        while ep_steps < max_episode_steps and steps_collected < num_steps:
            positions = rng.uniform(pos_lo, pos_hi).astype(np.float32)
            action_normalised = ((positions - pos_center) / pos_half_range).astype(np.float32)

            t = platform.append_desired_action(
                platform.Action(torque=np.zeros(9), position=positions)
            )
            robot_obs = platform.get_robot_observation(t)
            next_obs = np.concatenate([
                robot_obs.position, robot_obs.velocity,
                robot_obs.torque, robot_obs.tip_force,
            ]).astype(np.float32)

            ep_observations.append(next_obs)
            ep_actions.append(action_normalised)
            ep_rewards.append(0.0)
            ep_terminals.append(False)
            ep_truncations.append(ep_steps == max_episode_steps - 1)
            ep_infos.append({})

            ep_steps += 1
            steps_collected += 1

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes)")

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name': 'TriFingerPro-position-v0',
            'seed': seed,
            'num_episodes': episodes_collected,
            'total_steps': steps_collected,
            'max_episode_steps': max_episode_steps,
            'policy': 'random',
            'state_dim': 30,
            'action_dim': 9,
            'action_normalisation': 'position: (pos - center) / half_range per joint',
            'pos_center': pos_center.tolist(),
            'pos_half_range': pos_half_range.tolist(),
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {episodes_collected}")
    print(f"  Total steps: {steps_collected}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")

    return dataset


def generate_directional_biased_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_dir_biased.pkl',
    bias_prob: float = 0.9,
    bias_angle_deg: float = 45.0,
    cone_halfwidth_deg: float = 45.0,
    drag: float = 0.0,
):
    """Collect data with a probabilistically biased action policy.

    With probability bias_prob the action is sampled uniformly from a cone of
    half-width cone_halfwidth_deg centred on bias_angle_deg; otherwise uniform
    over all directions.  No state constraint.

    bias_prob=1, angle=45, halfwidth=45 ≈ hard positive-quadrant constraint.
    bias_prob=0 ≈ fully random generate().
    """
    env = ParticleCircle2DEnv(no_goal=True, max_episode_steps=max_episode_steps,
                               seed=seed, drag=drag)
    rng = np.random.default_rng(seed + 200)

    bias_angle = np.deg2rad(bias_angle_deg)
    cone_hw    = np.deg2rad(cone_halfwidth_deg)

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    print(f"Collecting {num_steps} steps "
          f"(dir-biased: prob={bias_prob}, angle={bias_angle_deg}°, cone±{cone_halfwidth_deg}°)...")

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        ep_observations = [obs]
        ep_actions      = []
        ep_rewards      = []
        ep_terminals    = []
        ep_truncations  = []
        ep_infos        = []

        done = False
        while not done and steps_collected < num_steps:
            if rng.random() < bias_prob:
                a_angle = rng.uniform(bias_angle - cone_hw, bias_angle + cone_hw)
                a_mag   = rng.uniform(0.0, env.max_action_norm)
                action  = np.array([np.cos(a_angle) * a_mag,
                                    np.sin(a_angle) * a_mag], dtype=np.float32)
            else:
                action = rng.uniform(-env.max_accel, env.max_accel, size=2).astype(np.float32)
                norm = float(np.linalg.norm(action))
                if norm > env.max_action_norm:
                    action = (action * (env.max_action_norm / norm)).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,      dtype=np.float32),
            'terminals':    np.array(ep_terminals,    dtype=bool),
            'truncations':  np.array(ep_truncations,  dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name':           'ParticleCircle2D-v0',
            'seed':               seed,
            'num_episodes':       episodes_collected,
            'total_steps':        steps_collected,
            'max_episode_steps':  max_episode_steps,
            'no_goal':            True,
            'policy':             'directional_biased',
            'bias_prob':          bias_prob,
            'bias_angle_deg':     bias_angle_deg,
            'cone_halfwidth_deg': cone_halfwidth_deg,
            'drag':               drag,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {episodes_collected}")
    print(f"  Total steps: {steps_collected}")
    print(f"  Obs shape:   {episodes[0]['observations'].shape}")
    print(f"  Action shape:{episodes[0]['actions'].shape}")
    return dataset


def generate_von_mises_2d(
    num_steps: int = 100_000,
    max_episode_steps: int = 10,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_2d_vonmises_k10.pkl',
    kappa: float = 10.0,
):
    """ParticleCircle2D with initial position angle ~ vonMises(0, kappa).

    Large kappa = concentrated near theta=0 (sigma ≈ 1/sqrt(kappa) rad).
    Short episodes (default 10 steps) prevent the particle from diffusing
    away from the concentrated region before the episode ends.
    Actions are uniformly random — no action bias — so the only asymmetry is
    in the coverage of the SO(2) orbit, not in the dynamics.
    kappa=0 falls back to the standard uniform distribution.
    """
    env = ParticleCircle2DEnv(no_goal=True, max_episode_steps=max_episode_steps, seed=seed)
    rng = np.random.default_rng(seed + 500)

    sigma_deg = float(np.degrees(1.0 / np.sqrt(kappa))) if kappa > 0 else float('inf')
    sigma_str = f"{sigma_deg:.1f}°" if np.isfinite(sigma_deg) else "uniform"
    print(f"Collecting {num_steps} steps "
          f"(von Mises: kappa={kappa}, sigma≈{sigma_str}, ep_steps={max_episode_steps})...")

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        # Override initial position angle; keep the radius sampled by env.reset
        r = float(np.sqrt(env.state[0] ** 2 + env.state[1] ** 2))
        angle = rng.vonmises(0.0, kappa) if kappa > 0 else rng.uniform(-np.pi, np.pi)
        env.state[0] = float(r * np.cos(angle))
        env.state[1] = float(r * np.sin(angle))
        obs = env._get_obs()

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            action = env.np_random.uniform(-env.max_accel, env.max_accel, size=2).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name':           'ParticleCircle2D-v0',
            'seed':               seed,
            'num_episodes':       episodes_collected,
            'total_steps':        steps_collected,
            'max_episode_steps':  max_episode_steps,
            'no_goal':            True,
            'policy':             'random',
            'bias':               'von_mises_angle',
            'kappa':              kappa,
            'sigma_deg':          sigma_deg if np.isfinite(sigma_deg) else None,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {episodes_collected}")
    print(f"  Total steps: {steps_collected}")
    print(f"  kappa={kappa}, sigma≈{sigma_str}")
    return dataset


def generate_von_mises_3d(
    num_steps: int = 100_000,
    max_episode_steps: int = 10,
    seed: int = 42,
    output_path: str = './datasets/particle_circle_3d_vonmises_k10.pkl',
    kappa: float = 10.0,
):
    """ParticleCircle3D with initial position direction ~ vMF(north_pole, kappa).

    Uses the same angle concentration as the 2D case: polar angle theta ~ |vonMises(0, kappa)|,
    azimuthal angle phi ~ Uniform(0, 2pi).  For kappa=0 falls back to uniform on sphere.
    Short episodes (default 10 steps) keep the particle near the concentrated region.
    """
    from jax_rl.envs.particleCircle3D import ParticleCircle3DEnv
    env = ParticleCircle3DEnv(no_goal=True, max_episode_steps=max_episode_steps, seed=seed)
    rng = np.random.default_rng(seed + 500)

    sigma_deg = float(np.degrees(1.0 / np.sqrt(kappa))) if kappa > 0 else float('inf')
    sigma_str = f"{sigma_deg:.1f}deg" if np.isfinite(sigma_deg) else "uniform"
    print(f"Collecting {num_steps} steps "
          f"(3D von Mises: kappa={kappa}, sigma~{sigma_str}, ep_steps={max_episode_steps})...")

    episodes = []
    steps_collected = 0
    episodes_collected = 0

    while steps_collected < num_steps:
        obs, _ = env.reset(seed=seed + episodes_collected)

        # Override initial position direction; keep radius sampled by env.reset
        r = float(np.linalg.norm(env.state[:3]))
        if kappa > 0:
            theta = abs(rng.vonmises(0.0, kappa))   # polar angle concentrated near north pole
        else:
            theta = rng.uniform(0.0, np.pi)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        env.state[0] = float(r * np.sin(theta) * np.cos(phi))
        env.state[1] = float(r * np.sin(theta) * np.sin(phi))
        env.state[2] = float(r * np.cos(theta))
        obs = env._get_obs()

        ep_observations = [obs]
        ep_actions = []
        ep_rewards = []
        ep_terminals = []
        ep_truncations = []
        ep_infos = []

        done = False
        while not done and steps_collected < num_steps:
            n_act = env.action_space.shape[0]
            action = env.np_random.uniform(-env.max_accel, env.max_accel, size=n_act).astype(np.float32)
            norm = float(np.linalg.norm(action))
            if norm > env.max_action_norm:
                action = (action * (env.max_action_norm / norm)).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_observations.append(next_obs)
            ep_terminals.append(terminated)
            ep_truncations.append(truncated)
            ep_infos.append(info)

            obs = next_obs
            steps_collected += 1
            done = terminated or truncated

        episodes.append({
            'observations': np.array(ep_observations, dtype=np.float32),
            'actions':      np.array(ep_actions,      dtype=np.float32),
            'rewards':      np.array(ep_rewards,       dtype=np.float32),
            'terminals':    np.array(ep_terminals,     dtype=bool),
            'truncations':  np.array(ep_truncations,   dtype=bool),
            'infos':        ep_infos,
        })
        episodes_collected += 1

        if steps_collected % 10_000 == 0:
            print(f"  {steps_collected}/{num_steps} steps ({episodes_collected} episodes)")

    env.close()

    dataset = {
        'episodes': episodes,
        'metadata': {
            'env_name':           'ParticleCircle3D-v0',
            'seed':               seed,
            'num_episodes':       episodes_collected,
            'total_steps':        steps_collected,
            'max_episode_steps':  max_episode_steps,
            'no_goal':            True,
            'policy':             'random',
            'bias':               'von_mises_polar_angle',
            'kappa':              kappa,
            'sigma_deg':          sigma_deg if np.isfinite(sigma_deg) else None,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes:    {episodes_collected}")
    print(f"  Total steps: {steps_collected}")
    print(f"  kappa={kappa}, sigma~{sigma_str}")
    return dataset


def generate_von_mises_sweep(
    kappa_values: list = None,
    num_steps: int = 100_000,
    max_episode_steps: int = 10,
    seed: int = 42,
    output_dir: str = './datasets',
):
    """Generate one dataset per kappa value for the von Mises concentration sweep."""
    if kappa_values is None:
        kappa_values = [1, 2, 5, 10, 20, 50]

    paths = []
    for kappa in kappa_values:
        tag = f"{int(kappa)}" if float(kappa) == int(kappa) else f"{kappa}"
        out = os.path.join(output_dir, f'particle_circle_2d_vonmises_k{tag}.pkl')
        print(f"\n=== kappa={kappa} → {out} ===")
        generate_von_mises_2d(
            num_steps=num_steps,
            max_episode_steps=max_episode_steps,
            seed=seed,
            output_path=out,
            kappa=kappa,
        )
        paths.append(out)

    print(f"\nVon Mises sweep complete. {len(paths)} datasets written:")
    for p in paths:
        print(f"  {p}")
    return paths


def generate_all_complement(
    num_steps: int = 100_000,
    max_episode_steps: int = 200,
    seed: int = 42,
    output_dir: str = './datasets',
):
    """Generate all three complement (negative-quadrant) datasets in one call."""
    paths = []
    for fn, name in [
        (generate_state_negative_2d,        'particle_circle_2d_state_negative.pkl'),
        (generate_action_negative_2d,        'particle_circle_2d_action_negative.pkl'),
        (generate_state_action_negative_2d,  'particle_circle_2d_state_action_negative.pkl'),
    ]:
        out = os.path.join(output_dir, name)
        print(f"\n=== Generating {name} ===")
        fn(num_steps=num_steps, max_episode_steps=max_episode_steps, seed=seed, output_path=out)
        paths.append(out)

    print(f"\nAll complement datasets written:")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',               type=str,   default='2d',
                        choices=['2d', '3d', '2d_state_biased', '2d_action_biased',
                                 '2d_state_action_biased', '2d_action_positive_flat',
                                 '2d_drag_sweep',
                                 '2d_state_neg', '2d_action_neg', '2d_state_action_neg',
                                 '2d_all_complement', '2d_dir_biased',
                                 '2d_von_mises', '2d_von_mises_sweep',
                                 '3d_von_mises',
                                 'trifinger', 'trifinger_position'])
    parser.add_argument('--num_steps',         type=int,   default=100_000)
    parser.add_argument('--max_episode_steps', type=int,   default=200)
    parser.add_argument('--seed',              type=int,   default=42)
    parser.add_argument('--output_path',       type=str,   default=None)
    parser.add_argument('--drag',              type=float, default=0.0,
                        help='Drag coefficient for 2d env (non-linearity factor)')
    parser.add_argument('--drag_values',       type=float, nargs='+', default=None,
                        help='List of drag values for 2d_drag_sweep (default: 0 0.1 0.25 0.5 1 2)')
    parser.add_argument('--bias_prob',         type=float, default=0.9,
                        help='Probability of sampling from biased direction (2d_dir_biased)')
    parser.add_argument('--bias_angle_deg',    type=float, default=45.0,
                        help='Centre direction of action bias in degrees (0=east, 45=NE, 90=north)')
    parser.add_argument('--cone_halfwidth_deg', type=float, default=45.0,
                        help='Half-width of the bias cone in degrees (45=quadrant, 90=hemisphere)')
    parser.add_argument('--kappa',              type=float, default=10.0,
                        help='Von Mises concentration for 2d_von_mises (0=uniform, larger=more concentrated)')
    parser.add_argument('--kappa_values',       type=float, nargs='+', default=None,
                        help='List of kappa values for 2d_von_mises_sweep (default: 1 2 5 10 20 50)')
    args = parser.parse_args()

    if args.env == '3d':
        out = args.output_path or './datasets/particle_circle_3d_random_uniform.pkl'
        generate_3d(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
        )
    elif args.env == '2d_state_biased':
        out = args.output_path or './datasets/particle_circle_2d_state_positive.pkl'
        generate_state_biased_2d(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
        )
    elif args.env == '2d_state_action_biased':
        out = args.output_path or './datasets/particle_circle_2d_state_action_positive.pkl'
        generate_state_action_biased_2d(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
        )
    elif args.env == '2d_action_biased':
        out = args.output_path or './datasets/particle_circle_2d_action_positive.pkl'
        generate_action_biased_2d(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
        )
    elif args.env == '2d_action_positive_flat':
        out = args.output_path or './datasets/particle_circle_2d_action_positive_flat.pkl'
        generate_action_positive_flat(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
        )
    elif args.env == '2d_drag_sweep':
        out_dir = args.output_path or './datasets'
        generate_drag_sweep(
            drag_values=args.drag_values,
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_dir=out_dir,
        )
    elif args.env == '2d_dir_biased':
        tag = f'b{args.bias_prob:.3f}-a{args.bias_angle_deg:.0f}-c{args.cone_halfwidth_deg:.0f}'
        out = args.output_path or f'./datasets/particle_circle_2d_dir_biased_{tag}.pkl'
        generate_directional_biased_2d(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
            bias_prob=args.bias_prob,
            bias_angle_deg=args.bias_angle_deg,
            cone_halfwidth_deg=args.cone_halfwidth_deg,
            drag=args.drag,
        )
    elif args.env == '2d_von_mises':
        ep_steps = args.max_episode_steps if args.max_episode_steps != 200 else 10
        tag = f"{int(args.kappa)}" if float(args.kappa) == int(args.kappa) else f"{args.kappa}"
        out = args.output_path or f'./datasets/particle_circle_2d_vonmises_k{tag}.pkl'
        generate_von_mises_2d(
            num_steps=args.num_steps,
            max_episode_steps=ep_steps,
            seed=args.seed,
            output_path=out,
            kappa=args.kappa,
        )
    elif args.env == '3d_von_mises':
        ep_steps = args.max_episode_steps if args.max_episode_steps != 200 else 10
        tag = f"{int(args.kappa)}" if float(args.kappa) == int(args.kappa) else f"{args.kappa}"
        out = args.output_path or f'./datasets/particle_circle_3d_vonmises_k{tag}.pkl'
        generate_von_mises_3d(
            num_steps=args.num_steps,
            max_episode_steps=ep_steps,
            seed=args.seed,
            output_path=out,
            kappa=args.kappa,
        )
    elif args.env == '2d_von_mises_sweep':
        ep_steps = args.max_episode_steps if args.max_episode_steps != 200 else 10
        out_dir = args.output_path or './datasets'
        generate_von_mises_sweep(
            kappa_values=args.kappa_values,
            num_steps=args.num_steps,
            max_episode_steps=ep_steps,
            seed=args.seed,
            output_dir=out_dir,
        )
    elif args.env == 'trifinger_position':
        out = args.output_path or './datasets/trifinger_position_random_uniform.pkl'
        generate_trifinger_position(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
        )
    elif args.env == 'trifinger':
        out = args.output_path or './datasets/trifinger_random_uniform.pkl'
        generate_trifinger(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
        )
    else:
        drag_tag = f'_drag{args.drag:.2f}' if args.drag != 0.0 else ''
        out = args.output_path or f'./datasets/particle_circle_2d{drag_tag}_random_uniform.pkl'
        generate(
            num_steps=args.num_steps,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            output_path=out,
            drag=args.drag,
        )