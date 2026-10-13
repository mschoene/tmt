"""Shared offline dataset loading helpers.

Used by train_tmt_nn.py, train_tmt_knn.py, and future baselines (LieSD, LiGAN, …).
"""

from __future__ import annotations

import numpy as np

try:
    import minari
except Exception:
    minari = None


def load_episodes(dataset_path: str) -> list:
    """Return a list of episode dicts from a .pkl file or a Minari dataset ID."""
    if dataset_path.endswith(".pkl"):
        import pickle
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        if "episodes" not in dataset:
            raise ValueError(f"Expected key 'episodes' in {dataset_path}")
        return dataset["episodes"]
    if minari is None:
        raise RuntimeError("minari is not available but dataset_path is not a .pkl file")
    ds = minari.load_dataset(dataset_path, download=True)
    return [
        dict(
            observations=ep.observations,
            actions=ep.actions,
            rewards=ep.rewards,
            terminals=ep.terminations,
        )
        for ep in ds.iterate_episodes()
    ]


def build_offline_arrays(dataset_path: str):
    """Build flat (states, actions, next_states, episode_ids, step_ids) arrays.

    Returns
    -------
    states       : np.ndarray (N, state_dim) float32
    actions      : np.ndarray (N, action_dim) float32  — clipped to (-1, 1)
    next_states  : np.ndarray (N, state_dim) float32
    episode_ids  : np.ndarray (N,) int64
    step_ids     : np.ndarray (N,) int64
    """
    episodes = load_episodes(dataset_path)
    states, actions, next_states, episode_ids, step_ids = [], [], [], [], []
    for ep_idx, ep in enumerate(episodes):
        obs = np.asarray(ep["observations"], dtype=np.float32)
        act = np.asarray(ep["actions"], dtype=np.float32)
        if act.ndim == 1:
            act = act[:, None]
        T = obs.shape[0] - 1
        states.append(obs[:T])
        next_states.append(obs[1:])
        actions.append(np.clip(act[:T], -0.99999, 0.99999))
        episode_ids.append(np.full(T, ep_idx, dtype=np.int64))
        step_ids.append(np.arange(T, dtype=np.int64))
    return (
        np.concatenate(states, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(next_states, axis=0),
        np.concatenate(episode_ids, axis=0),
        np.concatenate(step_ids, axis=0),
    )


def sample_transitions(
    dataset_path: str,
    num_points: int,
    seed: int,
    *,
    as_torch: bool = True,
):
    """Load, subsample (up to num_points), and sort transitions by episode/step.

    Parameters
    ----------
    dataset_path : str
        Path to a .pkl file or a Minari dataset ID.
    num_points : int
        Maximum number of transitions to keep.
    seed : int
        RNG seed for reproducible subsampling.
    as_torch : bool
        If True (default) also return torch.Tensor versions alongside numpy arrays.

    Returns
    -------
    numpy dict with keys: states, actions, next_states, episode_ids, step_ids
    torch dict with same keys (only when as_torch=True)
    """
    states_np, actions_np, next_states_np, episode_ids_np, step_ids_np = (
        build_offline_arrays(dataset_path)
    )
    n_total = states_np.shape[0]
    n_use = min(int(num_points), n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_use, replace=False) if n_use < n_total else np.arange(n_total)
    # Sort by (episode, step) so time ordering is preserved within episodes.
    idx = np.asarray(
        sorted(idx.tolist(), key=lambda i: (int(episode_ids_np[i]), int(step_ids_np[i]))),
        dtype=np.int64,
    )

    np_data = {
        "states": states_np[idx],
        "actions": actions_np[idx],
        "next_states": next_states_np[idx],
        "episode_ids": episode_ids_np[idx],
        "step_ids": step_ids_np[idx],
    }

    if not as_torch:
        return np_data, None

    import torch
    torch_data = {
        "states": torch.as_tensor(np_data["states"], dtype=torch.float32),
        "actions": torch.as_tensor(np_data["actions"], dtype=torch.float32),
        "next_states": torch.as_tensor(np_data["next_states"], dtype=torch.float32),
        "episode_ids": torch.as_tensor(np_data["episode_ids"], dtype=torch.long),
        "step_ids": torch.as_tensor(np_data["step_ids"], dtype=torch.long),
    }
    return np_data, torch_data
