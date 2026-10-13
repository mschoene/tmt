"""LieSD symmetry discovery: Jacobian-based tangent-space estimation.

Instead of the denoiser/score pipeline, we:
  1. Train a plain MLP dynamics model  f: (s, a) -> s'  with MSE loss.
  2. At each point z_i = [s_i; a_i; s'_i] compute the Jacobian
         J_i = df/d(s,a)  in  R^{state_dim x (state_dim + action_dim)}
     This gives the tangent space of the transition manifold analytically.
  3. Normal frame at z_i:
         N_i = [[-J_i^T], [I_{state_dim}]]   (shape z_dim x state_dim)
     followed by QR orthonormalisation — no eigendecomposition needed.
  4. Continuous and discrete discovery are identical to TMT-NN / TMT-KNN,
     reusing discover_continuous_symmetries and MultiDiscreteSymmetryDiscovery.

Advantages vs TMT-NN:
  - No sigma hyperparameter.
  - Jacobians are (state_dim x (s+a)) not (z_dim x z_dim): faster to compute.
  - Normal frames are closed-form: no eigh of a large matrix per point.
  - Training objective is plain MSE: simpler, typically faster to converge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.networks.score_symmetry_discovery import (
    ContinuousDiscoveryConfig,
    DiscreteDiscoveryConfig,
    GroupAssemblyConfig,
    MLP,
    MultiDiscreteSymmetryDiscovery,
    _assemble_group,
    _batched_manifold_rmse,
    _build_splits,
    _check_policy_symmetry,
    _empty_continuous_result,
    _empty_discrete_result,
    _merge_discrete_results,
    _to_numpy,
    discover_continuous_symmetries,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DynamicsTrainingConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 1024
    num_epochs: int = 200
    validation_split: float = 0.1
    holdout_split: float = 0.1
    grad_clip_norm: Optional[float] = 5.0
    seed: int = 0
    log_interval: int = 25
    lr_schedule: str = "cosine"
    lr_min_factor: float = 0.01
    lr_warmup_epochs: int = 0
    normalize_data: bool = True


@dataclass
class LieSDConfig:
    training: DynamicsTrainingConfig = field(default_factory=DynamicsTrainingConfig)
    continuous: ContinuousDiscoveryConfig = field(default_factory=ContinuousDiscoveryConfig)
    discrete: DiscreteDiscoveryConfig = field(default_factory=DiscreteDiscoveryConfig)
    group: GroupAssemblyConfig = field(default_factory=GroupAssemblyConfig)


# ---------------------------------------------------------------------------
# Dynamics model
# ---------------------------------------------------------------------------

class DynamicsModel(eqx.Module):
    """Plain MLP dynamics model: [s; a] -> s'."""
    net: MLP
    state_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...], *, key):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.net = MLP(state_dim + action_dim, state_dim, hidden_dims, key=key)

    def __call__(self, sa: jnp.ndarray) -> jnp.ndarray:
        """sa: (state_dim + action_dim,) -> s': (state_dim,)"""
        return self.net(sa)


# ---------------------------------------------------------------------------
# Dynamics-as-projector: plugs into the existing discovery pipeline
# ---------------------------------------------------------------------------

class DynamicsAsProjector(eqx.Module):
    """Wraps DynamicsModel as a manifold projector for the TMT pipeline.

    Implements two APIs expected by downstream code:

    __call__(z, sigma=None):
        Project z = [s; a; s'] onto the transition manifold by replacing s'
        with f(s, a).  Manifold distance = ||f(s,a) - s'||, used by the
        discrete search loss.

    get_normal_frames(z_points):
        Fast-path used by _compute_normal_frames instead of jacrev on the
        full z-space.  Computes J_f = df/d(s,a) at each point and returns
        N_i = QR([[-J_f^T], [I_s]]) — the analytic normal frame.
    """
    dynamics: DynamicsModel
    state_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    @property
    def data_dim(self) -> int:
        return 2 * self.state_dim + self.action_dim

    def __call__(self, z: jnp.ndarray, sigma=None) -> jnp.ndarray:
        single = z.ndim == 1
        if single:
            z = z[None]
        sa = z[:, : self.state_dim + self.action_dim]
        s_pred = jax.vmap(self.dynamics)(sa)
        out = jnp.concatenate([sa, s_pred], axis=-1)
        return out[0] if single else out

    def get_normal_frames(self, z_points: np.ndarray, frame_batch_size: int = 256) -> np.ndarray:
        """Return analytic normal frames for each query point.

        For z_i = [s_i; a_i; s'_i]:
          J_i = df/d(s,a)  shape (state_dim, state_dim + action_dim)
          N_raw_i = [[-J_i^T], [I_s]]  shape (z_dim, state_dim)
          N_i = QR(N_raw_i)[:, :state_dim]   (orthonormal normal frame)

        Returns
        -------
        frames : np.ndarray  shape (N, z_dim, state_dim)
        """
        sa_dim = self.state_dim + self.action_dim
        sa_points = np.asarray(z_points[:, :sa_dim], dtype=np.float32)

        def jac_single(sa):
            return jax.jacrev(self.dynamics)(sa)   # (state_dim, sa_dim)

        jac_fn = jax.jit(jax.vmap(jac_single))

        all_frames = []
        eye_s = np.eye(self.state_dim, dtype=np.float32)
        for start in range(0, sa_points.shape[0], frame_batch_size):
            chunk = jnp.asarray(sa_points[start: start + frame_batch_size])
            jacs = np.asarray(jac_fn(chunk))   # (b, state_dim, sa_dim)
            for J in jacs:
                N_raw = np.concatenate([-J.T, eye_s], axis=0)   # (z_dim, state_dim)
                Q, _ = np.linalg.qr(N_raw)
                all_frames.append(Q[:, : self.state_dim].astype(np.float32))

        return np.stack(all_frames)   # (N, z_dim, state_dim)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_lr_schedule_dyn(cfg: DynamicsTrainingConfig, total_steps: int):
    lr = float(cfg.lr)
    end_lr = lr * float(cfg.lr_min_factor)
    warmup_steps = int(cfg.lr_warmup_epochs) * max(1, total_steps // max(1, int(cfg.num_epochs)))
    sched = str(cfg.lr_schedule)
    if sched == "cosine":
        if warmup_steps > 0:
            return optax.warmup_cosine_decay_schedule(0.0, lr, warmup_steps, total_steps, end_lr)
        return optax.cosine_decay_schedule(lr, total_steps, alpha=cfg.lr_min_factor)
    if sched == "linear":
        if warmup_steps > 0:
            return optax.join_schedules(
                [optax.linear_schedule(0.0, lr, warmup_steps),
                 optax.linear_schedule(lr, end_lr, total_steps - warmup_steps)],
                [warmup_steps],
            )
        return optax.linear_schedule(lr, end_lr, total_steps)
    return lr


class DynamicsTrainer:
    def __init__(self, model: DynamicsModel, config: DynamicsTrainingConfig, key):
        self.model = model
        self.config = config
        self.key = key

    @staticmethod
    @eqx.filter_jit
    def _step(model, opt_state, optimizer, sa_batch, sp_batch):
        def loss_fn(m):
            sp_pred = jax.vmap(m)(sa_batch)
            return jnp.mean(jnp.sum((sp_pred - sp_batch) ** 2, axis=-1))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    def fit(self, z_std: np.ndarray, splits: dict, verbose: bool = False) -> dict:
        cfg = self.config
        sa_dim = self.model.state_dim + self.model.action_dim

        z_train = z_std[splits["train_idx"]]
        z_val = z_std[splits["val_idx"]] if splits["val_idx"].size else z_train[:0]

        sa_train = jnp.asarray(z_train[:, :sa_dim], dtype=jnp.float32)
        sp_train = jnp.asarray(z_train[:, sa_dim: sa_dim + self.model.state_dim], dtype=jnp.float32)
        sa_val = jnp.asarray(z_val[:, :sa_dim], dtype=jnp.float32) if z_val.shape[0] > 0 else None
        sp_val = jnp.asarray(z_val[:, sa_dim: sa_dim + self.model.state_dim], dtype=jnp.float32) if z_val.shape[0] > 0 else None

        n_train = int(sa_train.shape[0])
        steps_per_epoch = max(1, math.ceil(n_train / int(cfg.batch_size)))
        total_steps = int(cfg.num_epochs) * steps_per_epoch

        schedule = _build_lr_schedule_dyn(cfg, total_steps)
        tx_parts = []
        if cfg.grad_clip_norm is not None:
            tx_parts.append(optax.clip_by_global_norm(float(cfg.grad_clip_norm)))
        tx_parts.append(optax.adamw(schedule, weight_decay=float(cfg.weight_decay)))
        optimizer = optax.chain(*tx_parts)

        model = self.model
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        key = self.key

        train_losses, val_losses = [], []

        for epoch in range(int(cfg.num_epochs)):
            key, sk = jax.random.split(key)
            perm = np.asarray(jax.random.permutation(sk, n_train))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, int(cfg.batch_size)):
                idx = perm[start: start + int(cfg.batch_size)]
                model, opt_state, loss = self._step(
                    model, opt_state, optimizer,
                    sa_train[idx], sp_train[idx],
                )
                epoch_loss += float(loss)
                n_batches += 1
            train_losses.append(epoch_loss / max(n_batches, 1))

            if sa_val is not None and sp_val is not None:
                sp_pred = jax.vmap(model)(sa_val)
                val_loss = float(jnp.mean(jnp.sum((sp_pred - sp_val) ** 2, axis=-1)))
            else:
                val_loss = float("nan")
            val_losses.append(val_loss)

            if verbose and (
                epoch == 0
                or epoch == int(cfg.num_epochs) - 1
                or (epoch + 1) % max(1, int(cfg.log_interval)) == 0
            ):
                print(
                    f"[liesd/dynamics] epoch {epoch + 1}/{cfg.num_epochs} "
                    f"train_mse={train_losses[-1]:.4e} val_mse={val_loss:.4e}"
                )

        self.model = model
        return {"train_mse": train_losses, "val_mse": val_losses}


# ---------------------------------------------------------------------------
# Proxy training-config (satisfies discover_continuous_symmetries API)
# ---------------------------------------------------------------------------

@dataclass
class _DynTrainingCfgProxy:
    sigma_min: float = 1.0
    sigma_max: float = 1.0
    seed: int = 0
    sigma_eval: float = 0.0


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def discover_liesd_symmetries(
    states,
    actions,
    next_states,
    episode_ids,
    step_ids,
    state_dim: int,
    action_dim: int,
    device=None,
    config: Optional[LieSDConfig] = None,
    verbose: bool = False,
    discrete_log_interval: int = 250,
    only_discrete: bool = False,
    only_continuous: bool = False,
):
    """LieSD symmetry discovery.

    Mirrors the interface of discover_score_symmetries / discover_knn_symmetries.
    Replaces the denoiser with a plain dynamics model; normal frames come from
    the analytic Jacobian formula rather than eigendecomposing the denoiser Jacobian.

    Returns the same result-dict structure so logging/saving helpers are shared.
    """
    del device, episode_ids, step_ids
    cfg = config or LieSDConfig()
    cfg.discrete.log_interval = int(discrete_log_interval)

    states_np = _to_numpy(states).astype(np.float32)
    actions_np = _to_numpy(actions).astype(np.float32)
    next_states_np = _to_numpy(next_states).astype(np.float32)
    z = np.concatenate([states_np, actions_np, next_states_np], axis=-1).astype(np.float32)

    # --- Standardise ---
    splits = _build_splits(
        num_points=int(z.shape[0]),
        validation_split=float(cfg.training.validation_split),
        holdout_split=float(cfg.training.holdout_split),
        seed=int(cfg.training.seed),
    )
    z_train_raw = z[splits["train_idx"]]
    z_mean = z_train_raw.mean(axis=0)
    z_std_arr = z_train_raw.std(axis=0)
    if cfg.training.normalize_data:
        is_const  = z_std_arr < 1e-6
        z_scale   = np.where(is_const, 1.0, z_std_arr).astype(np.float32)
        z_shift   = np.where(is_const, 0.0, z_mean).astype(np.float32)
    else:
        z_scale = np.ones_like(z_mean, dtype=np.float32)
        z_shift = np.zeros_like(z_mean, dtype=np.float32)
    z_std_all = ((z - z_shift) / z_scale).astype(np.float32)

    z_train_std = z_std_all[splits["train_idx"]]
    z_val_std = z_std_all[splits["val_idx"]]
    z_holdout_std = z_std_all[splits["holdout_idx"]]

    # --- Train dynamics model ---
    if verbose:
        print("[phase] liesd: training dynamics model")

    key = jax.random.PRNGKey(int(cfg.training.seed))
    dyn_model = DynamicsModel(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        hidden_dims=tuple(cfg.training.hidden_dims),
        key=key,
    )
    trainer = DynamicsTrainer(dyn_model, cfg.training, key)
    dyn_history = trainer.fit(z_std_all, splits, verbose=verbose)
    dyn_model = trainer.model

    # Evaluate holdout dynamics MSE as a proxy for manifold quality.
    sa_dim = int(state_dim) + int(action_dim)
    holdout_mse = 0.0
    if z_holdout_std.shape[0] > 0:
        sa_ho = jnp.asarray(z_holdout_std[:, :sa_dim], dtype=jnp.float32)
        sp_ho = jnp.asarray(z_holdout_std[:, sa_dim: sa_dim + state_dim], dtype=jnp.float32)
        sp_pred = jax.vmap(dyn_model)(sa_ho)
        holdout_mse = float(jnp.mean(jnp.sum((sp_pred - sp_ho) ** 2, axis=-1)))

    train_mse = dyn_history["train_mse"][-1] if dyn_history["train_mse"] else 0.0

    if verbose:
        print(
            f"[phase] dynamics fitted  train_mse={train_mse:.4e}  holdout_mse={holdout_mse:.4e}"
        )

    history = {
        "train_total_loss": dyn_history["train_mse"],
        "val_total_loss": dyn_history["val_mse"],
        "diagnostics": {
            "full_manifold_rmse": train_mse,
            "train_manifold_rmse": train_mse,
            "holdout_manifold_rmse": holdout_mse,
            "sigma_eval": 0.0,
        },
        "split": splits,
    }

    projector = DynamicsAsProjector(
        dynamics=dyn_model,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
    )

    cfg.continuous.sigma_eval = 0.0
    training_cfg_proxy = _DynTrainingCfgProxy(seed=int(cfg.training.seed))

    if only_discrete:
        if verbose:
            print("[phase] liesd: continuous symmetry discovery SKIPPED (only_discrete=True)")
        continuous = _empty_continuous_result(int(state_dim), int(action_dim))
    else:
        if verbose:
            print("[phase] liesd: continuous symmetry discovery start")
        continuous = discover_continuous_symmetries(
            model=projector,
            z_train_std=z_train_std,
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            training_cfg=training_cfg_proxy,
            config=cfg.continuous,
            verbose=verbose,
        )

    if only_continuous:
        if verbose:
            print("[phase] liesd: discrete symmetry discovery SKIPPED (only_continuous=True)")
        discrete = _empty_discrete_result(int(state_dim), int(action_dim))
    else:
        n_candidates = int(cfg.discrete.num_candidates)
        cand_batch = int(cfg.discrete.candidates_batch_size)
        if cand_batch <= 0 or cand_batch >= n_candidates:
            if verbose:
                print("[phase] liesd: discrete symmetry discovery start")
            discrete = MultiDiscreteSymmetryDiscovery(
                model=projector,
                z_train_std=z_train_std,
                z_holdout_std=z_holdout_std,
                z_val_std=z_val_std,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                sigma_eval=0.0,
                config=cfg.discrete,
                training_history=history,
            ).fit(verbose=verbose)
        else:
            n_batches = math.ceil(n_candidates / cand_batch)
            if verbose:
                print(f"[phase] liesd: discrete symmetry discovery start "
                      f"({n_candidates} candidates in {n_batches} batches of {cand_batch})")
            batch_results = []
            remaining = n_candidates
            for batch_idx in range(n_batches):
                this_batch = min(cand_batch, remaining)
                batch_cfg = replace(cfg.discrete, num_candidates=this_batch,
                                    seed=cfg.discrete.seed + batch_idx * 1000)
                if verbose:
                    print(f"  [discrete] batch {batch_idx + 1}/{n_batches} ({this_batch} candidates)")
                batch_results.append(MultiDiscreteSymmetryDiscovery(
                    model=projector,
                    z_train_std=z_train_std,
                    z_holdout_std=z_holdout_std,
                    z_val_std=z_val_std,
                    state_dim=int(state_dim),
                    action_dim=int(action_dim),
                    sigma_eval=0.0,
                    config=batch_cfg,
                    training_history=history,
                ).fit(verbose=verbose))
                remaining -= this_batch
            discrete = _merge_discrete_results(
                batch_results, int(state_dim), int(action_dim),
                int(cfg.discrete.max_subgroup_elements),
            )

    group = _assemble_group(
        continuous=continuous,
        discrete=discrete,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        theta=float(cfg.group.validation_theta),
        max_elements=int(cfg.group.max_elements),
    )

    s_train_std = z_train_std[:, : int(state_dim)]
    a_train_std = z_train_std[:, int(state_dim): int(state_dim) + int(action_dim)]
    if verbose:
        print("[phase] liesd: policy symmetry check start")
    policy_check = _check_policy_symmetry(
        states_std=s_train_std,
        actions_std=a_train_std,
        continuous_generators=continuous["generators"],
        discrete_candidates=discrete["candidates"],
        seed=int(cfg.training.seed),
        verbose=verbose,
    )

    return {
        "model": projector,
        "training_history": history,
        "standardizer": {
            "mean": z_mean.astype(np.float32),
            "scale": z_scale.astype(np.float32),
        },
        "data": {
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "z_dim": int(z_std_all.shape[1]),
            "z_holdout_std": z_holdout_std.astype(np.float32),
            "sigma_eval": 0.0,
        },
        "continuous": continuous,
        "discrete": discrete,
        "group": group,
        "policy_check": policy_check,
    }
