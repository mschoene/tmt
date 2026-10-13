"""Score-based Lie symmetry discovery for offline control transitions.

Pipeline:
  1. Standardise z = [s; a; s'] coordinate-wise.
  2. Train sigma-conditioned denoiser d_phi(z; sigma) as a small-noise DAE.
  3. Estimate normal frames from I - d d_phi / d z.
  4. Solve continuous generator problem from the manifold-normal constraints.
  5. Search for multiple discrete signed-permutation symmetries using only
     denoiser manifold distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback used only when scipy missing.
    linear_sum_assignment = None


@dataclass
class ScoreTrainingConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 1024
    num_epochs: int = 300
    validation_split: float = 0.1
    holdout_split: float = 0.1
    grad_clip_norm: Optional[float] = 5.0
    sigma_min: float = 1e-2
    sigma_max: float = 1e-1
    sigma_eval: Optional[float] = None
    log_interval: int = 25
    seed: int = 0
    verbose: bool = True
    standardize_eps: float = 1e-6
    normalize_data: bool = True
    # LR schedule: "constant" | "cosine" | "linear"
    lr_schedule: str = "cosine"
    lr_min_factor: float = 0.01   # end LR = lr * lr_min_factor
    lr_warmup_epochs: int = 0     # linear ramp from 0→lr before decay starts


@dataclass
class ContinuousDiscoveryConfig:
    max_generators: int = 8
    num_frame_points: int = 2048
    frame_batch_size: int = 64
    spectral_gap_min_ratio: float = 1.5
    closure_tol: float = 1e-1
    closure_max_generators: int = 16
    sigma_eval: Optional[float] = None
    polish_steps: int = 0
    polish_lr: float = 5e-2
    polish_batch_size: int = 256
    use_symmetrized_normal_matrix: bool = False
    abs_threshold_frac: float = 1e-2
    # Geometric intertwiner: for each found generator (L_s, L_a), find the best
    # linear map W: R^{d_s} → R^{d_a} satisfying W L_s = L_a W (Sylvester eq.).
    # The residual ||L_a W - W L_s||_F / ||W||_F is a quality metric:
    # small → L_s and L_a are consistently related; large → spurious/decoupled.
    report_intertwiner_quality: bool = True
    # Reject generators whose intertwiner residual exceeds this (inf = no filter).
    max_intertwiner_residual: float = float("inf")
    # If True, replace L_a with the Sylvester solution F_a^+ L_s F_a given the
    # found W, making the generator exactly satisfy the intertwiner constraint.
    fix_L_a_from_intertwiner: bool = False
    # Multi-scale M: evaluate M(σ) at this many log-spaced sigmas spanning
    # [sigma_min, sigma_max] and average the trace-normalised matrices.
    # 1 (default) = single sigma_eval, identical to previous behaviour.
    multiscale_sigmas: int = 1
    # Tangent-projector variant for ablation studies (TMT-NN only).
    #   "default"  — eigendecompose symmetrised (I-P), rank-truncate to state_dim.
    #   "raw_P"    — use (I-P_i) full-rank, quadratic: M += C^T (I-P)^T (I-P) C.
    #   "raw_S"    — symmetrise I-P per frame, full-rank: M += C^T S^2 C.
    #   "M_sym_P"  — linear: M_raw += C^T (I-P) C, then M = 0.5*(M+M^T).
    projector_variant: str = "default"
    # Affine symmetry discovery: extend operator space with translation generators
    # (b_s, b_a) so that v = [vec(L_s); vec(L_a); b_s; b_a].
    # Enabled automatically when data has a lifted (constant-1) dimension, or
    # set explicitly to search for SE(n)-type symmetries without lifting.
    affine: bool = False
    # Minimum ||L_s|| (Frobenius, on the unit-norm generator vector) for a
    # generator to be kept.  Generators with a near-zero state block act
    # trivially on the state and are always spurious.
    # NOTE: L_a = 0 is a *valid* solution (e.g. joint torques are invariant
    # scalars under a rotation symmetry), so this threshold gates only on
    # ||L_s||, never on ||L_a||.  Set to 0.0 to disable entirely.
    min_state_norm: float = 1e-2

    use_crossval_refinement: bool = True
    crossval_secondary_gap_ratio: float = 3.0


@dataclass
class DiscreteDiscoveryConfig:
    num_candidates: int = 8
    lr: float = 1e-2
    num_steps: int = 1000
    batch_size: int = 1024
    sinkhorn_iters: int = 50
    temperature_start: float = 5.0
    temperature_end: float = 0.05
    anneal_power: float = 1.0
    mu_identity_repulsion: float = 0.1
    kappa_diversity: float = 1e-2
    # Weight of the action-space term in the non-identity barrier.
    # 0.0 (default): barrier fires only when Gs≈I, allowing Ga=I (action-invariant symmetries).
    # 1.0: old behaviour — barrier fires when both Gs≈I and Ga≈I; biases away from action-invariant solutions.
    identity_repulsion_action_weight: float = 0.0
    # Temperature of the non-identity barrier exp(-d_id / T).  Small T = sharp
    # decay, barrier fires only very close to identity.  Default 0.25 is too
    # small: random Sinkhorn init has d_id≈4, so exp(-4/0.25)≈e^{-16}≈0 and
    # the barrier gradient vanishes throughout training.  Increase T so the
    # barrier fires at initialization scale.
    identity_barrier_temperature: float = 0.25
    manifold_distance_tau: float = 0.2
    # Accept candidate G only if holdout_mfd(G) < rho_acceptance_threshold * identity_mfd.
    # This makes tau relative to the actual identity distance rather than an absolute scale,
    # preventing false positives when identity_mfd << manifold_distance_tau (e.g. liesd/knn).
    rho_acceptance_threshold: float = 2.0
    exclude_identity: bool = True
    max_subgroup_elements: int = 64
    log_interval: int = 200
    seed: int = 0
    # Geometric intertwiner: learn a shared W: R^{d_s}→R^{d_a} alongside the
    # permutation candidates.  Adds penalty mu_intertwiner * sum_k ||W Gs_k - Ga_k W||_F
    # to the Sinkhorn loss, coupling G_s and G_a through consistent geometry.
    # Optional — set False if you expect no linear coupling between state/action spaces.
    use_intertwiner: bool = False
    mu_intertwiner: float = 0.05
    # Duplicate-refresh: every refresh_interval steps (once temperature drops below
    # half the starting value) check for collapsed candidates and re-init them.
    # Set to 0 to disable.
    refresh_interval: int = 200
    duplicate_soft_threshold: float = 0.5
    # Run discrete search in sequential batches of this many candidates to cap
    # peak memory.  -1 (default) runs all num_candidates at once (original behaviour).
    candidates_batch_size: int = -1


@dataclass
class GroupAssemblyConfig:
    validation_theta: float = 1.0
    max_elements: int = 128
    composition_tol: float = 1e-6


@dataclass
class ScoreSymmetryDiscoveryConfig:
    training: ScoreTrainingConfig = field(default_factory=ScoreTrainingConfig)
    continuous: ContinuousDiscoveryConfig = field(default_factory=ContinuousDiscoveryConfig)
    discrete: DiscreteDiscoveryConfig = field(default_factory=DiscreteDiscoveryConfig)
    group: GroupAssemblyConfig = field(default_factory=GroupAssemblyConfig)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, (jax.Array, jnp.ndarray)):
        return np.asarray(x)
    return np.asarray(x)


def _to_jnp(x, dtype=jnp.float32) -> jnp.ndarray:
    return jnp.asarray(_to_numpy(x), dtype=dtype)


def _hyperoctahedral_size(d: int) -> int:
    """Size of the signed-permutation group B_d = 2^d * d!"""
    return (2 ** d) * math.factorial(d)


def _linear_schedule(start, end, step, total_steps, power=1.0):
    if total_steps <= 1:
        return float(end)
    frac = min(max(step / float(total_steps - 1), 0.0), 1.0) ** power
    return float(start * (1.0 - frac) + end * frac)


def _build_splits(num_points: int, validation_split: float, holdout_split: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_points)
    n_holdout = int(round(num_points * holdout_split))
    n_val = int(round(num_points * validation_split))
    n_holdout = min(max(n_holdout, 0), num_points)
    n_val = min(max(n_val, 0), num_points - n_holdout)
    holdout_idx = np.sort(perm[:n_holdout])
    val_idx = np.sort(perm[n_holdout:n_holdout + n_val])
    train_idx = np.sort(perm[n_holdout + n_val:])
    if train_idx.size == 0:
        train_idx = np.sort(perm[max(n_holdout + n_val - 1, 0):])
        if val_idx.size > 0:
            val_idx = val_idx[:-1]
    return {
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "holdout_idx": holdout_idx.astype(np.int64),
    }


class MLP(eqx.Module):
    layers: tuple

    def __init__(self, input_dim, output_dim, hidden_dims, *, key):
        dims = [int(input_dim), *[int(h) for h in hidden_dims], int(output_dim)]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = tuple(
            eqx.nn.Linear(i, o, key=k) for i, o, k in zip(dims[:-1], dims[1:], keys)
        )

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = x @ layer.weight.T + layer.bias
            x = jax.nn.silu(x)
        last = self.layers[-1]
        return x @ last.weight.T + last.bias


class SigmaConditionedDenoiser(eqx.Module):
    net: MLP
    data_dim: int = eqx.field(static=True)

    def __init__(self, data_dim, hidden_dims, *, key):
        self.data_dim = int(data_dim)
        self.net = MLP(self.data_dim + 1, self.data_dim, hidden_dims, key=key)

    def __call__(self, z, sigma):
        z = jnp.asarray(z, dtype=jnp.float32)
        single = z.ndim == 1
        if single:
            z = z[None, :]
        sigma = jnp.asarray(sigma, dtype=z.dtype)
        if sigma.ndim == 0:
            sigma = jnp.full((z.shape[0], 1), sigma)
        elif sigma.ndim == 1:
            sigma = sigma[:, None]
        elif sigma.ndim == 2 and sigma.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Unsupported sigma shape {sigma.shape}")
        feat = jnp.concatenate([z, jnp.log(jnp.clip(sigma, a_min=1e-8))], axis=-1)
        out = z + self.net(feat)
        return out[0] if single else out


def _build_lr_schedule(cfg: "ScoreTrainingConfig", total_steps: int):
    """Build an optax LR schedule from config.  Returns a schedule or a plain float."""
    schedule_type = str(getattr(cfg, "lr_schedule", "constant"))
    lr = float(cfg.lr)
    min_factor = float(getattr(cfg, "lr_min_factor", 0.01))
    warmup_epochs = int(getattr(cfg, "lr_warmup_epochs", 0))
    steps_per_epoch = max(1, total_steps // max(1, int(cfg.num_epochs)))
    warmup_steps = warmup_epochs * steps_per_epoch
    end_lr = lr * min_factor

    if schedule_type == "cosine":
        if warmup_steps > 0:
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=int(warmup_steps),
                decay_steps=int(total_steps),
                end_value=end_lr,
            )
        return optax.cosine_decay_schedule(lr, int(total_steps), alpha=min_factor)
    if schedule_type == "linear":
        if warmup_steps > 0:
            warmup_sched = optax.linear_schedule(0.0, lr, int(warmup_steps))
            decay_sched  = optax.linear_schedule(lr, end_lr, int(total_steps - warmup_steps))
            return optax.join_schedules([warmup_sched, decay_sched], [int(warmup_steps)])
        return optax.linear_schedule(lr, end_lr, int(total_steps))
    return lr  # "constant" — scalar accepted by adamw


class ScoreDenoiserTrainer:
    def __init__(self, model: SigmaConditionedDenoiser, config: ScoreTrainingConfig, key):
        self.model = model
        self.config = config
        self.key = key
        # Placeholder optimizer — rebuilt with proper schedule inside fit().
        tx_parts = []
        if config.grad_clip_norm is not None:
            tx_parts.append(optax.clip_by_global_norm(float(config.grad_clip_norm)))
        tx_parts.append(optax.adamw(config.lr, weight_decay=config.weight_decay))
        self.optimizer = optax.chain(*tx_parts)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

    def _sample_sigmas(self, key, batch_size):
        log_lo = math.log(float(self.config.sigma_min))
        log_hi = math.log(float(self.config.sigma_max))
        u = jax.random.uniform(key, (batch_size,), minval=log_lo, maxval=log_hi)
        return jnp.exp(u)

    @staticmethod
    @eqx.filter_jit
    def _step(model, opt_state, optimizer, clean, key, sigma_min, sigma_max):
        log_lo = jnp.log(jnp.asarray(sigma_min, dtype=jnp.float32))
        log_hi = jnp.log(jnp.asarray(sigma_max, dtype=jnp.float32))
        key_sigma, key_noise = jax.random.split(key)
        sigma = jnp.exp(jax.random.uniform(key_sigma, (clean.shape[0],), minval=log_lo, maxval=log_hi))
        noisy = clean + sigma[:, None] * jax.random.normal(key_noise, clean.shape)

        def loss_fn(m):
            pred = m(noisy, sigma)
            return jnp.mean(jnp.sum((pred - clean) ** 2, axis=-1))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @staticmethod
    @eqx.filter_jit
    def _eval_loss(model, clean, key, sigma_min, sigma_max):
        log_lo = jnp.log(jnp.asarray(sigma_min, dtype=jnp.float32))
        log_hi = jnp.log(jnp.asarray(sigma_max, dtype=jnp.float32))
        key_sigma, key_noise = jax.random.split(key)
        sigma = jnp.exp(jax.random.uniform(key_sigma, (clean.shape[0],), minval=log_lo, maxval=log_hi))
        noisy = clean + sigma[:, None] * jax.random.normal(key_noise, clean.shape)
        pred = model(noisy, sigma)
        return jnp.mean(jnp.sum((pred - clean) ** 2, axis=-1))

    def fit(self, z_std: np.ndarray):
        cfg = self.config
        splits = _build_splits(
            num_points=int(z_std.shape[0]),
            validation_split=float(cfg.validation_split),
            holdout_split=float(cfg.holdout_split),
            seed=int(cfg.seed),
        )
        z_j = jnp.asarray(z_std, dtype=jnp.float32)
        train_idx = splits["train_idx"]
        val_idx = splits["val_idx"]

        # Rebuild optimizer with proper LR schedule now that dataset size is known.
        n_train = int(train_idx.size)
        steps_per_epoch = max(1, math.ceil(n_train / int(cfg.batch_size)))
        total_steps = int(cfg.num_epochs) * steps_per_epoch
        schedule = _build_lr_schedule(cfg, total_steps)
        tx_parts = []
        if cfg.grad_clip_norm is not None:
            tx_parts.append(optax.clip_by_global_norm(float(cfg.grad_clip_norm)))
        tx_parts.append(optax.adamw(schedule, weight_decay=cfg.weight_decay))
        self.optimizer = optax.chain(*tx_parts)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        if cfg.verbose:
            print(
                f"[score] LR schedule={getattr(cfg, 'lr_schedule', 'constant')} "
                f"lr={cfg.lr:.2e} total_steps={total_steps} "
                f"warmup_epochs={getattr(cfg, 'lr_warmup_epochs', 0)}"
            )

        history = {"train_total_loss": [], "val_total_loss": []}
        best_val = float("inf")
        best_model = self.model

        for epoch in range(int(cfg.num_epochs)):
            train_perm = np.random.default_rng(int(cfg.seed) + epoch).permutation(train_idx)
            batch_losses = []
            for start in range(0, train_perm.size, int(cfg.batch_size)):
                batch_idx = train_perm[start:start + int(cfg.batch_size)]
                if batch_idx.size == 0:
                    continue
                clean = z_j[jnp.asarray(batch_idx)]
                self.key, step_key = jax.random.split(self.key)
                self.model, self.opt_state, loss = self._step(
                    self.model,
                    self.opt_state,
                    self.optimizer,
                    clean,
                    step_key,
                    float(cfg.sigma_min),
                    float(cfg.sigma_max),
                )
                batch_losses.append(loss)  # keep as JAX scalar; sync once per epoch below

            train_loss = float(jnp.mean(jnp.stack(batch_losses))) if batch_losses else 0.0
            history["train_total_loss"].append(train_loss)

            if val_idx.size > 0:
                self.key, eval_key = jax.random.split(self.key)
                val_loss = float(
                    self._eval_loss(
                        self.model,
                        z_j[jnp.asarray(val_idx)],
                        eval_key,
                        float(cfg.sigma_min),
                        float(cfg.sigma_max),
                    )
                )
            else:
                val_loss = train_loss
            history["val_total_loss"].append(val_loss)

            if val_loss <= best_val:
                best_val = val_loss
                best_model = self.model

            if cfg.verbose and (
                epoch == 0
                or epoch == int(cfg.num_epochs) - 1
                or (epoch + 1) % max(1, int(cfg.log_interval)) == 0
            ):
                print(
                    f"[score] epoch {epoch + 1}/{cfg.num_epochs} "
                    f"train={train_loss:.4e} val={val_loss:.4e}"
                )

        self.model = best_model
        history["split"] = splits
        history["diagnostics"] = self.diagnostics(z_std, splits)
        return history

    def diagnostics(self, z_std: np.ndarray, splits: Dict[str, np.ndarray]):
        sigma_eval = float(self.config.sigma_eval) if self.config.sigma_eval is not None else math.sqrt(float(self.config.sigma_min) * float(self.config.sigma_max))
        # Use numpy indexing to avoid materialising large JAX device arrays for
        # the full dataset — _batched_manifold_rmse handles batching internally.
        z_np = _to_numpy(z_std).astype(np.float32)
        full_rmse = _batched_manifold_rmse(self.model, z_np, sigma_eval, batch_size=4096)
        train_rmse = _batched_manifold_rmse(
            self.model, z_np[splits["train_idx"]], sigma_eval, batch_size=4096
        )
        holdout_idx = splits["holdout_idx"]
        holdout_rmse = (
            _batched_manifold_rmse(
                self.model, z_np[holdout_idx], sigma_eval, batch_size=4096
            )
            if holdout_idx.size > 0
            else full_rmse
        )
        return {
            "sigma_eval": sigma_eval,
            "full_manifold_rmse": full_rmse,
            "train_manifold_rmse": train_rmse,
            "holdout_manifold_rmse": holdout_rmse,
            "data_dim": int(z_std.shape[1]),
            "num_points": int(z_std.shape[0]),
        }


@eqx.filter_jit
def _manifold_sqerr(model, z_batch, sigma_eval):
    pred = model(z_batch, sigma_eval)
    return jnp.sum((pred - z_batch) ** 2, axis=-1)


def _batched_manifold_rmse(model, z, sigma_eval, batch_size=4096):
    z_np = _to_numpy(z)
    if z_np.shape[0] == 0:
        return 0.0
    errs = []
    for start in range(0, z_np.shape[0], batch_size):
        batch = jnp.asarray(z_np[start:start + batch_size], dtype=jnp.float32)
        errs.append(np.asarray(_manifold_sqerr(model, batch, float(sigma_eval))))
    sq = np.concatenate(errs, axis=0)
    return float(np.sqrt(np.mean(sq)))


def _ctilde(z: np.ndarray, state_dim: int, action_dim: int,
            affine: bool = False) -> np.ndarray:
    """C̃ matrix: maps operator vector v → displacement δz = (δs, δa, δs').

    Linear (affine=False): v = [vec(L_s); vec(L_a)], size n²+m².
    Affine (affine=True):  v = [vec(L_s); vec(L_a); b_s; b_a], size n²+m²+n+m.
    Same b_s block in top (δs) and bottom (δs') enforces s and s' shift together.
    """
    n, m = state_dim, action_dim
    s  = z[:n]
    a  = z[n:n + m]
    sp = z[n + m:]
    In = np.eye(n, dtype=np.float32)
    Im = np.eye(m, dtype=np.float32)

    top_lin = np.concatenate(
        [np.kron(s[None, :],  In), np.zeros((n, m ** 2), dtype=np.float32)], axis=1)
    mid_lin = np.concatenate(
        [np.zeros((m, n ** 2), dtype=np.float32), np.kron(a[None, :], Im)], axis=1)
    bot_lin = np.concatenate(
        [np.kron(sp[None, :], In), np.zeros((n, m ** 2), dtype=np.float32)], axis=1)

    if not affine:
        return np.concatenate([top_lin, mid_lin, bot_lin], axis=0)

    Zns = np.zeros((n, m), dtype=np.float32)
    Zms = np.zeros((m, n), dtype=np.float32)
    top = np.concatenate([top_lin, In,  Zns], axis=1)
    mid = np.concatenate([mid_lin, Zms, Im ], axis=1)
    bot = np.concatenate([bot_lin, In,  Zns], axis=1)
    return np.concatenate([top, mid, bot], axis=0)


def _pack_generator(L_s: np.ndarray, L_a: np.ndarray,
                    b_s: Optional[np.ndarray] = None,
                    b_a: Optional[np.ndarray] = None) -> np.ndarray:
    parts = [L_s.reshape(-1), L_a.reshape(-1)]
    if b_s is not None:
        parts.append(np.asarray(b_s).reshape(-1))
    if b_a is not None:
        parts.append(np.asarray(b_a).reshape(-1))
    return np.concatenate(parts, axis=0)


def _split_generator(v: np.ndarray, state_dim: int, action_dim: int,
                     affine: bool = False):
    n, m = state_dim, action_dim
    L_s = v[:n ** 2].reshape(n, n)
    L_a = v[n ** 2: n ** 2 + m ** 2].reshape(m, m)
    if not affine:
        return L_s, L_a
    b_s = v[n ** 2 + m ** 2: n ** 2 + m ** 2 + n]
    b_a = v[n ** 2 + m ** 2 + n:]
    return L_s, L_a, b_s, b_a


def _generator_bracket(v1: np.ndarray, v2: np.ndarray,
                       state_dim: int, action_dim: int,
                       affine: bool = False):
    """Lie bracket [(L1,b1),(L2,b2)] = ([L1,L2], L1 b2 - L2 b1) (affine algebra)."""
    if affine:
        Ls1, La1, bs1, ba1 = _split_generator(v1, state_dim, action_dim, affine=True)
        Ls2, La2, bs2, ba2 = _split_generator(v2, state_dim, action_dim, affine=True)
        return _pack_generator(
            Ls1 @ Ls2 - Ls2 @ Ls1,
            La1 @ La2 - La2 @ La1,
            Ls1 @ bs2 - Ls2 @ bs1,
            La1 @ ba2 - La2 @ ba1,
        )
    Ls1, La1 = _split_generator(v1, state_dim, action_dim)
    Ls2, La2 = _split_generator(v2, state_dim, action_dim)
    return _pack_generator(Ls1 @ Ls2 - Ls2 @ Ls1, La1 @ La2 - La2 @ La1)


def _projection_residual(vec: np.ndarray, basis: Sequence[np.ndarray]):
    if not basis:
        return vec, np.zeros((0,), dtype=np.float32)
    B = np.stack(basis, axis=1)
    coeff, *_ = np.linalg.lstsq(B, vec, rcond=None)
    recon = B @ coeff
    return vec - recon, coeff


def _round_signed_permutation(soft: np.ndarray, sign_pre: np.ndarray) -> np.ndarray:
    soft = np.asarray(soft, dtype=np.float32)
    signs = np.sign(np.asarray(sign_pre, dtype=np.float32))
    signs[signs == 0.0] = 1.0
    if linear_sum_assignment is None:
        cols = []
        used = set()
        for row in range(soft.shape[0]):
            order = np.argsort(-np.abs(soft[row]))
            chosen = next((int(c) for c in order if int(c) not in used), int(order[0]))
            cols.append(chosen)
            used.add(chosen)
    else:
        _, col_idx = linear_sum_assignment(-np.abs(soft))
        cols = [int(c) for c in col_idx]
    P = np.zeros_like(soft)
    for row, col in enumerate(cols):
        P[row, col] = 1.0
    return (signs[:, None] * P).astype(np.float32)


def _key_for_pair(Gs: np.ndarray, Ga: np.ndarray):
    return (
        tuple(np.rint(np.asarray(Gs)).astype(np.int8).reshape(-1).tolist()),
        tuple(np.rint(np.asarray(Ga)).astype(np.int8).reshape(-1).tolist()),
    )


def _close_discrete_subgroup(candidates: Sequence[dict], state_dim: int, action_dim: int, max_elements: int):
    eye_s = np.eye(state_dim, dtype=np.float32)
    eye_a = np.eye(action_dim, dtype=np.float32)
    elements = [{"kind": "identity", "D_s": eye_s, "D_a": eye_a}]
    seen = {_key_for_pair(eye_s, eye_a)}
    for cand in candidates:
        key = _key_for_pair(cand["hard_P_s"], cand["hard_P_a"])
        if key in seen:
            continue
        seen.add(key)
        elements.append(
            {
                "kind": "discrete_generator",
                "D_s": cand["hard_P_s"],
                "D_a": cand["hard_P_a"],
                "source_index": int(cand["index"]),
                "manifold_distance": float(cand["holdout_manifold_distance"]),
            }
        )
    changed = True
    while changed and len(elements) < int(max_elements):
        changed = False
        snapshot = list(elements)
        for left in snapshot:
            for right in snapshot:
                Gs = left["D_s"] @ right["D_s"]
                Ga = left["D_a"] @ right["D_a"]
                key = _key_for_pair(Gs, Ga)
                if key in seen:
                    continue
                seen.add(key)
                elements.append(
                    {
                        "kind": "discrete_composition",
                        "D_s": Gs.astype(np.float32),
                        "D_a": Ga.astype(np.float32),
                        "source": (left["kind"], right["kind"]),
                    }
                )
                changed = True
                if len(elements) >= int(max_elements):
                    break
            if len(elements) >= int(max_elements):
                break
    return {"elements": elements, "num_elements": len(elements)}


def _apply_group_to_z(z_batch: jnp.ndarray, Gs: jnp.ndarray, Ga: jnp.ndarray, state_dim: int, action_dim: int):
    s = z_batch[:, :state_dim]
    a = z_batch[:, state_dim:state_dim + action_dim]
    sp = z_batch[:, state_dim + action_dim:]
    s_g = jnp.einsum("bd,ked->kbe", s, Gs)
    a_g = jnp.einsum("bd,ked->kbe", a, Ga)
    sp_g = jnp.einsum("bd,ked->kbe", sp, Gs)
    return jnp.concatenate([s_g, a_g, sp_g], axis=-1)


def _sinkhorn(logits, temperature, n_iters):
    log_alpha = logits / jnp.maximum(jnp.asarray(temperature, dtype=logits.dtype), 1e-8)

    def body(la, _):
        la = la - jax.scipy.special.logsumexp(la, axis=-1, keepdims=True)
        la = la - jax.scipy.special.logsumexp(la, axis=-2, keepdims=True)
        return la, None

    log_alpha, _ = jax.lax.scan(body, log_alpha, None, length=int(n_iters))
    return jnp.exp(log_alpha)


class _MultiDiscreteParams(eqx.Module):
    state_logits: jax.Array
    action_logits: jax.Array
    state_sign_pre: jax.Array
    action_sign_pre: jax.Array
    # Geometric intertwiner W: (d_a, d_s).  All-zeros when use_intertwiner=False
    # (penalty weight mu_intertwiner=0 ensures it has no effect on the loss then).
    intertwiner_W: jax.Array


class MultiDiscreteSymmetryDiscovery:
    def __init__(
        self,
        model,
        z_train_std: np.ndarray,
        z_holdout_std: np.ndarray,
        state_dim: int,
        action_dim: int,
        sigma_eval: float,
        config: DiscreteDiscoveryConfig,
        training_history: dict,
        z_val_std: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.z_train = _to_jnp(z_train_std)
        # Combine val + holdout for a larger unbiased eval set (val is unused in discrete opt)
        if z_val_std is not None and z_val_std.shape[0] > 0:
            z_eval = np.concatenate([z_val_std, z_holdout_std], axis=0)
        else:
            z_eval = z_holdout_std
        self.z_holdout = _to_jnp(z_eval if z_eval.shape[0] > 0 else z_train_std)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.sigma_eval = float(sigma_eval)

        self.config = config
        self.training_history = training_history
                

    def _init_params(self):
        cfg = self.config
        rng = np.random.default_rng(int(cfg.seed))
        state_logits = []
        action_logits = []
        state_sign_pre = []
        action_sign_pre = []
        K = int(cfg.num_candidates)
        # Half the candidates are sign-biased toward negative (p_neg=0.75 per dim).
        # Random init alone leaves candidates ~n_s/2 sign-flips away from any
        # mostly-negative target (e.g. g0 needs 7/8 dims negative). Biased init
        # reduces the expected number of wrong signs to ~n_s/4, making the
        # manifold gradient task easier. The other half stays fully random to
        # maintain coverage of positive-sign and mixed-sign symmetries.
        for k in range(K):
            perm_s = np.eye(self.state_dim, dtype=np.float32)[rng.permutation(self.state_dim)]
            perm_a = np.eye(self.action_dim, dtype=np.float32)[rng.permutation(self.action_dim)]
            state_logits.append(2.0 * perm_s + 0.05 * rng.standard_normal((self.state_dim, self.state_dim)))
            action_logits.append(2.0 * perm_a + 0.05 * rng.standard_normal((self.action_dim, self.action_dim)))
            if k < K // 2:
                s_sign = rng.choice([-2.0, 2.0], size=(self.state_dim,),
                                    p=[0.75, 0.25]).astype(np.float32)
                a_sign = rng.choice([-2.0, 2.0], size=(self.action_dim,),
                                    p=[0.75, 0.25]).astype(np.float32)
            else:
                s_sign = rng.choice([-2.0, 2.0], size=(self.state_dim,)).astype(np.float32)
                a_sign = rng.choice([-2.0, 2.0], size=(self.action_dim,)).astype(np.float32)
            state_sign_pre.append(s_sign)
            action_sign_pre.append(a_sign)
        # W: (d_a, d_s) rectangular intertwiner.  Random unit-norm init so gradients
        # are non-zero from step 0.  Zero weight (mu_intertwiner=0) when not used.
        W_raw = rng.standard_normal((self.action_dim, self.state_dim)).astype(np.float32)
        W_init = W_raw / (float(np.linalg.norm(W_raw)) + 1e-12)
        return _MultiDiscreteParams(
            state_logits=jnp.asarray(np.stack(state_logits), dtype=jnp.float32),
            action_logits=jnp.asarray(np.stack(action_logits), dtype=jnp.float32),
            state_sign_pre=jnp.asarray(np.stack(state_sign_pre), dtype=jnp.float32),
            action_sign_pre=jnp.asarray(np.stack(action_sign_pre), dtype=jnp.float32),
            intertwiner_W=jnp.asarray(W_init),
        )
    

    def _temperature(self, step):
        return _linear_schedule(
            self.config.temperature_start,
            self.config.temperature_end,
            step,
            self.config.num_steps,
            self.config.anneal_power,
        )

    @staticmethod
    @eqx.filter_jit
    def _step(
        params,
        opt_state,
        optimizer,
        model,
        z_batch,
        sigma_eval,
        temperature,
        sinkhorn_iters,
        mu_identity_repulsion,
        kappa_diversity,
        state_dim,
        action_dim,
        mu_intertwiner,  # float: weight for W-intertwiner penalty; 0.0 to disable
        action_id_weight,  # float: 0.0 = state-only barrier (allows Ga=I), 1.0 = old joint barrier
        identity_barrier_temperature,  # float: temperature T in exp(-d_id/T); larger = fires further from identity
        noise_key,  # PRNGKey: adds DAE-style noise gz+σε before model call; zero-noise when sigma_eval=0
    ):
        def loss_fn(p):
                Qs = _sinkhorn(p.state_logits, temperature, sinkhorn_iters)
                Qa = _sinkhorn(p.action_logits, temperature, sinkhorn_iters)
                Gs = jnp.tanh(p.state_sign_pre)[:, :, None] * Qs
                Ga = jnp.tanh(p.action_sign_pre)[:, :, None] * Qa

                # Manifold loss: scan over candidates sequentially so peak GPU
                # memory is O(batch_size * data_dim) regardless of num_candidates,
                # instead of the previous O(K * batch_size * data_dim) that OOMed.
                s  = z_batch[:, :state_dim]
                a  = z_batch[:, state_dim:state_dim + action_dim]
                sp = z_batch[:, state_dim + action_dim:]

                # Per-candidate noise keys for DAE-style manifold loss.
                # Adds σ·ε to Gz before the model call so the loss is the actual DAE
                # objective: E[||d(Gz+σε, σ) - Gz||²].  For sigma_eval=0 (liesd, knn)
                # the noise is zero and behaviour is unchanged.
                K = Gs.shape[0]
                noise_keys = jax.random.split(noise_key, K)

                def _manifold_k(carry, inp):
                    Gs_k, Ga_k, key_k = inp
                    gz_k = jnp.concatenate(
                        [s @ Gs_k.T, a @ Ga_k.T, sp @ Gs_k.T], axis=-1
                    )
                    noise = jax.random.normal(key_k, gz_k.shape) * sigma_eval
                    pred_k = model(gz_k + noise, sigma_eval)
                    m_k = jnp.mean(jnp.sum((pred_k - gz_k) ** 2, axis=-1))
                    return carry, m_k

                _, manifold_per_k = jax.lax.scan(_manifold_k, None, (Gs, Ga, noise_keys))

                # Barrier active only near identity, not a global push-away.
                # d_id uses state distance only by default (action_id_weight=0.0) so that
                # the optimizer is not biased against action-invariant symmetries (Ga=I).
                # Set action_id_weight=1.0 to restore the old joint-space barrier.
                eye_s = jnp.eye(state_dim, dtype=Gs.dtype)[None, :, :]
                eye_a = jnp.eye(action_dim, dtype=Ga.dtype)[None, :, :]
                d_id = (jnp.sum((Gs - eye_s) ** 2, axis=(1, 2))
                        + action_id_weight * jnp.sum((Ga - eye_a) ** 2, axis=(1, 2)))
                identity_barrier = jnp.sum(jnp.exp(-d_id / identity_barrier_temperature))

                flat = jnp.concatenate(
                    [Gs.reshape((Gs.shape[0], -1)), Ga.reshape((Ga.shape[0], -1))],
                    axis=-1,
                )
                # ||a-b||² = ||a||² + ||b||² - 2<a,b>  avoids materialising (K,K,d²) tensor.
                norm_sq = jnp.sum(flat ** 2, axis=-1)          # (K,)
                gram = flat @ flat.T                             # (K, K)
                pairwise_dist_sq = (norm_sq[:, None] + norm_sq[None, :] - 2.0 * gram + 1e-8)
                mask = jnp.triu(jnp.ones_like(pairwise_dist_sq), k=1)
                diversity = jnp.sum(mask / (pairwise_dist_sq + 0.1))

                # Geometric intertwiner: ||G_a W - W G_s||_F per candidate k.
                # G_a W = W G_s couples state/action generators through a shared
                # (d_a, d_s) linear map.  Normalised by ||W||² so W scale is free.
                # mu_intertwiner=0 makes this zero-cost when the option is disabled.
                W = p.intertwiner_W                          # (d_a, d_s)
                Ga_W = jnp.einsum("kij,jl->kil", Ga, W)    # (k, d_a, d_s)
                W_Gs = jnp.einsum("ij,kjl->kil", W, Gs)    # (k, d_a, d_s)
                intertwiner_res = Ga_W - W_Gs               # (k, d_a, d_s)
                w_nrm_sq = jnp.sum(W ** 2) + 1e-8
                intertwiner_loss = (
                    jnp.sum(jnp.mean(intertwiner_res ** 2, axis=(1, 2))) / w_nrm_sq
                )

                total = (
                    jnp.sum(manifold_per_k)
                    + mu_identity_repulsion * identity_barrier
                    + kappa_diversity * diversity
                    + mu_intertwiner * intertwiner_loss
                )
                aux = (jnp.mean(manifold_per_k), identity_barrier, diversity, Gs, Ga, Qs, Qa)
                return total, aux

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(params, eqx.is_array))
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss, aux

    def _refresh_duplicates(
        self, params: "_MultiDiscreteParams", step: int
    ) -> tuple["_MultiDiscreteParams", int]:
        """Re-initialize candidates whose soft (Gs, Ga) are too similar to an earlier one.

        Only runs when temperature < 0.5 * temperature_start (soft matrices are
        meaningfully sharp).  Returns (new_params, num_refreshed).
        """
        cfg = self.config
        threshold = float(getattr(cfg, "duplicate_soft_threshold", 0.5))
        temp = self._temperature(step)

        Qs = np.asarray(_sinkhorn(params.state_logits, temp, int(cfg.sinkhorn_iters)))
        Qa = np.asarray(_sinkhorn(params.action_logits, temp, int(cfg.sinkhorn_iters)))
        Gs = np.tanh(np.asarray(params.state_sign_pre))[:, :, None] * Qs
        Ga = np.tanh(np.asarray(params.action_sign_pre))[:, :, None] * Qa

        K = int(cfg.num_candidates)
        rng = np.random.default_rng(int(cfg.seed) + step)

        new_sl = np.asarray(params.state_logits)
        new_al = np.asarray(params.action_logits)
        new_ss = np.asarray(params.state_sign_pre)
        new_as = np.asarray(params.action_sign_pre)

        active = [0]
        refreshed = 0
        for k in range(1, K):
            Gs_k = Gs[k].reshape(-1)
            Ga_k = Ga[k].reshape(-1)
            is_dup = any(
                float(np.linalg.norm(
                    np.concatenate([Gs_k - Gs[j].reshape(-1), Ga_k - Ga[j].reshape(-1)])
                )) < threshold
                for j in active
            )
            if is_dup:
                perm_s = np.eye(self.state_dim, dtype=np.float32)[rng.permutation(self.state_dim)]
                perm_a = np.eye(self.action_dim, dtype=np.float32)[rng.permutation(self.action_dim)]
                new_sl[k] = 2.0 * perm_s + 0.05 * rng.standard_normal((self.state_dim, self.state_dim))
                new_al[k] = 2.0 * perm_a + 0.05 * rng.standard_normal((self.action_dim, self.action_dim))
                new_ss[k] = rng.choice([-2.0, 2.0], size=(self.state_dim,)).astype(np.float32)
                new_as[k] = rng.choice([-2.0, 2.0], size=(self.action_dim,)).astype(np.float32)
                refreshed += 1
            else:
                active.append(k)

        if refreshed == 0:
            return params, 0
        return _MultiDiscreteParams(
            state_logits=jnp.asarray(new_sl),
            action_logits=jnp.asarray(new_al),
            state_sign_pre=jnp.asarray(new_ss),
            action_sign_pre=jnp.asarray(new_as),
            intertwiner_W=params.intertwiner_W,
        ), refreshed

    def fit(self, verbose=False):
        cfg = self.config
        params = self._init_params()

        # Compute identity manifold distance before optimization so we can use it
        # to calibrate the acceptance threshold (rho-based tau).
        identity_mfd_pre = float(_batched_manifold_rmse(self.model, self.z_holdout, self.sigma_eval))
        train_rmse = float(self.training_history["diagnostics"]["train_manifold_rmse"])
        self.config.manifold_distance_tau = max(float(self.config.manifold_distance_tau), 5.0 * train_rmse)
        # Effective tau: accept G only if holdout_mfd < rho * identity_mfd.
        # Falls back to cfg.manifold_distance_tau if identity_mfd is zero.
        _rho = float(getattr(cfg, "rho_acceptance_threshold", 2.0))
        effective_tau = (
            min(float(self.config.manifold_distance_tau), _rho * identity_mfd_pre)
            if identity_mfd_pre > 0.0
            else float(self.config.manifold_distance_tau)
        )

        optimizer = optax.adam(float(cfg.lr))
        opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
        history = {"loss_total": [], "loss_manifold": [], "loss_identity_push": [], "diversity": [], "temperature": []}
        key = jax.random.PRNGKey(int(cfg.seed))
        _raw_loss: list = []
        _raw_manifold: list = []
        _raw_id_push: list = []
        _raw_diversity: list = []


        for step in range(int(cfg.num_steps)):
            temperature = self._temperature(step)
            key, subkey, noise_key = jax.random.split(key, 3)
            batch_idx = jax.random.randint(
                subkey,
                (min(int(cfg.batch_size), int(self.z_train.shape[0])),),
                minval=0,
                maxval=int(self.z_train.shape[0]),
            )
            z_batch = jnp.take(self.z_train, batch_idx, axis=0)
            params, opt_state, loss, aux = self._step(
                params,
                opt_state,
                optimizer,
                self.model,
                z_batch,
                float(self.sigma_eval),
                jnp.asarray(temperature, dtype=jnp.float32),
                int(cfg.sinkhorn_iters),
                float(cfg.mu_identity_repulsion),
                float(cfg.kappa_diversity),
                int(self.state_dim),
                int(self.action_dim),
                float(cfg.mu_intertwiner) if getattr(cfg, "use_intertwiner", False) else 0.0,
                float(cfg.identity_repulsion_action_weight),
                float(cfg.identity_barrier_temperature),
                noise_key,
            )
            manifold, identity_push, diversity, _, _, _, _ = aux
            _raw_loss.append(loss)
            _raw_manifold.append(manifold)
            _raw_id_push.append(identity_push)
            _raw_diversity.append(diversity)
            history["temperature"].append(float(temperature))

            # Duplicate refresh: re-init collapsed candidates once soft matrices
            # are sharp enough (temperature < half of start).
            refresh_interval = int(getattr(cfg, "refresh_interval", 200))
            if (
                refresh_interval > 0
                and (step + 1) % refresh_interval == 0
                and temperature < float(cfg.temperature_start) * 0.5
            ):
                params, n_refreshed = self._refresh_duplicates(params, step)
                if n_refreshed > 0:
                    # Reset Adam moments for all candidates so stale second-moment
                    # estimates don't freeze the freshly re-initialized ones.
                    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
                    if verbose:
                        print(
                            f"[discrete/score] step {step + 1}: refreshed {n_refreshed} duplicate candidate(s)"
                        )

            if verbose and (
                step == 0
                or step == int(cfg.num_steps) - 1
                or (step + 1) % max(1, int(cfg.log_interval)) == 0
            ):
                print(
                    f"[discrete/score] step {step + 1}/{cfg.num_steps} "
                    f"loss={float(loss):.4e} mfd={float(manifold):.4e} "
                    f"id_push={float(identity_push):.4e}"
                )

        # Batch GPU→CPU transfer: one sync instead of 4×num_steps individual syncs.
        history["loss_total"] = np.asarray(jnp.stack(_raw_loss)).tolist()
        history["loss_manifold"] = np.asarray(jnp.stack(_raw_manifold)).tolist()
        history["loss_identity_push"] = np.asarray(jnp.stack(_raw_id_push)).tolist()
        history["diversity"] = np.asarray(jnp.stack(_raw_diversity)).tolist()

        final_temp = float(self._temperature(int(cfg.num_steps) - 1))
        Qs = np.asarray(_sinkhorn(params.state_logits, final_temp, int(cfg.sinkhorn_iters)))
        Qa = np.asarray(_sinkhorn(params.action_logits, final_temp, int(cfg.sinkhorn_iters)))
        state_sign_pre = np.asarray(params.state_sign_pre)
        action_sign_pre = np.asarray(params.action_sign_pre)
        soft_Gs = np.asarray(np.tanh(state_sign_pre)[:, :, None] * Qs, dtype=np.float32)
        soft_Ga = np.asarray(np.tanh(action_sign_pre)[:, :, None] * Qa, dtype=np.float32)

        rounded = []
        eye_s = np.eye(self.state_dim, dtype=np.float32)
        eye_a = np.eye(self.action_dim, dtype=np.float32)
        for idx in range(int(cfg.num_candidates)):
            hard_s = _round_signed_permutation(Qs[idx], state_sign_pre[idx])
            hard_a = _round_signed_permutation(Qa[idx], action_sign_pre[idx])
            z_holdout_np = np.asarray(self.z_holdout)
            s = z_holdout_np[:, :self.state_dim] @ hard_s.T
            a = z_holdout_np[:, self.state_dim:self.state_dim + self.action_dim] @ hard_a.T
            sp = z_holdout_np[:, self.state_dim + self.action_dim:] @ hard_s.T
            z_trans = np.concatenate([s, a, sp], axis=-1).astype(np.float32)
            holdout_distance = _batched_manifold_rmse(
                self.model, jnp.asarray(z_trans), self.sigma_eval
            )
            item = {
                "index": int(idx),
                "soft_P_s": soft_Gs[idx],
                "soft_P_a": soft_Ga[idx],
                "hard_P_s": hard_s,
                "hard_P_a": hard_a,
                "state_sign": np.sign(state_sign_pre[idx]).astype(np.float32),
                "action_sign": np.sign(action_sign_pre[idx]).astype(np.float32),
                "holdout_manifold_distance": float(holdout_distance),
            }
            if cfg.exclude_identity and np.allclose(hard_s, eye_s) and np.allclose(hard_a, eye_a):
                item["rejection_reason"] = "identity"
            elif holdout_distance >= effective_tau:
                item["rejection_reason"] = "manifold_distance_above_tau"
            else:
                item["rejection_reason"] = ""
            rounded.append(item)

        accepted = []
        rejections = []
        seen = set()
        for item in sorted(rounded, key=lambda x: x["holdout_manifold_distance"]):
            key_pair = _key_for_pair(item["hard_P_s"], item["hard_P_a"])
            if item["rejection_reason"]:
                rejections.append(item)
                continue
            if key_pair in seen:
                item = {**item, "rejection_reason": "duplicate"}
                rejections.append(item)
                continue
            seen.add(key_pair)
            accepted.append(item)

        subgroup = _close_discrete_subgroup(
            accepted, self.state_dim, self.action_dim, max_elements=int(cfg.max_subgroup_elements)
        )
        best = accepted[0] if accepted else min(rounded, key=lambda x: x["holdout_manifold_distance"])
        learned_W = np.asarray(params.intertwiner_W) if getattr(cfg, "use_intertwiner", False) else None
        best_mfd = float(best["holdout_manifold_distance"])
        rho_mfd = (best_mfd / identity_mfd_pre) if identity_mfd_pre > 0.0 else float("nan")
        return {
            "history": history,
            "soft_P_s": best["soft_P_s"],
            "soft_P_a": best["soft_P_a"],
            "hard_P_s": best["hard_P_s"],
            "hard_P_a": best["hard_P_a"],
            "candidates": accepted,
            "rejections": rejections,
            "all_candidates": rounded,
            "subgroup": subgroup,
            "best_holdout_manifold_distance": best_mfd,
            "identity_manifold_distance": identity_mfd_pre,
            "rho_mfd": rho_mfd,
            "acceptance_tau": float(effective_tau),
            "learned_intertwiner_W": learned_W,
        }


 

def _polish_generator(
    init_v: np.ndarray,
    A_stack: np.ndarray,
    state_dim: int,
    action_dim: int,
    polish_steps: int,
    polish_lr: float,
    polish_batch_size: int,
    seed: int,
):
    if polish_steps <= 0 or A_stack.shape[0] == 0:
        return init_v
    opt = optax.adam(float(polish_lr))
    v = jnp.asarray(init_v, dtype=jnp.float32)
    opt_state = opt.init(v)
    key = jax.random.PRNGKey(int(seed))

    @jax.jit
    def step(v, opt_state, batch):
        def loss_fn(x):
            proj = jnp.einsum("bij,j->bi", batch, x)
            loss = jnp.mean(jnp.sum(proj ** 2, axis=-1))
            loss += 1e-2 * (jnp.sum(x ** 2) - 1.0) ** 2
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(v)
        updates, opt_state = opt.update(grads, opt_state, v)
        v = optax.apply_updates(v, updates)
        v = v / (jnp.linalg.norm(v) + 1e-12)
        return v, opt_state, loss

    for _ in range(int(polish_steps)):
        key, subkey = jax.random.split(key)
        if A_stack.shape[0] <= int(polish_batch_size):
            batch = jnp.asarray(A_stack, dtype=jnp.float32)
        else:
            idx = np.asarray(
                jax.random.randint(subkey, (int(polish_batch_size),), 0, int(A_stack.shape[0]))
            )
            batch = jnp.asarray(A_stack[idx], dtype=jnp.float32)
        v, opt_state, _ = step(v, opt_state, batch)
    return _normalize_generator(np.asarray(v), state_dim, action_dim)


def _assemble_group(
    continuous: dict,
    discrete: dict,
    state_dim: int,
    action_dim: int,
    theta: float,
    max_elements: int,
):
    elements = []
    subgroup_elems = discrete["subgroup"]["elements"]
    for elem in subgroup_elems[: int(max_elements)]:
        elements.append(
            {
                "kind": elem["kind"],
                "D_s": np.asarray(elem["D_s"], dtype=np.float32),
                "D_a": np.asarray(elem["D_a"], dtype=np.float32),
            }
        )
    for gen in continuous["generators"]:
        if len(elements) >= int(max_elements):
            break
        D_s = np.asarray(jax.scipy.linalg.expm(theta * jnp.asarray(gen["L_s"])), dtype=np.float32)
        D_a = np.asarray(jax.scipy.linalg.expm(theta * jnp.asarray(gen["L_a"])), dtype=np.float32)
        elements.append(
            {
                "kind": "continuous_exp",
                "generator_index": int(gen["index"]),
                "D_s": D_s,
                "D_a": D_a,
            }
        )
    if not elements:
        elements.append(
            {
                "kind": "identity",
                "D_s": np.eye(state_dim, dtype=np.float32),
                "D_a": np.eye(action_dim, dtype=np.float32),
            }
        )
    return {"elements": elements, "num_elements": len(elements)}


def _train_bc_policy(
    states_std: np.ndarray,
    actions_std: np.ndarray,
    hidden_dims=(64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = 0,
):
    """Train tiny BC policy MLP: s_std → a_std using equinox/jax."""
    key = jax.random.PRNGKey(seed)
    state_dim = states_std.shape[1]
    action_dim = actions_std.shape[1]
    model = MLP(state_dim, action_dim, hidden_dims, key=key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    s_jnp = jnp.asarray(states_std, dtype=jnp.float32)
    a_jnp = jnp.asarray(actions_std, dtype=jnp.float32)
    n = int(s_jnp.shape[0])

    @eqx.filter_jit
    def step_fn(model, opt_state, s_b, a_b):
        def loss_fn(m):
            pred = jax.vmap(m)(s_b)
            return jnp.mean((pred - a_b) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    rng = np.random.default_rng(seed)
    for _ in range(int(epochs)):
        perm = rng.permutation(n)
        for start in range(0, n, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            model, opt_state, _ = step_fn(model, opt_state, s_jnp[idx], a_jnp[idx])

    return model


def _check_policy_symmetry(
    states_std: np.ndarray,
    actions_std: np.ndarray,
    continuous_generators: list,
    discrete_candidates: list,
    bc_hidden_dims=(64, 64),
    bc_epochs: int = 200,
    bc_lr: float = 1e-3,
    bc_batch_size: int = 512,
    seed: int = 0,
    max_eval_points: int = 10000,
    verbose: bool = False,
) -> dict:
    """Check whether transition symmetries are also policy symmetries.

    Trains a BC policy π: s → a, then for each symmetry (G_s, G_a) computes:
      discrete:   residual = E[||G_a π(s) - π(G_s s)||²] / E[||a||²]
      continuous: residual = E[||L_a π(s) - J_π(s) L_s s||²] / E[||a||²]

    Small residual → symmetry is also a policy symmetry.
    Does NOT reject anything — purely diagnostic.
    """
    if verbose:
        print("[policy_check] training BC policy for symmetry check...")

    n = states_std.shape[0]
    eval_n = min(int(max_eval_points), n)
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(n, size=eval_n, replace=False)
    s_eval = states_std[eval_idx]
    a_eval = actions_std[eval_idx]

    bc_policy = _train_bc_policy(
        states_std, actions_std,
        hidden_dims=bc_hidden_dims,
        epochs=bc_epochs,
        lr=bc_lr,
        batch_size=bc_batch_size,
        seed=seed,
    )

    s_jnp = jnp.asarray(s_eval, dtype=jnp.float32)
    a_jnp = jnp.asarray(a_eval, dtype=jnp.float32)
    pi_s = jax.vmap(bc_policy)(s_jnp)
    a_sq_mean = float(jnp.mean(a_jnp ** 2))
    bc_mse = float(jnp.mean((pi_s - a_jnp) ** 2))
    bc_mse_normalized = bc_mse / max(a_sq_mean, 1e-8)

    if verbose:
        print(f"[policy_check] BC MSE (norm): {bc_mse_normalized:.3e}")

    discrete_checks = []
    for cand in discrete_candidates:
        Ps = jnp.asarray(cand["hard_P_s"], dtype=jnp.float32)
        Pa = jnp.asarray(cand["hard_P_a"], dtype=jnp.float32)
        pi_Gs_s = jax.vmap(bc_policy)(s_jnp @ Ps.T)
        Ga_pi_s = pi_s @ Pa.T
        residual = float(jnp.mean((Ga_pi_s - pi_Gs_s) ** 2)) / max(a_sq_mean, 1e-8)
        discrete_checks.append({
            "index": int(cand["index"]),
            "policy_equivariance_residual": residual,
        })
        if verbose:
            print(f"  discrete[{int(cand['index'])}] policy_equiv_residual={residual:.3e}")

    # Continuous: check L_a π(s) ≈ J_π(s) L_s s
    @eqx.filter_jit
    def jac_batch_fn(s_b):
        return jax.vmap(jax.jacrev(bc_policy))(s_b)

    Js = np.asarray(jac_batch_fn(s_jnp), dtype=np.float64)
    pi_s_np = np.asarray(pi_s, dtype=np.float64)
    s_eval_f64 = s_eval.astype(np.float64)

    continuous_checks = []
    for gen in continuous_generators:
        Ls = np.asarray(gen["L_s"], dtype=np.float64)
        La = np.asarray(gen["L_a"], dtype=np.float64)
        La_pi = pi_s_np @ La.T
        Ls_s = s_eval_f64 @ Ls.T
        J_Ls_s = np.einsum("nab,nb->na", Js, Ls_s)
        residual = float(np.mean((La_pi - J_Ls_s) ** 2)) / max(a_sq_mean, 1e-8)
        continuous_checks.append({
            "index": int(gen["index"]),
            "policy_equivariance_residual": residual,
        })
        if verbose:
            print(f"  continuous[{int(gen['index'])}] policy_equiv_residual={residual:.3e}")

    return {
        "bc_mse_normalized": bc_mse_normalized,
        "discrete": discrete_checks,
        "continuous": continuous_checks,
    }


def _empty_continuous_result(state_dim: int, action_dim: int) -> dict:
    op_dim = state_dim ** 2 + action_dim ** 2
    return {
        "generators": [],
        "frame_indices": np.zeros(0, dtype=np.int64),
        "normal_frame_metadata": {},
        "spectrum": np.zeros(0, dtype=np.float32),
        "matrix_M": np.zeros((op_dim, op_dim), dtype=np.float32),
        "picked_k": 0,
        "spectral_gap_index": -1,
        "spectral_gap_ratio": 0.0,
        "closure_error": 0.0,
        "closure_error_before_completion": 0.0,
        "closure_added_generators": [],
        "closure_rejected_brackets": [],
        "block_decoupled_dropped": [],
        "sigma_eval": 0.0,
        "gap_info": {},
        "projector_variant": "skipped",
        "mean_projector_idempotency_err": 0.0,
        "max_projector_idempotency_err": 0.0,
    }


def _empty_discrete_result(state_dim: int, action_dim: int) -> dict:
    return {
        "history": {"loss_total": [], "loss_manifold": [], "loss_identity_push": []},
        "soft_P_s": np.eye(state_dim, dtype=np.float32),
        "soft_P_a": np.eye(action_dim, dtype=np.float32),
        "hard_P_s": np.eye(state_dim, dtype=np.float32),
        "hard_P_a": np.eye(action_dim, dtype=np.float32),
        "candidates": [],
        "rejections": [],
        "all_candidates": [],
        "subgroup": {"elements": []},
        "best_holdout_manifold_distance": float("inf"),
        "identity_manifold_distance": float("nan"),
        "rho_mfd": float("nan"),
        "acceptance_tau": 0.0,
        "learned_intertwiner_W": None,
    }


def _merge_discrete_results(
    results: List[dict],
    state_dim: int,
    action_dim: int,
    max_subgroup_elements: int,
) -> dict:
    """Merge discrete results from multiple candidate-batched runs.

    Re-runs global dedup and subgroup closure over the combined candidate pool.
    """
    all_rounded: List[dict] = []
    for r in results:
        all_rounded.extend(r["all_candidates"])

    accepted = []
    rejections = []
    seen: set = set()
    for item in sorted(all_rounded, key=lambda x: x["holdout_manifold_distance"]):
        key_pair = _key_for_pair(item["hard_P_s"], item["hard_P_a"])
        if item["rejection_reason"]:
            rejections.append(item)
            continue
        if key_pair in seen:
            rejections.append({**item, "rejection_reason": "duplicate"})
            continue
        seen.add(key_pair)
        accepted.append(item)

    subgroup = _close_discrete_subgroup(
        accepted, state_dim, action_dim, max_elements=max_subgroup_elements
    )
    best = accepted[0] if accepted else min(all_rounded, key=lambda x: x["holdout_manifold_distance"])
    best_mfd = float(best["holdout_manifold_distance"])
    # identity_manifold_distance is the same for all batches (same model + holdout data)
    identity_mfd = float(results[0].get("identity_manifold_distance", float("nan")))
    rho_mfd = (best_mfd / identity_mfd) if identity_mfd > 0.0 and not math.isnan(identity_mfd) else float("nan")
    return {
        "history": results[0]["history"],
        "soft_P_s": best["soft_P_s"],
        "soft_P_a": best["soft_P_a"],
        "hard_P_s": best["hard_P_s"],
        "hard_P_a": best["hard_P_a"],
        "candidates": accepted,
        "rejections": rejections,
        "all_candidates": all_rounded,
        "subgroup": subgroup,
        "best_holdout_manifold_distance": best_mfd,
        "identity_manifold_distance": identity_mfd,
        "rho_mfd": rho_mfd,
        "acceptance_tau": results[0]["acceptance_tau"],
        "learned_intertwiner_W": results[0]["learned_intertwiner_W"],
    }


def discover_score_symmetries(
    states,
    actions,
    next_states,
    episode_ids,
    step_ids,
    state_dim,
    action_dim,
    device=None,
    config: Optional[ScoreSymmetryDiscoveryConfig] = None,
    verbose=False,
    discrete_log_interval=250,
    only_discrete: bool = False,
    only_continuous: bool = False,
):
    del device, episode_ids, step_ids
    cfg = config or ScoreSymmetryDiscoveryConfig()
    cfg.discrete.log_interval = int(discrete_log_interval)

    states_np = _to_numpy(states).astype(np.float32)
    actions_np = _to_numpy(actions).astype(np.float32)
    next_states_np = _to_numpy(next_states).astype(np.float32)
    z = np.concatenate([states_np, actions_np, next_states_np], axis=-1).astype(np.float32)

    initial_splits = _build_splits(
        num_points=int(z.shape[0]),
        validation_split=float(cfg.training.validation_split),
        holdout_split=float(cfg.training.holdout_split),
        seed=int(cfg.training.seed),
    )
    train_idx = initial_splits["train_idx"]
    z_train = z[train_idx]
    z_mean = z_train.mean(axis=0)
    z_std = z_train.std(axis=0)
    if cfg.training.normalize_data:
        is_const = z_std < float(cfg.training.standardize_eps)
        z_scale = np.where(is_const, 1.0, z_std).astype(np.float32)
        # constant dims (e.g. lifted-dimension = 1): skip mean shift too, or (1-1)/1 = 0
        z_shift = np.where(is_const, 0.0, z_mean).astype(np.float32)
    else:
        z_scale = np.ones_like(z_mean, dtype=np.float32)
        z_shift = np.zeros_like(z_mean, dtype=np.float32)
    z_standardized = ((z - z_shift) / z_scale).astype(np.float32)

    key = jax.random.PRNGKey(int(cfg.training.seed))
    model_key, fit_key = jax.random.split(key)
    model = SigmaConditionedDenoiser(
        data_dim=int(z_standardized.shape[1]),
        hidden_dims=tuple(cfg.training.hidden_dims),
        key=model_key,
    )
    trainer = ScoreDenoiserTrainer(model=model, config=cfg.training, key=fit_key)
    history = trainer.fit(z_standardized)
    model = trainer.model

    split = history["split"]
    z_train_std = z_standardized[split["train_idx"]]
    z_holdout_std = z_standardized[split["holdout_idx"]]
    z_val_std = z_standardized[split["val_idx"]]

    if verbose:
        diag = history["diagnostics"]
        print(
            f"[phase] denoiser trained "
            f"train_mfd={diag['train_manifold_rmse']:.4e} "
            f"holdout_mfd={diag['holdout_manifold_rmse']:.4e}"
        )
    # _ALL_PROJECTOR_VARIANTS = ["default", "raw_P", "raw_S", "M_sym_P"]
    _ALL_PROJECTOR_VARIANTS = ["raw_P"]
    # selected_variant = getattr(cfg.continuous, "projector_variant", "default")
    selected_variant = getattr(cfg.continuous, "projector_variant", "raw_P")
    if only_discrete:
        if verbose:
            print("[phase] continuous score discovery SKIPPED (only_discrete=True)")
        continuous_all_variants = {v: _empty_continuous_result(int(state_dim), int(action_dim)) for v in _ALL_PROJECTOR_VARIANTS}
    else:
        if verbose:
            print("[phase] continuous score discovery start (all projector variants)")
        continuous_all_variants = {}
        for _variant in _ALL_PROJECTOR_VARIANTS:
            if verbose:
                print(f"  [continuous] variant={_variant}")
            _cont_cfg = replace(cfg.continuous, projector_variant=_variant)
            continuous_all_variants[_variant] = discover_continuous_symmetries(
                model=model,
                z_train_std=z_train_std,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                training_cfg=cfg.training,
                config=_cont_cfg,
                verbose=verbose,
            )
    continuous = continuous_all_variants[selected_variant]

    if only_continuous:
        if verbose:
            print("[phase] discrete score discovery SKIPPED (only_continuous=True)")
        discrete = _empty_discrete_result(int(state_dim), int(action_dim))
    else:
        _sigma_eval_discrete = float(cfg.training.sigma_eval) if cfg.training.sigma_eval is not None else math.sqrt(float(cfg.training.sigma_min) * float(cfg.training.sigma_max))
        n_candidates = int(cfg.discrete.num_candidates)
        batch_size = int(cfg.discrete.candidates_batch_size)
        if batch_size <= 0 or batch_size >= n_candidates:
            if verbose:
                print("[phase] discrete score discovery start")
            discrete = MultiDiscreteSymmetryDiscovery(
                model=model,
                z_train_std=z_train_std,
                z_holdout_std=z_holdout_std,
                z_val_std=z_val_std,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                sigma_eval=_sigma_eval_discrete,
                config=cfg.discrete,
                training_history=history,
            ).fit(verbose=verbose)
        else:
            # Run multiple sequential batches of candidates to cap peak memory.
            n_batches = math.ceil(n_candidates / batch_size)
            if verbose:
                print(f"[phase] discrete score discovery start ({n_candidates} candidates in {n_batches} batches of {batch_size})")
            batch_results = []
            remaining = n_candidates
            for batch_idx in range(n_batches):
                this_batch = min(batch_size, remaining)
                batch_cfg = replace(cfg.discrete, num_candidates=this_batch, seed=cfg.discrete.seed + batch_idx * 1000)
                if verbose:
                    print(f"  [discrete] batch {batch_idx + 1}/{n_batches} ({this_batch} candidates)")
                result = MultiDiscreteSymmetryDiscovery(
                    model=model,
                    z_train_std=z_train_std,
                    z_holdout_std=z_holdout_std,
                    z_val_std=z_val_std,
                    state_dim=int(state_dim),
                    action_dim=int(action_dim),
                    sigma_eval=_sigma_eval_discrete,
                    config=batch_cfg,
                    training_history=history,
                ).fit(verbose=verbose)
                batch_results.append(result)
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

    # --- Policy symmetry check (diagnostic only, does not filter symmetries) ---
    s_train_std = z_train_std[:, :int(state_dim)]
    a_train_std = z_train_std[:, int(state_dim):int(state_dim) + int(action_dim)]
    if verbose:
        print("[phase] policy symmetry check start")
    policy_check = _check_policy_symmetry(
        states_std=s_train_std,
        actions_std=a_train_std,
        continuous_generators=continuous["generators"],
        discrete_candidates=discrete["candidates"],
        seed=int(cfg.training.seed),
        verbose=verbose,
    )

    _sigma_eval_used = float(cfg.training.sigma_eval) if cfg.training.sigma_eval is not None else math.sqrt(float(cfg.training.sigma_min) * float(cfg.training.sigma_max))

    return {
        "model": model,
        "training_history": history,
        "standardizer": {
            "mean": z_mean.astype(np.float32),
            "scale": z_scale.astype(np.float32),
        },
        "data": {
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "z_dim": int(z_standardized.shape[1]),
            "z_holdout_std": z_holdout_std.astype(np.float32),
            "sigma_eval": _sigma_eval_used,
        },
        "continuous": continuous,
        "continuous_all_variants": continuous_all_variants,
        "discrete": discrete,
        "group": group,
        "policy_check": policy_check,
    }












# -----------------------------------------------------------------------------
# 1. Normal-frame extraction: symmetric eigendecomposition of (I - P_sym).
# -----------------------------------------------------------------------------
def _compute_normal_frames(
    model,
    z_points: np.ndarray,
    sigma_eval: float,
    state_dim: int,
    frame_batch_size: int,
    use_symmetrized_normal_matrix: bool = True,  # kept for signature parity
):
    """Top-n eigenvectors of the symmetric part of (I - dd/dz).

    The mathematical ground truth -- the orthogonal projector onto the normal
    space of the manifold -- is symmetric. The learned denoiser Jacobian
    P(z) = dd_phi/dz is only approximately a projector, so we take the
    symmetric part of (I - P) and eigendecompose with eigh. The previous
    np.linalg.eig path produced complex eigenvectors and silent QR rotations
    that drifted into noise.
    """
    data_dim = int(z_points.shape[1])

    # Fast path: models that precompute normal frames (e.g. KNNManifoldProjector)
    # expose get_normal_frames() to avoid jacrev through dynamic indexing which
    # causes multi-hour XLA compilation and CPU thrashing.
    # Process in frame_batch_size chunks so large num_frame_points values don't
    # allocate a single (N, d, n_normals) array and use batched QR for speed.
    if hasattr(model, "get_normal_frames"):
        frames = []
        metadata = []
        tangent_evals_empty = np.zeros(max(data_dim - state_dim, 0), dtype=np.float32)
        eigvals_ones = np.ones(state_dim, dtype=np.float32)
        meta_entry = {
            "normal_eigenvalues": eigvals_ones,
            "tangent_eigenvalues": tangent_evals_empty[-3:] if tangent_evals_empty.size else np.zeros(0),
            "spectral_separation": float("nan"),
        }
        for start in range(0, z_points.shape[0], int(frame_batch_size)):
            chunk = z_points[start:start + int(frame_batch_size)]
            batch_nb = model.get_normal_frames(chunk)  # (B, d, n_normals)
            q, _ = np.linalg.qr(batch_nb)             # (B, d, min(d, n_normals)) — batched
            for i in range(len(chunk)):
                frames.append(q[i, :, :state_dim].astype(np.float32))
                metadata.append(meta_entry)
        return np.stack(frames, axis=0), metadata

    def jac_single(x):
        return jax.jacrev(lambda y: model(y, sigma_eval))(x)

    jac_batch = jax.jit(jax.vmap(jac_single))
    frames = []
    metadata = []
    for start in range(0, z_points.shape[0], int(frame_batch_size)):
        batch = jnp.asarray(
            z_points[start:start + int(frame_batch_size)], dtype=jnp.float32
        )
        jacobians = np.asarray(jac_batch(batch))
        for jac in jacobians:
            mat = np.eye(data_dim, dtype=np.float32) - jac
            mat_sym = 0.5 * (mat + mat.T) # symmetrize projection matrix for stable eigendecomposition
            evals, evecs = np.linalg.eigh(mat_sym)
            order = np.argsort(evals)[::-1][:state_dim]
            frame = np.asarray(evecs[:, order], dtype=np.float32)
            eigvals = np.asarray(evals[order], dtype=np.float32)
            # Already orthonormal because eigh returns orthonormal eigenvectors,
            # but re-QR for numerical hygiene.
            q, _ = np.linalg.qr(frame)
            frames.append(q[:, :state_dim].astype(np.float32))
            # Quality of the projector: gap between the n-th and (n+1)-th
            # symmetric eigenvalues. Big gap = clean separation between
            # normal and tangent. Small gap = denoiser hasn't learned the
            # manifold geometry well at this sigma.
            tangent_evals = np.asarray(np.sort(evals)[: data_dim - state_dim], dtype=np.float32)
            # Idempotency check: ‖P_i² - P_i‖_F / ‖P_i‖_F where P_i = jac.
            # For a true orthogonal projector this is 0; ≲0.1 is healthy.
            Pi = jac.astype(np.float64)
            Pi_norm = float(np.linalg.norm(Pi, 'fro'))
            idem_err = (float(np.linalg.norm(Pi @ Pi - Pi, 'fro')) / Pi_norm
                        if Pi_norm > 1e-12 else float("nan"))
            metadata.append(
                {
                    "normal_eigenvalues": eigvals,
                    "tangent_eigenvalues": tangent_evals[-3:] if tangent_evals.size else np.zeros(0),
                    "spectral_separation": float(eigvals.min() - tangent_evals.max())
                    if tangent_evals.size
                    else float("nan"),
                    "projector_idempotency_err": idem_err,
                }
            )
    return np.stack(frames, axis=0), metadata


# -----------------------------------------------------------------------------
# 1b. Raw projector extraction — for projector_variant != "default".
# -----------------------------------------------------------------------------
def _compute_raw_normal_matrices(
    model,
    z_points: np.ndarray,
    sigma_eval: float,
    frame_batch_size: int,
) -> tuple:
    """Return (I - P_i) matrices of shape (N, d, d).

    For KNNManifoldProjector (has get_normal_frames), uses N_i N_i^T (the
    exact normal projector) to avoid the slow jacrev path over dynamic indexing.
    For NN denoisers, computes I - dd_phi/dz via jacrev.

    Returns (mats, metadata) where metadata is a list of stub dicts.
    """
    data_dim = int(z_points.shape[1])
    meta_stub = {
        "normal_eigenvalues": np.zeros(0, dtype=np.float32),
        "tangent_eigenvalues": np.zeros(0, dtype=np.float32),
        "spectral_separation": float("nan"),
    }

    if hasattr(model, "get_normal_frames"):
        mats, metadata = [], []
        for start in range(0, z_points.shape[0], frame_batch_size):
            chunk = z_points[start:start + frame_batch_size]
            nb = model.get_normal_frames(chunk)  # (B, d, n_normals)
            for n_i in nb:
                mats.append((n_i @ n_i.T).astype(np.float32))
                metadata.append(meta_stub)
        return np.stack(mats, axis=0), metadata

    def jac_single(x):
        return jax.jacrev(lambda y: model(y, sigma_eval))(x)

    jac_batch = jax.jit(jax.vmap(jac_single))
    eye = np.eye(data_dim, dtype=np.float32)
    mats, metadata = [], []
    for start in range(0, z_points.shape[0], frame_batch_size):
        batch = jnp.asarray(
            z_points[start:start + frame_batch_size], dtype=jnp.float32
        )
        jacobians = np.asarray(jac_batch(batch))
        for jac in jacobians:
            mats.append(eye - jac.astype(np.float32))
            metadata.append(meta_stub)
    return np.stack(mats, axis=0), metadata


# -----------------------------------------------------------------------------
# 2. Joint normalisation of (L_s, L_a).
# -----------------------------------------------------------------------------
def _normalize_generator(v: np.ndarray, state_dim: int, action_dim: int) -> np.ndarray:
    """Unit-norm the full (L_s, L_a) vector, not just the L_s block.

    The previous version divided by ||L_s||. When the eigenvector has a
    small L_s block (which happens whenever the action-block constraint
    is weak -- e.g. small action-state coupling), this inflated noise in
    L_a by 1/eps and produced bogus generators where L_s was a tiny rotation
    paired with an enormous L_a sign flip.
    """
    v = np.asarray(v, dtype=np.float32)
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-12:
        return v
    return (v / nrm).astype(np.float32)


def _refine_nullspace_by_action_sparsity(
    eigenvectors: np.ndarray,   # (op_dim, k_extended)
    eigenvalues: np.ndarray,    # (k_extended,)  -- for diagnostics only
    state_dim: int,
    action_dim: int,
    n_out: int,                 # how many refined generators to return
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Within span of bottom-k eigenvectors find directions minimising ||L_a||_F.

    The near-null subspace of M_TMT may be degenerate when the action block
    of M is weakly constrained (L_a ≈ 0 for the true symmetry, or actions are
    free coordinates on the manifold).  eigh returns an arbitrary orthonormal
    basis for this subspace; the true generator is a specific linear combination
    that minimises action-block energy.  This secondary eigenvalue problem
    recovers it without enforcing any constraint on L_a.
    """
    n, m = state_dim, action_dim
    la_start = n ** 2
    la_end   = n ** 2 + m ** 2
    k = eigenvectors.shape[1]
    # op_dim = eigenvectors.shape[0]

    # Action-block rows of the subspace basis
    V_action = eigenvectors[la_start:la_end, :]   # (m², k)

    # Secondary eigenvalue problem: min_{||α||=1} ||V_action α||²
    # = min eigenvector of G = V_action.T V_action  (k×k, cheap)
    G = V_action.T @ V_action
    sub_evals, sub_evecs = np.linalg.eigh(G)       # ascending order

    if verbose:
        ns = [float(np.linalg.norm(eigenvectors[:la_start, i]))  for i in range(k)]
        na = [float(np.linalg.norm(eigenvectors[la_start:la_end, i])) for i in range(k)]
        print(f"\n[nullspace refine] original basis (k={k}):")
        print(f"  {'i':>3}  {'λ_M':>11}  {'||Ls||':>8}  {'||La||':>8}  {'la/ls':>7}")
        for i in range(k):
            ratio = na[i] / (ns[i] + 1e-12)
            print(f"  {i:>3}  {eigenvalues[i]:>11.4e}  {ns[i]:>8.4f}  {na[i]:>8.4f}  {ratio:>7.4f}")
        print(f"  secondary evals (action-block energy): {sub_evals}")

    refined = []
    for i in range(min(n_out, k)):
        alpha = sub_evecs[:, i]           # minimises action-block norm
        v = eigenvectors @ alpha
        v = v / (float(np.linalg.norm(v)) + 1e-12)
        refined.append(v.astype(np.float32))

    diag = {
        "secondary_eigenvalues": sub_evals.tolist(),
        "secondary_eigenvectors": sub_evecs.tolist(),
    }
    return np.stack(refined, axis=1), diag   # (op_dim, n_out)

# -----------------------------------------------------------------------------
# 3. Spectral-gap detection: absolute + relative criterion in log-space.
# -----------------------------------------------------------------------------
def _pick_generator_count(
    eigenvalues: np.ndarray,
    max_generators: int,
    spectral_gap_min_ratio: float,
    abs_threshold_frac: float = 5e-2,
):
    """Pick k by combining an absolute and a log-relative gap.

    Heuristic:
      - A candidate eigenvalue is "small" if it is below abs_threshold_frac
        times the largest eigenvalue. Otherwise we have no symmetry.
      - Among small eigenvalues, pick the largest log-gap to the next one.
      - If the log-gap is below log(spectral_gap_min_ratio), the spectrum
        has no clean separation -- return min(k_small, 1) and warn via the
        diagnostics dict.
    """
    print(f"eigenvalues: {eigenvalues}")

    if eigenvalues.size == 0 or max_generators <= 0:
        return 0, {"gap_index": -1, "gap_ratio": 0.0, "k_small": 0, "ev size": eigenvalues.size, "warning": "empty"}

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eps = 1e-30
    lam_max = float(eigenvalues[-1]) if eigenvalues.size else 1.0
    if lam_max <= 0:
        lam_max = 1.0
    abs_cut = abs_threshold_frac * lam_max
    k_small = int(np.searchsorted(eigenvalues, abs_cut))
    k_small = max(0, min(k_small, max_generators))
    if k_small == 0:
        return 0, {
            "gap_index": -1,
            "gap_ratio": 0.0,
            "k_small": 0,
            "lam_max": lam_max,
            "abs_cut": abs_cut,
            "warning": "no eigenvalue below absolute threshold; no symmetry",
        }
    if k_small == 1:
        return 1, {"gap_index": 0, "gap_ratio": float("inf"), "k_small": 1}

    log_e = np.log(np.maximum(eigenvalues, 0.0) + eps)
    diffs = log_e[1:k_small + 1] - log_e[:k_small]  # log(lambda_{i+1}/lambda_i)
    if diffs.size == 0:
        return k_small, {"gap_index": k_small - 1, "gap_ratio": float("inf"), "k_small": k_small}
    best = int(np.argmax(diffs))
    best_ratio = float(np.exp(diffs[best]))
    if best_ratio < float(spectral_gap_min_ratio):
        return k_small, {
            "gap_index": best,
            "gap_ratio": best_ratio,
            "k_small": k_small,
            "warning": f"weak gap inside small block (ratio={best_ratio:.2f}); using k_small={k_small}",
        }
    return best + 1, {"gap_index": best, "gap_ratio": best_ratio, "k_small": k_small}


# -----------------------------------------------------------------------------
# 4. Lie-bracket closure with M-norm sanity check.
# _split_generator / _pack_generator / _generator_bracket defined above (section 3).


def _projection_residual(vec: np.ndarray, basis: Sequence[np.ndarray]):
    if not basis:
        return vec, np.zeros((0,), dtype=np.float32)
    B = np.stack(basis, axis=1)
    coeff, *_ = np.linalg.lstsq(B, vec, rcond=None)
    recon = B @ coeff
    return vec - recon, coeff


def _complete_lie_closure(
    vectors: Sequence[np.ndarray],
    state_dim: int,
    action_dim: int,
    tol: float,
    max_generators: int,
    M: np.ndarray,
    bracket_quality_factor: float = 5.0,
    affine: bool = False,
):
    """Add bracket residuals only if they themselves satisfy the constraint.

    The previous version added any bracket whose projection residual was
    above tol, regardless of whether the bracket was a valid symmetry. With
    noisy generators that produced runaway 16-element bases. We now require
    that v^T M v on the candidate residual is no worse than
    bracket_quality_factor times the worst objective among the seed
    generators -- otherwise the bracket is rejected as an artefact of noise.
    """
    M32 = np.asarray(M, dtype=np.float32)
    seed_objs = []
    basis = []
    for v in vectors:
        vn = _normalize_generator(np.asarray(v, dtype=np.float32), state_dim, action_dim)
        basis.append(vn)
        seed_objs.append(float(vn @ (M32 @ vn)))
    if not seed_objs:
        return basis, {
            "closure_error_before_completion": 0.0,
            "closure_error": 0.0,
            "added_generators": [],
            "rejected_brackets": [],
        }
    seed_obj_max = max(seed_objs) + 1e-12
    quality_cut = bracket_quality_factor * seed_obj_max

    added = []
    rejected = []
    changed = True
    max_error = 0.0
    while changed and len(basis) < int(max_generators):
        changed = False
        current = list(basis)
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                bracket = _generator_bracket(current[i], current[j], state_dim, action_dim, affine)
                bracket_norm = float(np.linalg.norm(bracket))
                if bracket_norm <= 1e-10:
                    continue
                residual, coeff = _projection_residual(bracket, basis)
                err = float(np.linalg.norm(residual) / (bracket_norm + 1e-12))
                max_error = max(max_error, err)
                if err <= float(tol):
                    continue  # bracket is already in the span -- closure OK
                # Candidate new generator: normalise and check that it itself
                # is approximately a symmetry.
                cand = _normalize_generator(residual, state_dim, action_dim)
                cand_obj = float(cand @ (M32 @ cand))
                if cand_obj > quality_cut:
                    rejected.append(
                        {
                            "from_pair": (int(i), int(j)),
                            "closure_residual": err,
                            "objective": cand_obj,
                            "quality_cut": float(quality_cut),
                            "reason": "bracket residual is not a valid symmetry",
                        }
                    )
                    continue
                if any(abs(float(cand @ b)) > 0.999 for b in basis):
                    continue  # duplicate
                basis.append(cand)
                added.append(
                    {
                        "from_pair": (int(i), int(j)),
                        "closure_error": err,
                        "objective": cand_obj,
                        "coefficients": np.asarray(coeff, dtype=np.float32),
                    }
                )
                changed = True
                if len(basis) >= int(max_generators):
                    break
            if len(basis) >= int(max_generators):
                break
    final_error = 0.0
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            bracket = _generator_bracket(basis[i], basis[j], state_dim, action_dim, affine)
            bn = float(np.linalg.norm(bracket))
            if bn <= 1e-10:
                continue
            residual, _ = _projection_residual(bracket, basis)
            final_error = max(final_error, float(np.linalg.norm(residual) / (bn + 1e-12)))
    return basis, {
        "closure_error_before_completion": float(max_error),
        "closure_error": float(final_error),
        "added_generators": added,
        "rejected_brackets": rejected,
        "seed_objective_max": float(seed_obj_max),
        "bracket_quality_cut": float(quality_cut),
    }


# -----------------------------------------------------------------------------
# 5. Continuous discovery: better sigma_eval default + per-generator quality.
# -----------------------------------------------------------------------------
def _block_norms(v: np.ndarray, state_dim: int, action_dim: int) -> Tuple[float, float]:
    n, m = state_dim, action_dim
    v = np.asarray(v, dtype=np.float32)
    L_s = v[:n ** 2].reshape(n, n)
    L_a = v[n ** 2: n ** 2 + m ** 2].reshape(m, m)
    return float(np.linalg.norm(L_s)), float(np.linalg.norm(L_a))


def _polish_generator(
    init_v, A_stack, state_dim, action_dim,
    polish_steps, polish_lr, polish_batch_size, seed,
):
    """Re-uses the original polish; kept here for completeness."""
    if polish_steps <= 0 or A_stack.shape[0] == 0:
        return init_v
    opt = optax.adam(float(polish_lr))
    v = jnp.asarray(init_v, dtype=jnp.float32)
    opt_state = opt.init(v)
    key = jax.random.PRNGKey(int(seed))

    @jax.jit
    def step(v, opt_state, batch):
        def loss_fn(x):
            proj = jnp.einsum("bij,j->bi", batch, x)
            loss = jnp.mean(jnp.sum(proj ** 2, axis=-1))
            loss += 1e-2 * (jnp.sum(x ** 2) - 1.0) ** 2
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(v)
        updates, opt_state = opt.update(grads, opt_state, v)
        v = optax.apply_updates(v, updates)
        v = v / (jnp.linalg.norm(v) + 1e-12)
        return v, opt_state, loss

    for _ in range(int(polish_steps)):
        key, subkey = jax.random.split(key)
        if A_stack.shape[0] <= int(polish_batch_size):
            batch = jnp.asarray(A_stack, dtype=jnp.float32)
        else:
            idx = np.asarray(
                jax.random.randint(subkey, (int(polish_batch_size),), 0, int(A_stack.shape[0]))
            )
            batch = jnp.asarray(A_stack[idx], dtype=jnp.float32)
        v, opt_state, _ = step(v, opt_state, batch)
    return _normalize_generator(np.asarray(v), state_dim, action_dim)



def orbit_score(v, model, z_data, state_dim, action_dim, sigma_eval,
                t_values=(0.1, 0.3, 0.5, 1.0), batch_size=4096):
    """Manifold distance after applying exp(t*v) to data, averaged over t."""
    n, m = state_dim, action_dim
    Ls = v[:n**2].reshape(n, n).astype(np.float64)
    La = v[n**2:n**2+m**2].reshape(m, m).astype(np.float64)
    z = np.asarray(z_data, dtype=np.float32)
    s, a, sp = z[:, :n], z[:, n:n+m], z[:, n+m:]

    scores = []
    for t in t_values:
        Gs = np.asarray(jax.scipy.linalg.expm(t * Ls), dtype=np.float32)
        Ga = np.asarray(jax.scipy.linalg.expm(t * La), dtype=np.float32)
        z_t = np.concatenate([s @ Gs.T, a @ Ga.T, sp @ Gs.T], axis=-1)
        scores.append(_batched_manifold_rmse(model, z_t, sigma_eval, batch_size))
    # Also include -t to symmetrise (real generators are invariant under sign)
    return float(np.mean(scores)), scores  # mean over t, per-t values


def orbit_refine_subspace(
    V_K,                          # (op_dim, K) — bottom-K eigenvectors from eigh(M, N)
    model, z_data,
    state_dim, action_dim, sigma_eval,
    n_inits=8,                    # random starts in K-dim alpha-space
    n_steps=300,
    lr=3e-3,
    batch_size=1024,
    t_values=(0.25, 0.5, 1.0),
    dedup_threshold=0.9,          # |alpha_i · alpha_j| above this is a duplicate
    seed=0,
    verbose=False,
):
    """Find a basis of span(V_K) by minimising orbit error, with K-dim sphere
    constraints and multi-start deduplication.  Pure post-processing — does not
    enforce any structural prior on the output."""
    n, m = int(state_dim), int(action_dim)
    op_dim, K = V_K.shape
    V_K_np = np.asarray(V_K, dtype=np.float32)
    V_K_j  = jnp.asarray(V_K_np)
    z = jnp.asarray(z_data, dtype=jnp.float32)
    t_arr = jnp.asarray(t_values, dtype=jnp.float32)
    bs = min(int(batch_size), int(z.shape[0]))

    @eqx.filter_jit
    def _step(model, alpha, opt_state, batch, optimizer):
        def loss_fn(a):
            v_raw = V_K_j @ a
            v = v_raw / (jnp.linalg.norm(v_raw) + 1e-12)
            Ls = v[:n * n].reshape(n, n)
            La = v[n * n: n * n + m * m].reshape(m, m)
            s_, ac_, sp_ = batch[:, :n], batch[:, n:n + m], batch[:, n + m:]

            def per_t(t):
                Gs = jax.scipy.linalg.expm(t * Ls)
                Ga = jax.scipy.linalg.expm(t * La)
                z_t = jnp.concatenate([s_ @ Gs.T, ac_ @ Ga.T, sp_ @ Gs.T], axis=-1)
                pred = model(z_t, sigma_eval)
                return jnp.mean(jnp.sum((pred - z_t) ** 2, axis=-1))

            return jnp.mean(jax.vmap(per_t)(t_arr))

        loss, grads = jax.value_and_grad(loss_fn)(alpha)
        grads = grads - alpha * jnp.dot(grads, alpha)       # K-sphere projection
        updates, opt_state = optimizer.update(grads, opt_state, alpha)
        alpha = alpha + updates
        alpha = alpha / (jnp.linalg.norm(alpha) + 1e-12)
        return alpha, opt_state, loss

    rng = np.random.default_rng(int(seed))
    candidates = []
    for init_i in range(int(n_inits)):
        alpha0 = rng.standard_normal(K).astype(np.float32)
        alpha0 /= np.linalg.norm(alpha0) + 1e-12
        alpha = jnp.asarray(alpha0)
        optimizer = optax.adam(float(lr))
        opt_state = optimizer.init(alpha)
        key = jax.random.PRNGKey(int(seed) + 31 * init_i)
        for _ in range(int(n_steps)):
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (bs,), 0, int(z.shape[0]))
            alpha, opt_state, _ = _step(model, alpha, opt_state, z[idx], optimizer)
        alpha_np = np.asarray(alpha)
        v = V_K_np @ alpha_np
        v = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        orbit_mean, orbit_per_t = orbit_score(
            v, model, z_data, n, m, sigma_eval, t_values=t_values,
        )
        candidates.append({
            "v": v, "alpha": alpha_np.astype(np.float32),
            "orbit_score": orbit_mean, "orbit_per_t": orbit_per_t,
        })
        if verbose:
            print(f"  init[{init_i}] orbit_score={orbit_mean:.3e}")

    # Rank, dedupe in full generator space (v vectors, already unit-norm).
    # Alpha-space dedup fails when two distinct alpha directions project through
    # V_K to nearly-identical generators — e.g. different 2D sub-block embeddings
    # of the same rotation all score well and look distinct in alpha-space.
    candidates.sort(key=lambda d: d["orbit_score"])
    kept = []
    for c in candidates:
        if any(abs(float(c["v"] @ k["v"])) > float(dedup_threshold) for k in kept):
            continue
        kept.append(c)
    return kept


def discover_continuous_symmetries(
    model,
    z_train_std: np.ndarray,
    state_dim: int,
    action_dim: int,
    training_cfg,
    config,
    verbose=False,
    *,
    precomputed_M: Optional[np.ndarray] = None,
    # extra knobs surfaced for tuning
    min_state_norm: float = 1e-2,
    abs_threshold_frac: float = 1e-1,
):
    # If sigma_eval not set, use geometric mean of [sigma_min, sigma_max].
    if config.sigma_eval is None:
        sigma_eval = float(
            math.sqrt(float(training_cfg.sigma_min) * float(training_cfg.sigma_max))
        )
    else:
        sigma_eval = float(config.sigma_eval)

    # ---- frame / projector extraction + M assembly ----
    n_frames = min(int(config.num_frame_points), int(z_train_std.shape[0]))
    frame_idx = (
        np.linspace(0, z_train_std.shape[0] - 1, n_frames, dtype=np.int64)
        if n_frames > 0
        else np.zeros((0,), dtype=np.int64)
    )
    z_subset = np.asarray(z_train_std[frame_idx], dtype=np.float32)

    affine = bool(getattr(config, "affine", False))
    operator_dim = state_dim ** 2 + action_dim ** 2 + (state_dim + action_dim if affine else 0)
    print(
        f"[continuous/score] state_dim={state_dim} action_dim={action_dim} "
        f"operator_dim={operator_dim} affine={affine}"
    )
    
    variant = getattr(config, "projector_variant", "default")

    # ---- build sigma schedule ----
    multiscale_n = int(getattr(config, "multiscale_sigmas", 1))
    if multiscale_n <= 1:
        sigma_list = [sigma_eval]
    else:
        log_lo = math.log(float(training_cfg.sigma_min))
        log_hi = math.log(float(training_cfg.sigma_max))
        sigma_list = np.exp(np.linspace(log_lo, log_hi, multiscale_n)).tolist()
        if verbose:
            print(
                f"[continuous/score] multi-scale M: {multiscale_n} sigmas "
                f"[{sigma_list[0]:.3e} .. {sigma_list[-1]:.3e}]"
            )

    _precomputed = precomputed_M is not None
    M_acc = np.zeros((operator_dim, operator_dim), dtype=np.float64)
    A_stack_list = []
    frame_meta = []


    for sig in ([] if _precomputed else sigma_list):
        M_sig = np.zeros((operator_dim, operator_dim), dtype=np.float64)

        if variant == "default":
            frames_sig, meta_sig = _compute_normal_frames(
                model,
                z_subset,
                sigma_eval=sig,
                state_dim=state_dim,
                frame_batch_size=int(config.frame_batch_size),
                use_symmetrized_normal_matrix=True,
            )
            for z_i, N_i in zip(z_subset, frames_sig):
                A_i = N_i.T @ _ctilde(z_i, state_dim, action_dim, affine)
                A_stack_list.append(A_i.astype(np.float32))
                M_sig += A_i.T @ A_i
            if n_frames > 0:
                M_sig /= float(n_frames)
        else:
            # Ablation variants: compute full (I - P_i) matrices, skip rank truncation.
            raw_mats, meta_sig = _compute_raw_normal_matrices(
                model, z_subset, sig, int(config.frame_batch_size)
            )
            for z_i, mat in zip(z_subset, raw_mats):
                C_i = _ctilde(z_i, state_dim, action_dim, affine).astype(np.float64)
                if variant == "M_sym_P":
                    # Linear use of (I-P_i): M_raw += C^T (I-P) C, symmetrise at end.
                    M_sig += C_i.T @ mat.astype(np.float64) @ C_i
                else:
                    # Quadratic use: M += A^T A where A = proj @ C.
                    proj = 0.5 * (mat + mat.T) if variant == "raw_S" else mat
                    A_i = proj.astype(np.float64) @ C_i  # (d, operator_dim)
                    A_stack_list.append(A_i.astype(np.float32))
                    M_sig += A_i.T @ A_i
            if n_frames > 0:
                M_sig /= float(n_frames)
            if variant == "M_sym_P":
                M_sig = 0.5 * (M_sig + M_sig.T)

        # Normalise by trace so each scale contributes equally when aggregating.
        if multiscale_n > 1:
            tr = float(np.trace(M_sig))
            if tr > 1e-30:
                M_sig /= tr

        M_acc += M_sig
        frame_meta.extend(meta_sig)

    if _precomputed:
        M = np.asarray(precomputed_M, dtype=np.float64)
    else:
        M = M_acc / len(sigma_list)

        # # — compute M on the second half of frames independently:
        # if z_cv is not None:
        #     M_cross = _compute_M_on_frames(
        #         model, z_cv, state_dim, action_dim,
        #         sigma_list, variant, int(config.frame_batch_size), affine, multiscale_n,
        #     )
        #     if verbose:
        #         print(f"[crossval] M_cross computed on {len(z_cv)} independent frames")
        # else:
        #     M_cross = None


    # A_stack used by _polish_generator; empty array disables polishing.
    A_stack = (
        np.array(A_stack_list)
        if A_stack_list
        else np.zeros((0, 1, operator_dim), dtype=np.float32)
    )

    if affine and verbose:
        n, m = state_dim, action_dim
        assert M.shape == (operator_dim, operator_dim), (
            f"M shape {M.shape} != ({operator_dim},{operator_dim}) — affine off?"
        )

        def _pack_affine(Ls, La, bs, ba):
            v = np.concatenate([Ls.reshape(-1), La.reshape(-1), bs, ba]).astype(np.float64)
            nrm = np.linalg.norm(v)
            return v / nrm if nrm > 1e-12 else v

        v_tx = _pack_affine(np.zeros((n, n)), np.zeros((m, m)),
                            np.eye(n)[0], np.zeros(m))
        v_ty = _pack_affine(np.zeros((n, n)), np.zeros((m, m)),
                            np.eye(n)[1], np.zeros(m))
        Ls_rot = np.zeros((n, n)); Ls_rot[0, 1] = -1; Ls_rot[1, 0] = 1
        if n >= 4:
            Ls_rot[2, 3] = -1; Ls_rot[3, 2] = 1
        La_rot = np.zeros((m, m)); La_rot[0, 1] = -1; La_rot[1, 0] = 1
        v_rot = _pack_affine(Ls_rot, La_rot, np.zeros(n), np.zeros(m))
        print("[continuous/score] affine M sanity:")
        for name, v in [("rot", v_rot), ("tx", v_tx), ("ty", v_ty)]:
            print(f"  v^T M v [{name}] = {float(v @ M @ v):.3e}")
        print(f"  bottom 6 eigvals: {np.sort(np.linalg.eigvalsh(M))[:6]}")


    # Build N alongside M
    N = np.zeros_like(M)
    for z_i in z_subset:
        C_i = _ctilde(z_i, state_dim, action_dim, affine).astype(np.float64)
        N += C_i.T @ C_i
    N /= float(n_frames)
    print(f"N spectrum: {np.linalg.eigvalsh(N)}")


    # NEW
    from scipy.linalg import eigh as scipy_eigh
    # make N positive definite for the GEP solver, by adding a small multiple of its trace to the diagonal. just in case
    N = N + 1e-6 * (np.trace(N) / N.shape[0]) * np.eye(N.shape[0])

    # eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues, eigenvectors = scipy_eigh(M, N)  
    # eigenvalues, eigenvectors = np.linalg.eigh(M, N)   # solves M v = λ N v

    k, gap_info = _pick_generator_count(
        eigenvalues,
        int(config.max_generators),
        float(config.spectral_gap_min_ratio),
        abs_threshold_frac=abs_threshold_frac,
    )
    
    # print GAP specturm before refinement
    if verbose:
        k_show = min(12, operator_dim)
        la_start = state_dim ** 2
        la_end   = state_dim ** 2 + action_dim ** 2
        print(f"\n[M_TMT spectrum] bottom {k_show} eigenvectors:")
        print(f"  {'i':>3}  {'lambda':>12}  {'||Ls||':>9}  {'||La||':>9}  "
              f"{'la/ls':>7}  {'skew_s':>8}")
        for i in range(k_show):
            v  = eigenvectors[:, i]
            Ls = v[:la_start].reshape(state_dim, state_dim)
            La = v[la_start:la_end].reshape(action_dim, action_dim)
            nLs   = float(np.linalg.norm(Ls))
            nLa   = float(np.linalg.norm(La))
            skew  = (float(np.linalg.norm(Ls - Ls.T)) / (nLs + 1e-12))
            ratio = nLa / (nLs + 1e-12)
            print(f"  {i:>3}  {eigenvalues[i]:>12.4e}  {nLs:>9.4f}  {nLa:>9.4f}  "
                  f"{ratio:>7.4f}  {skew:>8.4f}")
        print()


    n_subspace = min(int(k) + 4, eigenvectors.shape[1])
    V_K = eigenvectors[:, :n_subspace]                  # (op_dim, n_subspace)
    orbit_kept = orbit_refine_subspace(
        V_K, model, z_train_std, state_dim, action_dim, sigma_eval,
        n_inits=max(2 * n_subspace, 16),
        # n_steps=300,
        n_steps=2000,
        verbose=verbose,
    )

    # 4. Diagnostic: baselines for orbit score (so you can read the absolute numbers)
    identity_orbit, _ = orbit_score(
        np.zeros(operator_dim, dtype=np.float32),
        model, z_train_std, state_dim, action_dim, sigma_eval,
    )
    rng_diag = np.random.default_rng(int(training_cfg.seed) + 7)
    random_orbits = []
    for _ in range(3):
        vr = rng_diag.standard_normal(operator_dim).astype(np.float32)
        vr /= np.linalg.norm(vr) + 1e-12
        random_orbits.append(orbit_score(vr, model, z_train_std, state_dim, action_dim, sigma_eval)[0])
    print(f"[orbit baselines] identity={identity_orbit:.3e}  "
        f"random_unit_mean={float(np.mean(random_orbits)):.3e}")
    for i, c in enumerate(orbit_kept):
        print(f"  orbit_refined[{i}] score={c['orbit_score']:.3e}  per_t={c['orbit_per_t']}")

    # 5. Re-pack the refined directions as a 2D array for the existing downstream code
    eigenvectors = np.stack([c["v"] for c in orbit_kept], axis=1)   # (op_dim, n_kept)
    n_avail = eigenvectors.shape[1]
    k = min(int(k), n_avail)    

    # 6. Display post-refine spectrum (uses orbit ranking, since GEP eigenvalues
    #    no longer correspond column-for-column to the new vectors)
    if verbose:
        k_show = min(12, n_avail)
        print(f"\n[Post-refine, ranked by orbit] bottom {k_show}:")
        print(f"  {'i':>3}  {'orbit':>11}  {'||Ls||':>9}  {'||La||':>9}  {'la/ls':>7}  {'skew_s':>8}")
        for i in range(k_show):
            v  = eigenvectors[:, i]
            Ls = v[:la_start].reshape(state_dim, state_dim)
            La = v[la_start:la_end].reshape(action_dim, action_dim)
            nLs = float(np.linalg.norm(Ls)); nLa = float(np.linalg.norm(La))
            skew = float(np.linalg.norm(Ls - Ls.T)) / (nLs + 1e-12)
            ratio = nLa / (nLs + 1e-12)
            print(f"  {i:>3}  {orbit_kept[i]['orbit_score']:>11.4e}  {nLs:>9.4f}  {nLa:>9.4f}  {ratio:>7.4f}  {skew:>8.4f}")


    # ---- raw generator readout ----
    raw_vectors = []
    raw_block_ratios = []
    M32 = M.astype(np.float32)
    for idx in range(int(k)):
        v = _normalize_generator(eigenvectors[:, idx], state_dim, action_dim)
        v = _polish_generator(
            v,
            np.asarray(A_stack, dtype=np.float32),
            state_dim,
            action_dim,
            polish_steps=int(config.polish_steps),
            polish_lr=float(config.polish_lr),
            polish_batch_size=int(config.polish_batch_size),
            seed=int(training_cfg.seed) + idx + 17,
        )
        nLs, nLa = _block_norms(v, state_dim, action_dim)
        raw_block_ratios.append((nLs, nLa))
        raw_vectors.append(v)

    # ---- near-trivial-state filter -------------------------------------------
    # Drop generators whose state block is essentially zero (they act trivially on the state and are always useless).
    # NOTE: L_a = 0 is a *valid* solution when actions are invariant under the symmetry (e.g. joint torques are scalars under a rotation). 
    # We therefore gate on ||L_s|| only, never on ||L_a||.
    # Exception: affine mode skips this filter (pure translations have L_s = 0).
    _min_state_norm = float(getattr(config, "min_state_norm", min_state_norm))
    if affine:
        kept_vectors = list(raw_vectors)
        dropped = []
    else:
        kept_vectors = []
        dropped = []
        for v, (ns, na) in zip(raw_vectors, raw_block_ratios):
            if ns >= _min_state_norm:
                kept_vectors.append(v)
            else:
                block_ratio = min(ns, na) / max(ns, na, 1e-12)
                dropped.append({"L_s_norm": ns, "L_a_norm": na, "block_ratio": block_ratio})


    # ---- closure with M-norm sanity check ------------------------------------
    closure_vectors, closure_diag = _complete_lie_closure(
        kept_vectors,
        state_dim=state_dim,
        action_dim=action_dim,
        tol=float(config.closure_tol),
        max_generators=int(config.closure_max_generators),
        M=M32,
        bracket_quality_factor=5.0,
        affine=affine,
    )

    do_intertwiner = getattr(config, "report_intertwiner_quality", True)
    max_sv_res     = float(getattr(config, "max_intertwiner_residual", float("inf")))
    fix_La         = getattr(config, "fix_L_a_from_intertwiner", False)

    generators = []
    for idx, vec in enumerate(closure_vectors):
        if affine:
            L_s, L_a, b_s, b_a = _split_generator(vec, state_dim, action_dim, affine=True)
        else:
            L_s, L_a = _split_generator(vec, state_dim, action_dim)
            b_s = np.zeros(state_dim, dtype=np.float32)
            b_a = np.zeros(action_dim, dtype=np.float32)
        nLs = float(np.linalg.norm(L_s))
        nLa = float(np.linalg.norm(L_a))

        if affine:
            nBs = float(np.linalg.norm(b_s))
            nBa = float(np.linalg.norm(b_a))
            state_total  = math.sqrt(nLs ** 2 + nBs ** 2)
            action_total = math.sqrt(nLa ** 2 + nBa ** 2)
            block_ratio  = min(state_total, action_total) / max(state_total, action_total, 1e-12)
        else:
            nBs = nBa = 0.0
            block_ratio = min(nLs, nLa) / max(nLs, nLa, 1e-12)

        entry = {
            "index": int(idx),
            "L_s": L_s.astype(np.float32),
            "L_a": L_a.astype(np.float32),
            "b_s": b_s.astype(np.float32),
            "b_a": b_a.astype(np.float32),
            "objective": float(vec @ (M32 @ vec)),
            "from_closure": bool(idx >= len(kept_vectors)),
            "state_norm": nLs,
            "action_norm": nLa,
            "b_s_norm": nBs,
            "b_a_norm": nBa,
            "block_ratio": block_ratio,
            "skew_part_frac_s": float(np.linalg.norm(L_s - L_s.T) / (np.linalg.norm(L_s) + 1e-12)),
            "skew_part_frac_a": float(np.linalg.norm(L_a - L_a.T) / (np.linalg.norm(L_a) + 1e-12)),
        }

        generators.append(entry)

    # Compute a single shared intertwiner W for all generators simultaneously.
    # Stacks the Kronecker systems for all (L_s^k, L_a^k) pairs so that the
    # resulting W satisfies L_a^k W ≈ W L_s^k for every k, not just each alone.
    if do_intertwiner and generators:
        W_joint, joint_min_sv, joint_residuals = _joint_intertwiner_quality(
            [g["L_s"] for g in generators],
            [g["L_a"] for g in generators],
        )
        W_pinv = np.linalg.pinv(W_joint) if fix_La else None
        for g, res in zip(generators, joint_residuals):
            g["intertwiner_W"] = W_joint
            g["intertwiner_min_sv"] = joint_min_sv
            g["intertwiner_residual"] = res
            if fix_La and float(np.linalg.norm(g["L_s"])) > 1e-12:
                g["L_a"] = (W_joint @ np.asarray(g["L_s"], dtype=np.float64) @ W_pinv).astype(np.float32)
                g["L_a_fixed_from_intertwiner"] = True

    # Optional filter by intertwiner residual.
    if do_intertwiner and np.isfinite(max_sv_res):
        before = len(generators)
        generators = [g for g in generators
                      if g.get("intertwiner_residual", 0.0) <= max_sv_res]
        if verbose and len(generators) < before:
            print(
                f"[continuous/score] intertwiner filter removed "
                f"{before - len(generators)} generator(s) "
                f"(residual > {max_sv_res:.2f})"
            )

    if verbose:
        warn = gap_info.get("warning", "")
        print(
            f"[continuous/score] variant={variant} sigma_eval={sigma_eval:.3e} "
            f"frames={n_frames} k_small={gap_info.get('k_small', 0)} "
            f"k_chosen={k} k_kept_after_block_filter={len(kept_vectors)} "
            f"k_after_closure={len(generators)} "
            f"gap_ratio={gap_info['gap_ratio']:.3e} "
            f"closure_error={closure_diag['closure_error']:.3e}"
        )
        if warn:
            print(f"[continuous/score] WARNING: {warn}")
        if dropped:
            print(f"[continuous/score] dropped {len(dropped)} block-decoupled directions")
        if do_intertwiner and generators:
            for g in generators:
                print(
                    f"  gen[{g['index']}] intertwiner_residual={g.get('intertwiner_residual', float('nan')):.3e} "
                    f"min_sv={g.get('intertwiner_min_sv', float('nan')):.3e}"
                )

    idem_errs = [m["projector_idempotency_err"] for m in frame_meta
                 if "projector_idempotency_err" in m
                 and not np.isnan(m["projector_idempotency_err"])]
    mean_idem_err = float(np.mean(idem_errs)) if idem_errs else float("nan")
    max_idem_err  = float(np.max(idem_errs))  if idem_errs else float("nan")

    return {
        "generators": generators,
        "frame_indices": frame_idx,
        "normal_frame_metadata": frame_meta,
        "spectrum": eigenvalues.astype(np.float32),
        "matrix_M": M.astype(np.float32),
        "picked_k": int(k),
        "spectral_gap_index": int(gap_info["gap_index"]),
        "spectral_gap_ratio": float(gap_info["gap_ratio"]),
        "closure_error": float(closure_diag["closure_error"]),
        "closure_error_before_completion": float(closure_diag["closure_error_before_completion"]),
        "closure_added_generators": closure_diag["added_generators"],
        "closure_rejected_brackets": closure_diag.get("rejected_brackets", []),
        "block_decoupled_dropped": dropped,
        "sigma_eval": sigma_eval,
        "gap_info": gap_info,
        "projector_variant": variant,
        "mean_projector_idempotency_err": mean_idem_err,
        "max_projector_idempotency_err":  max_idem_err,
    }





# =============================================================================
# GEOMETRIC INTERTWINER QUALITY  (for continuous generators)
# =============================================================================

def _intertwiner_quality(
    L_s: np.ndarray,
    L_a: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Find the best linear map W: R^{d_s}→R^{d_a} satisfying G_a W = W G_s.

    Solves the homogeneous Sylvester equation via SVD of the Kronecker system

        K = I_{d_s} ⊗ L_a − L_s^T ⊗ I_{d_a}   (shape d_s·d_a × d_s·d_a)

    The right singular vector of K for the smallest singular value is vec(W*)
    (column-major), giving the minimum-residual intertwiner W*.

    Returns
    -------
    W_best      : (d_a, d_s) unit-norm intertwiner
    min_sv      : smallest singular value of K (near 0 → good intertwiner exists)
    residual    : ||L_a W* − W* L_s||_F / (||W*||_F + 1e-12)
    """
    Ls = np.asarray(L_s, dtype=np.float64)
    La = np.asarray(L_a, dtype=np.float64)
    d_s, d_a = Ls.shape[0], La.shape[0]

    K = (
        np.kron(np.eye(d_s), La)       # I_{d_s} ⊗ L_a : (d_s*d_a, d_s*d_a)
        - np.kron(Ls.T, np.eye(d_a))  # L_s^T ⊗ I_{d_a}: same shape
    )
    _, sv, Vt = np.linalg.svd(K, full_matrices=False)
    w_vec = Vt[-1]                                     # (d_s*d_a,) — column-major vec(W*)
    W = w_vec.reshape(d_a, d_s, order="F").astype(np.float32)
    w_nrm = float(np.linalg.norm(W))
    if w_nrm > 1e-12:
        W = W / w_nrm
    res = La @ W - W @ Ls
    residual = float(np.linalg.norm(res)) / (float(np.linalg.norm(W)) + 1e-12)
    return W, float(sv[-1]), residual


def _joint_intertwiner_quality(
    L_s_list: list,
    L_a_list: list,
) -> Tuple[np.ndarray, float, list]:
    """Find a single shared W: R^{d_s}→R^{d_a} satisfying L_a^k W = W L_s^k for all k.

    Stacks the Kronecker systems for every generator and finds the right singular
    vector with the smallest singular value — one W that works for all generators
    simultaneously rather than a per-generator solution.

    Returns
    -------
    W_joint    : (d_a, d_s) unit-norm shared intertwiner
    min_sv     : smallest singular value of the stacked K (near 0 → good joint W)
    residuals  : list of per-generator ||L_a^k W - W L_s^k||_F / ||W||_F residuals
    """
    Ls_arr = [np.asarray(L, dtype=np.float64) for L in L_s_list]
    La_arr = [np.asarray(L, dtype=np.float64) for L in L_a_list]
    d_s = Ls_arr[0].shape[0]
    d_a = La_arr[0].shape[0]

    K_blocks = [
        np.kron(np.eye(d_s), La) - np.kron(Ls.T, np.eye(d_a))
        for Ls, La in zip(Ls_arr, La_arr)
    ]
    K_joint = np.vstack(K_blocks)  # (K * d_s*d_a, d_s*d_a)
    _, sv, Vt = np.linalg.svd(K_joint, full_matrices=False)
    w_vec = Vt[-1]
    W = w_vec.reshape(d_a, d_s, order="F").astype(np.float32)
    w_nrm = float(np.linalg.norm(W))
    if w_nrm > 1e-12:
        W = W / w_nrm

    residuals = [
        float(np.linalg.norm(La @ W - W @ Ls)) / (float(np.linalg.norm(W)) + 1e-12)
        for Ls, La in zip(Ls_arr, La_arr)
    ]
    return W, float(sv[-1]), residuals


# =============================================================================
# SCORE MODEL VISUALISATION
# =============================================================================

def visualize_score_model(
    model,
    z_std: np.ndarray,
    state_dim: int,
    action_dim: int,
    sigma_eval: float,
    batch_size: int = 4096,
    n_scatter: int = 500,
) -> dict:
    """Visualise denoiser quality on z_std = standardised [s; a; s'] transitions.

    Returns dict of matplotlib Figure objects:
      'manifold_distances' — per-block histogram of ||d(z)−z||² / dim
      'per_dim_rmse'       — per-dimension RMSE bar chart
      'denoiser_scatter'   — scatter d(z)_i vs z_i for every dimension
      'coverage'           — PCA 2-D projection of z colored by manifold distance

    How to read them
    ----------------
    * 'manifold_distances': low values ≈ z is on the learned manifold.
      Ideally the state s and next-state s' blocks have similar (low) error;
      the action block may be higher if actions are less constrained.
    * 'per_dim_rmse': spiky bars in one block → the denoiser struggles with
      that dimension; often a sign of poor sigma_eval choice or insufficient
      training on that part of z.
    * 'denoiser_scatter': points on the diagonal ≈ perfect reconstruction.
      Vertical scatter indicates the denoiser averages over the manifold.
    * 'coverage': bright regions (high mfd) are either data gaps or manifold
      boundaries where the denoiser extrapolates.  Good coverage = uniform dark.

    Common failure modes visible here
    ----------------------------------
    1. sigma_eval too small → all RMSE values near 0 (denoiser acts as identity;
       Jacobian I − dd/dz ≈ 0; no useful normal directions for continuous sym.).
    2. sigma_eval too large → RMSE high everywhere; denoiser averages too broadly.
    3. Action block RMSE >> state block → action manifold poorly learned; L_a
       estimates from the normal frames will be noisy relative to L_s.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        raise ImportError("matplotlib required for visualize_score_model")

    z_np = _to_numpy(z_std).astype(np.float32)
    n, total_dim = z_np.shape
    a_start  = state_dim
    sp_start = state_dim + action_dim

    # ── Run denoiser in batches ───────────────────────────────────────────────
    denoised = np.concatenate([
        np.asarray(model(jnp.asarray(z_np[s:s + batch_size]), float(sigma_eval)))
        for s in range(0, n, batch_size)
    ], axis=0)                                        # (n, total_dim)
    sq_err       = (denoised - z_np) ** 2            # (n, total_dim)
    per_dim_rmse = np.sqrt(np.mean(sq_err, axis=0))  # (total_dim,)

    block_sq = {
        "state s":      sq_err[:, :a_start],
        "action a":     sq_err[:, a_start:sp_start],
        "next state s'": sq_err[:, sp_start:],
    }
    block_mfd = {k: np.mean(v, axis=-1) for k, v in block_sq.items()}  # (n,) each
    figs = {}

    # ── 1. Per-block manifold distance histograms ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, mfd), color in zip(
        axes, block_mfd.items(), ["steelblue", "tomato", "seagreen"]
    ):
        ax.hist(mfd, bins=60, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("mean sq error / dim", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.set_title(
            f"{name}\nRMSE = {float(np.sqrt(np.mean(mfd))):.4f}", fontsize=9
        )
    fig.suptitle(
        f"Denoiser manifold distances  (σ_eval={sigma_eval:.3e})\n"
        "Low = point is on the learned data manifold", fontsize=10
    )
    fig.tight_layout()
    figs["manifold_distances"] = fig

    # ── 2. Per-dimension RMSE bar chart ───────────────────────────────────────
    dim_colors = (
        ["steelblue"] * a_start
        + ["tomato"]   * (sp_start - a_start)
        + ["seagreen"] * (total_dim - sp_start)
    )
    fig, ax = plt.subplots(figsize=(max(8, total_dim * 0.45), 4))
    ax.bar(range(total_dim), per_dim_rmse, color=dim_colors, alpha=0.85)
    ax.axvline(a_start  - 0.5, color="k", ls="--", lw=0.8, label="s|a split")
    ax.axvline(sp_start - 0.5, color="k", ls=":",  lw=0.8, label="a|s' split")
    ax.set_xlabel("dimension index", fontsize=9)
    ax.set_ylabel("RMSE", fontsize=9)
    ax.set_title("Per-dimension denoiser RMSE", fontsize=10)
    ax.legend(handles=[
        Patch(facecolor="steelblue", label=f"s   [0:{a_start})"),
        Patch(facecolor="tomato",    label=f"a   [{a_start}:{sp_start})"),
        Patch(facecolor="seagreen",  label=f"s'  [{sp_start}:{total_dim})"),
    ], fontsize=8)
    fig.tight_layout()
    figs["per_dim_rmse"] = fig

    # ── 3. Scatter: d(z) vs z per dimension ──────────────────────────────────
    rng  = np.random.default_rng(0)
    idx  = rng.choice(n, size=min(n_scatter, n), replace=False)
    z_s  = z_np[idx]
    d_s_ = denoised[idx]
    ncols = min(8, total_dim)
    nrows = math.ceil(total_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2), squeeze=False)
    ax_iter = iter(axes.flat)
    for dim_i in range(total_dim):
        ax    = next(ax_iter)
        color = ("steelblue" if dim_i < a_start
                 else "tomato" if dim_i < sp_start else "seagreen")
        block = "s" if dim_i < a_start else ("a" if dim_i < sp_start else "s'")
        ax.scatter(z_s[:, dim_i], d_s_[:, dim_i], c=color, s=4, alpha=0.5)
        lo = min(z_s[:, dim_i].min(), d_s_[:, dim_i].min())
        hi = max(z_s[:, dim_i].max(), d_s_[:, dim_i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6)
        ax.set_title(f"dim {dim_i} ({block})\n{per_dim_rmse[dim_i]:.3f}", fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in ax_iter:
        ax.set_visible(False)
    fig.suptitle("Denoiser output d(z) vs input z", fontsize=10)
    fig.tight_layout()
    figs["denoiser_scatter"] = fig

    # ── 4. PCA coverage colored by total manifold distance ───────────────────
    sub_n = min(4096, n)
    z_sub = z_np[:sub_n]
    _, _, Vt = np.linalg.svd(z_sub - z_sub.mean(0), full_matrices=False)
    pca2      = z_np @ Vt[:2].T                   # (n, 2)
    total_mfd = np.mean(sq_err, axis=-1)           # (n,)
    fig, ax   = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pca2[:, 0], pca2[:, 1],
                    c=np.log1p(total_mfd), cmap="plasma", s=5, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="log(1 + mfd)")
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.set_title(
        "Data coverage: colour = log manifold distance\n"
        "(dark = well on manifold, bright = outlier / data gap)", fontsize=9
    )
    fig.tight_layout()
    figs["coverage"] = fig

    return figs