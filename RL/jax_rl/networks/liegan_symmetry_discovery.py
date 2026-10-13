"""LieGAN symmetry discovery in two modes.

  mode="marginal"
      Discriminate  (g_s·s, g_a·a, g_s·s')  vs  (s, a, s').
      Finds symmetries of the joint distribution p(s, a, s').

  mode="conditional"
      Discriminate  (s, a, g_s·s')  vs  (s, a, s').
      Finds symmetries of the conditional p(s' | s, a): the (s,a) context is
      kept fixed so the discriminator can use it to judge whether g_s·s' is a
      plausible next state.  This is a strictly tighter test than the marginal.

Generator parametrises K Lie algebra generators (L_s, L_a) coupled via an
intertwiner W (L_s = W^+ L_a W).  Group elements are sampled via matrix
exponential of random linear combinations of the generators.  Training is a
standard non-saturating GAN (BCE) with isometry, non-triviality, and
intertwiner regularisation.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.networks.score_symmetry_discovery import (
    MLP,
    _build_splits,
    _to_numpy,
    _sinkhorn,
    _round_signed_permutation,
    _key_for_pair,
    _close_discrete_subgroup,
    _linear_schedule,
    _hyperoctahedral_size,
    _merge_discrete_results,
    DiscreteDiscoveryConfig,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LieGANGeneratorConfig:
    K: int = 4                      # number of Lie algebra generators
    hidden_dims: Tuple[int, ...] = (256, 256)
    min_scale: float = 0.2          # minimum sampling std per generator
    init_scale: float = 0.3         # initial sampling std
    eps_pinv: float = 1e-3          # regularisation for W pseudoinverse
    use_intertwiner: bool = True    # couple L_s and L_a via W (L_s = W^+ L_a W)
    use_affine: bool = False        # add per-generator bias (b_s, b_a): g·x = R·x + t
    fixed_scale: bool = False       # sample alpha ~ Uniform(-0.5, 0.5) instead of Normal*scale


@dataclass
class LieGANDiscriminatorConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)


@dataclass
class LieGANTrainingConfig:
    mode: str = "marginal"          # "marginal" | "conditional"
    gen_loss_flavor: str = "mse"    # conditional only: "mse" | "adversarial"
    num_steps: int = 50_000
    batch_size: int = 512
    gen_lr: float = 3e-4
    disc_lr: float = 3e-4
    grad_clip: float = 1.0
    n_disc_steps: int = 1           # discriminator updates per generator update
    validation_split: float = 0.1
    holdout_split: float = 0.1
    seed: int = 0
    log_interval: int = 1000
    # regularisation weights
    lambda_isometry: float = 0.5    # push L + L^T → 0  (skew-symmetric generators)
    lambda_nontrivial: float = 1.0  # push ||g·x - x|| ≥ margin
    nontrivial_margin: float = 0.05
    lambda_diverse: float = 0.5     # push generators apart
    lambda_intertwine: float = 1.0  # push W L_s ≈ L_a W
    lambda_scale_floor: float = 0.5  # keep sampling scale above min_scale
    normalize_data: bool = True
    lambda_scale_sparse: float = 0.0 # L1 penalty on (scale - min_scale): kills generators without enough discriminator support
    lambda_rank: float = 0.1        # rank log-barrier on L_s (and L_a when use_intertwiner=False)
    lambda_rank_action: float = 0.0 # rank log-barrier on L_a; 0 = disabled (L_a=0 is a valid solution)


@dataclass
class LieGANConfig:
    generator: LieGANGeneratorConfig = field(default_factory=LieGANGeneratorConfig)
    discriminator: LieGANDiscriminatorConfig = field(default_factory=LieGANDiscriminatorConfig)
    training: LieGANTrainingConfig = field(default_factory=LieGANTrainingConfig)
    discrete: DiscreteDiscoveryConfig = field(default_factory=DiscreteDiscoveryConfig)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LieSymmetryGenerator(eqx.Module):
    """K Lie algebra generators (L_a, L_s) with learnable scales.

    Parameters stored:
      L_a_raw    (K, a, a)  — raw action generators (Frobenius-normalised on use)
      L_s_raw    (K, s, s)  — raw state generators (used only when use_intertwiner=False)
      W_raw      (a, s)     — intertwiner (used only when use_intertwiner=True)
      log_scales (K,)       — log(softplus⁻¹(scale - min_scale))
      b_s_raw    (K, s)     — bias generators for state (use_affine=True only)
      b_a_raw    (K, a)     — bias generators for action (use_affine=True only)

    use_intertwiner=True  (default):
        L_s = W^+ L_a W  so that  W L_s = L_a W.  L_s_raw is ignored.
    use_intertwiner=False:
        L_s and L_a are learned independently.
    use_affine=True:
        Group action is g·x = R·x + t  where R = exp(Σ α_k L) and t = Σ α_k b.
    """
    L_a_raw: jnp.ndarray
    L_s_raw: jnp.ndarray
    W_raw: jnp.ndarray
    log_scales: jnp.ndarray
    b_s_raw: jnp.ndarray
    b_a_raw: jnp.ndarray
    state_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    eps_pinv: float = eqx.field(static=True)
    min_scale: float = eqx.field(static=True)
    use_intertwiner: bool = eqx.field(static=True)
    use_affine: bool = eqx.field(static=True)
    fixed_scale: bool = eqx.field(static=True)

    def __init__(self, state_dim, action_dim, K, *, key,
                 init_scale=0.3, eps_pinv=1e-3, min_scale=0.2,
                 use_intertwiner=True, use_affine=False, fixed_scale=False):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.K = int(K)
        self.eps_pinv = float(eps_pinv)
        self.min_scale = float(min_scale)
        self.use_intertwiner = bool(use_intertwiner)
        self.use_affine = bool(use_affine)
        self.fixed_scale = bool(fixed_scale)

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.L_a_raw = jax.random.normal(k1, (K, action_dim, action_dim)) * 0.1
        self.L_s_raw = jax.random.normal(k2, (K, state_dim,  state_dim))  * 0.1
        self.W_raw   = jax.random.normal(k3, (action_dim, state_dim))     * 0.1
        init_val = float(np.log(np.expm1(max(init_scale - min_scale, 1e-6))))
        self.log_scales = jnp.full((K,), init_val)
        # bias generators — initialised near zero; only used when use_affine=True
        self.b_s_raw = jax.random.normal(k5, (K, state_dim))  * 0.01
        self.b_a_raw = jax.random.normal(k6, (K, action_dim)) * 0.01

    # ---- helpers -----------------------------------------------------------

    def _normalized(self):
        """Return (L_a, W, W_pinv, L_s, scales)."""
        fro_a = jnp.linalg.norm(self.L_a_raw, axis=(-2, -1), keepdims=True)
        L_a = self.L_a_raw / (fro_a + 1e-8)              # (K, a, a)

        row_norms = jnp.linalg.norm(self.W_raw, axis=-1, keepdims=True)
        W = self.W_raw / jnp.maximum(row_norms, 1e-3)    # (a, s)
        gram = W @ W.T + self.eps_pinv * jnp.eye(self.action_dim)
        W_pinv = W.T @ jnp.linalg.inv(gram)              # (s, a)

        if self.use_intertwiner:
            L_s = jnp.einsum("ij,kjl,lm->kim", W_pinv, L_a, W)  # (K, s, s)
        else:
            fro_s = jnp.linalg.norm(self.L_s_raw, axis=(-2, -1), keepdims=True)
            L_s = self.L_s_raw / (fro_s + 1e-8)                  # (K, s, s)

        scales = jax.nn.softplus(self.log_scales) + self.min_scale   # (K,)
        return L_a, W, W_pinv, L_s, scales

    def sample(self, rng: jax.Array, batch_size: int):
        """Sample (g_s, g_a, t_s, t_a, alpha) for a batch.

        Returns
        -------
        g_s   (B, s, s)  — linear part for state
        g_a   (B, a, a)  — linear part for action
        t_s   (B, s)     — translation for state  (zeros if use_affine=False)
        t_a   (B, a)     — translation for action (zeros if use_affine=False)
        alpha (B, K)     — sampled coefficients

        Group action: g·x = g_s @ x + t_s  (affine), or g_s @ x  (linear).
        """
        L_a, _, _, L_s, scales = self._normalized()
        rng_alpha, rng_sign = jax.random.split(rng)
        if self.fixed_scale:
            alpha = jax.random.uniform(rng_alpha, (batch_size, self.K), minval=-0.5, maxval=0.5)
        else:
            alpha = jax.random.normal(rng_alpha, (batch_size, self.K)) * scales[None]
            signs = jnp.where(jax.random.bernoulli(rng_sign, 0.5, (batch_size, 1)), 1.0, -1.0)
            alpha = alpha * signs

        A_a = jnp.einsum("bk,kij->bij", alpha, L_a)
        A_s = jnp.einsum("bk,kij->bij", alpha, L_s)
        g_a = jax.vmap(jax.scipy.linalg.expm)(A_a)
        g_s = jax.vmap(jax.scipy.linalg.expm)(A_s)

        if self.use_affine:
            t_s = jnp.einsum("bk,ki->bi", alpha, self.b_s_raw)   # (B, s)
            t_a = jnp.einsum("bk,ki->bi", alpha, self.b_a_raw)   # (B, a)
        else:
            t_s = jnp.zeros((batch_size, self.state_dim))
            t_a = jnp.zeros((batch_size, self.action_dim))

        return g_s, g_a, t_s, t_a, alpha


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(eqx.Module):
    net: MLP

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], *, key):
        self.net = MLP(input_dim, 1, hidden_dims, key=key)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.net(z)[..., 0]   # scalar logit per sample


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _bce_disc(real_logits, fake_logits):
    real_loss = -jnp.mean(jax.nn.log_sigmoid(real_logits))
    fake_loss = -jnp.mean(jax.nn.log_sigmoid(-fake_logits))
    return 0.5 * (real_loss + fake_loss)


def _bce_gen(fake_logits):
    return -jnp.mean(jax.nn.log_sigmoid(fake_logits))



def _apply_g(g: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Batched matrix-vector:  g (B,d,d) × x (B,d) → (B,d)."""
    return jnp.einsum("bij,bj->bi", g, x)


def _regularization(gen: LieSymmetryGenerator, states, actions, alpha, rank_action: bool = False):
    L_a, W, _, L_s, scales = gen._normalized()
    cfg_min = gen.min_scale

    # Isometry: push generators toward skew-symmetric (L + L^T ≈ 0)
    skew_a = jnp.mean((L_a + jnp.swapaxes(L_a, -1, -2)) ** 2)
    skew_s = jnp.mean((L_s + jnp.swapaxes(L_s, -1, -2)) ** 2)
    r_iso = 0.5 * (skew_a + skew_s)

    # Intertwiner: W L_s - L_a W ≈ 0  (only meaningful when use_intertwiner=False,
    # since the coupling is exact by construction when use_intertwiner=True)
    if not gen.use_intertwiner:
        WLs = jnp.einsum("ij,kjl->kil", W, L_s)
        LaW = jnp.einsum("kij,jl->kil", L_a, W)
        r_intertwine = jnp.mean((WLs - LaW) ** 2)
    else:
        r_intertwine = jnp.array(0.0)

    # Diversity: push generators apart (Gram off-diagonal)
    flat = L_a.reshape(gen.K, -1)
    gram = flat @ flat.T
    off = gram - jnp.diag(jnp.diag(gram))
    r_diverse = jnp.mean(off ** 2)

    # Non-triviality: group action moves *states* by at least margin.
    # L_a = 0 is a valid solution (e.g. torques are invariant scalars), so we
    # gate only on state displacement — not the average of both.
    A_s = jnp.einsum("k,kij->ij", scales, L_s)
    g_s_test = jax.scipy.linalg.expm(A_s)
    disp_s = jnp.mean(jnp.sum((states @ g_s_test.T - states) ** 2, axis=-1))
    r_nontrivial = disp_s

    # Scale floor: keep sampling std above min_scale
    r_scale = jnp.mean(jnp.maximum(cfg_min - scales, 0.0) ** 2)

    # Scale sparsity: L1 pressure on excess scale above the floor.
    # Generators without sufficient discriminator support will collapse to
    # min_scale; only those that genuinely help fool D will remain active.
    r_sparse = jnp.mean(scales - cfg_min)

    # Rank regularization: log-barrier on row and column norms of L_s (and
    # optionally L_a).  Prevents complete dimension dropout without requiring
    # full rank.  L_a = 0 is a valid solution (scalar-action symmetry), so
    # the action rank barrier is disabled by default (rank_action=False).
    if not gen.use_intertwiner:
        row_norms_s = jnp.linalg.norm(L_s, axis=-1)
        col_norms_s = jnp.linalg.norm(L_s, axis=-2)
        r_rank_s = 0.5 * (jnp.mean(-jnp.log(row_norms_s + 1e-6))
                          + jnp.mean(-jnp.log(col_norms_s + 1e-6)))
    else:
        r_rank_s = jnp.array(0.0)
    if rank_action:
        row_norms_a = jnp.linalg.norm(L_a, axis=-1)
        col_norms_a = jnp.linalg.norm(L_a, axis=-2)
        r_rank_a = 0.5 * (jnp.mean(-jnp.log(row_norms_a + 1e-6))
                          + jnp.mean(-jnp.log(col_norms_a + 1e-6)))
    else:
        r_rank_a = jnp.array(0.0)
    r_rank = 0.5 * (r_rank_s + r_rank_a)

    return r_iso, r_intertwine, r_diverse, r_nontrivial, r_scale, r_sparse, r_rank


# ---------------------------------------------------------------------------
# Single train step (one discriminator + one generator update)
# ---------------------------------------------------------------------------

def _make_fake(z_batch, g_s, g_a, state_dim, action_dim, t_s=None, t_a=None):
    """Apply affine transform (g_s, g_a, t_s, t_a) to (s, a, s').

    g·s = g_s @ s + t_s,  g·a = g_a @ a + t_a,  g·s' = g_s @ s' + t_s.
    Same translation t_s applied to both s and s' (translation is a global shift).
    t_s / t_a default to zero (linear-only mode).
    """
    s  = z_batch[:, :state_dim]
    a  = z_batch[:, state_dim: state_dim + action_dim]
    sp = z_batch[:, state_dim + action_dim:]
    s_t  = _apply_g(g_s, s)
    a_t  = _apply_g(g_a, a)
    sp_t = _apply_g(g_s, sp)
    if t_s is not None:
        s_t  = s_t  + t_s
        sp_t = sp_t + t_s
    if t_a is not None:
        a_t  = a_t  + t_a
    return jnp.concatenate([s_t, a_t, sp_t], axis=-1)


def _build_step_marginal(state_dim: int, action_dim: int,
                         lambda_iso, lambda_intertwine, lambda_diverse,
                         lambda_nontrivial, nontrivial_margin, lambda_scale, lambda_sparse, lambda_rank,
                         rank_action: bool = False):
    """Marginal mode: D distinguishes (g·s, g·a, g·s') vs (s, a, s')."""

    @eqx.filter_jit
    def step(gen, disc, gen_opt_state, disc_opt_state,
             gen_optimizer, disc_optimizer,
             z_batch, rng):

        s = z_batch[:, :state_dim]
        a = z_batch[:, state_dim: state_dim + action_dim]
        rng, rng_g1, rng_g2 = jax.random.split(rng, 3)

        g_s_d, g_a_d, t_s_d, t_a_d, _ = jax.lax.stop_gradient(gen.sample(rng_g1, z_batch.shape[0]))
        z_fake_d = _make_fake(z_batch, g_s_d, g_a_d, state_dim, action_dim, t_s_d, t_a_d)

        def disc_loss_fn(d):
            rl = d(z_batch)
            fl = d(z_fake_d)
            return _bce_disc(rl, fl), (jnp.mean(jax.nn.sigmoid(rl)),
                                       jnp.mean(jax.nn.sigmoid(fl)))

        (d_loss, (real_val, fake_val)), d_grads = eqx.filter_value_and_grad(
            disc_loss_fn, has_aux=True)(disc)
        d_updates, disc_opt_state_new = disc_optimizer.update(
            d_grads, disc_opt_state, eqx.filter(disc, eqx.is_array))
        disc_new = eqx.apply_updates(disc, d_updates)

        def gen_loss_fn(g):
            g_s, g_a, t_s, t_a, alpha = g.sample(rng_g2, z_batch.shape[0])
            z_fake = _make_fake(z_batch, g_s, g_a, state_dim, action_dim, t_s, t_a)
            adv = _bce_gen(disc_new(z_fake))
            r_iso, r_int, r_div, r_ntriv, r_scale, r_sparse, r_rank = _regularization(g, s, a, alpha, rank_action=rank_action)
            r_ntriv_hinge = jnp.maximum(nontrivial_margin - r_ntriv, 0.0) ** 2
            total = (adv
                     + lambda_iso        * r_iso
                     + lambda_intertwine * r_int
                     + lambda_diverse    * r_div
                     + lambda_nontrivial * r_ntriv_hinge
                     + lambda_scale      * r_scale
                     + lambda_sparse     * r_sparse
                     + lambda_rank       * r_rank)
            return total, (adv, r_iso, r_int, r_div, r_ntriv, r_scale, r_sparse, r_rank)

        (g_loss, (adv, r_iso, r_int, r_div, r_ntriv, r_scale, r_sparse, r_rank)), g_grads = (
            eqx.filter_value_and_grad(gen_loss_fn, has_aux=True)(gen))
        g_updates, gen_opt_state_new = gen_optimizer.update(
            g_grads, gen_opt_state, eqx.filter(gen, eqx.is_array))
        gen_new = eqx.apply_updates(gen, g_updates)

        metrics = {
            "loss/disc": d_loss,
            "loss/gen_total": g_loss,
            "loss/gen_adv": adv,
            "reg/isometry": r_iso,
            "reg/intertwine": r_int,
            "reg/diverse": r_div,
            "reg/nontrivial_disp": r_ntriv,
            "reg/scale_floor": r_scale,
            "reg/scale_sparse": r_sparse,
            "reg/rank": r_rank,
            "disc/real_validity": real_val,
            "disc/fake_validity": fake_val,
            "disc/separation": real_val - fake_val,
        }
        return gen_new, disc_new, gen_opt_state_new, disc_opt_state_new, rng, metrics

    return step


def _build_step_conditional(state_dim: int, action_dim: int,
                             gen_loss_flavor: str,
                             lambda_iso, lambda_intertwine, lambda_diverse,
                             lambda_nontrivial, nontrivial_margin, lambda_scale, lambda_sparse, lambda_rank,
                             rank_action: bool = False):
    """Conditional mode: D distinguishes matched (s,a,s') vs shuffled-s' negatives.

    D-step: train on (s_i, a_i, s'_i) real vs (s_i, a_i, s'_{π(i)}) mismatched.
    Both have identical (s,a) marginal so D must track p(s'|s,a).
    At optimum: D*(s,a,s') = log p(s'|s,a) - log p(s') + c(s,a).

    G-step (gap preservation): Δ_i = D(s,a,s') - D(s,a,s'_shuf).
    If g is a conditional symmetry, Δ̃_i = D(g·s,g·a,g·s') - D(g·s,g·a,g·s'_shuf) ≈ Δ_i.
    c(s,a) and log p(s') cancel in the differences so c is unidentified.

    gen_loss_flavor:
      "mse"        — L_G = E[||Δ_i - Δ̃_i||²]   (strong pointwise test)
      "adversarial" — L_G = -E[log σ(Δ̃_i)]       (make transformed gap large)
    """

    @eqx.filter_jit
    def step(gen, disc, gen_opt_state, disc_opt_state,
             gen_optimizer, disc_optimizer,
             z_batch, rng):

        s  = z_batch[:, :state_dim]
        a  = z_batch[:, state_dim: state_dim + action_dim]
        sp = z_batch[:, state_dim + action_dim:]

        rng, rng_perm, rng_g2 = jax.random.split(rng, 3)

        perm    = jax.random.permutation(rng_perm, z_batch.shape[0])
        sp_shuf = sp[perm]
        z_real  = z_batch
        z_mism  = jnp.concatenate([s, a, sp_shuf], axis=-1)

        # --- D: matched vs same-context-shuffled ---
        def disc_loss_fn(d):
            d_real = d(z_real)
            d_mism = d(z_mism)
            return -jnp.mean(jax.nn.log_sigmoid(d_real - d_mism)), (
                jnp.mean(jax.nn.sigmoid(d_real - d_mism)),   # accuracy
            )

        (d_loss, (acc,)), d_grads = eqx.filter_value_and_grad(
            disc_loss_fn, has_aux=True)(disc)
        d_updates, disc_opt_state_new = disc_optimizer.update(
            d_grads, disc_opt_state, eqx.filter(disc, eqx.is_array))
        disc_new = eqx.apply_updates(disc, d_updates)

        # --- G: gap preserved under g ---
        def gen_loss_fn(g):
            g_s, g_a, t_s, t_a, alpha = g.sample(rng_g2, z_batch.shape[0])
            z_real_t = _make_fake(z_real, g_s, g_a, state_dim, action_dim, t_s, t_a)
            z_mism_t = _make_fake(z_mism, g_s, g_a, state_dim, action_dim, t_s, t_a)

            # stop-gradient on D values from original space
            d_real   = jax.lax.stop_gradient(disc_new(z_real))
            d_mism   = jax.lax.stop_gradient(disc_new(z_mism))
            d_real_t = disc_new(z_real_t)
            d_mism_t = disc_new(z_mism_t)

            delta   = d_real   - d_mism      # original gap
            delta_t = d_real_t - d_mism_t    # transformed gap

            if gen_loss_flavor == "mse":
                adv = jnp.mean((delta - delta_t) ** 2)
            else:
                adv = -jnp.mean(jax.nn.log_sigmoid(delta_t))

            r_iso, r_int, r_div, r_ntriv, r_scale, r_sparse, r_rank = _regularization(g, s, a, alpha, rank_action=rank_action)
            r_ntriv_hinge = jnp.maximum(nontrivial_margin - r_ntriv, 0.0) ** 2
            total = (adv
                     + lambda_iso        * r_iso
                     + lambda_intertwine * r_int
                     + lambda_diverse    * r_div
                     + lambda_nontrivial * r_ntriv_hinge
                     + lambda_scale      * r_scale
                     + lambda_sparse     * r_sparse
                     + lambda_rank       * r_rank)
            return total, (adv, r_iso, r_int, r_div, r_ntriv, r_scale, r_sparse, r_rank)

        (g_loss, (adv, r_iso, r_int, r_div, r_ntriv, r_scale, r_sparse, r_rank)), g_grads = (
            eqx.filter_value_and_grad(gen_loss_fn, has_aux=True)(gen))
        g_updates, gen_opt_state_new = gen_optimizer.update(
            g_grads, gen_opt_state, eqx.filter(gen, eqx.is_array))
        gen_new = eqx.apply_updates(gen, g_updates)

        metrics = {
            "loss/disc": d_loss,
            "loss/gen_total": g_loss,
            "loss/gen_adv": adv,
            "reg/isometry": r_iso,
            "reg/intertwine": r_int,
            "reg/diverse": r_div,
            "reg/nontrivial_disp": r_ntriv,
            "reg/scale_floor": r_scale,
            "reg/scale_sparse": r_sparse,
            "reg/rank": r_rank,
            "disc/accuracy": acc,
            "disc/separation": acc - 0.5,
        }
        return gen_new, disc_new, gen_opt_state_new, disc_opt_state_new, rng, metrics

    return step


def _build_step(mode: str, gen_loss_flavor: str, state_dim: int, action_dim: int,
                lambda_iso, lambda_intertwine, lambda_diverse,
                lambda_nontrivial, nontrivial_margin, lambda_scale, lambda_sparse, lambda_rank,
                rank_action: bool = False):
    if mode == "conditional":
        return _build_step_conditional(
            state_dim, action_dim, gen_loss_flavor,
            lambda_iso, lambda_intertwine, lambda_diverse,
            lambda_nontrivial, nontrivial_margin, lambda_scale, lambda_sparse, lambda_rank,
            rank_action=rank_action)
    return _build_step_marginal(
        state_dim, action_dim,
        lambda_iso, lambda_intertwine, lambda_diverse,
        lambda_nontrivial, nontrivial_margin, lambda_scale, lambda_sparse, lambda_rank,
        rank_action=rank_action)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LieGANTrainer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: LieGANConfig,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = config
        z_dim = 2 * state_dim + action_dim

        key = jax.random.PRNGKey(int(config.training.seed))
        k_gen, k_disc = jax.random.split(key)

        self.gen = LieSymmetryGenerator(
            state_dim=state_dim,
            action_dim=action_dim,
            K=int(config.generator.K),
            key=k_gen,
            init_scale=float(config.generator.init_scale),
            eps_pinv=float(config.generator.eps_pinv),
            min_scale=float(config.generator.min_scale),
            use_intertwiner=bool(config.generator.use_intertwiner),
            use_affine=bool(config.generator.use_affine),
            fixed_scale=bool(config.generator.fixed_scale),
        )
        self.disc = Discriminator(
            input_dim=z_dim,
            hidden_dims=tuple(config.discriminator.hidden_dims),
            key=k_disc,
        )

        t = config.training
        gen_tx = optax.chain(
            optax.clip_by_global_norm(float(t.grad_clip)),
            optax.adam(float(t.gen_lr)),
        )
        disc_tx = optax.chain(
            optax.clip_by_global_norm(float(t.grad_clip)),
            optax.adam(float(t.disc_lr)),
        )
        self.gen_opt_state  = gen_tx.init(eqx.filter(self.gen,  eqx.is_array))
        self.disc_opt_state = disc_tx.init(eqx.filter(self.disc, eqx.is_array))
        self.gen_optimizer  = gen_tx
        self.disc_optimizer = disc_tx

        self._step = _build_step(
            mode=str(t.mode),
            gen_loss_flavor=str(t.gen_loss_flavor),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            lambda_iso=float(t.lambda_isometry),
            lambda_intertwine=float(t.lambda_intertwine),
            lambda_diverse=float(t.lambda_diverse),
            lambda_nontrivial=float(t.lambda_nontrivial),
            nontrivial_margin=float(t.nontrivial_margin),
            lambda_scale=float(t.lambda_scale_floor),
            lambda_sparse=float(getattr(t, "lambda_scale_sparse", 0.0)),
            lambda_rank=float(t.lambda_rank),
            rank_action=float(getattr(t, "lambda_rank_action", 0.0)) > 0,
        )

    def fit(self, z_std: np.ndarray, verbose: bool = False) -> dict:
        cfg = self.cfg.training
        rng = jax.random.PRNGKey(int(cfg.seed) + 1)
        n = int(z_std.shape[0])
        history: dict = {k: [] for k in [
            "loss/disc", "loss/gen_total", "loss/gen_adv",
            "reg/isometry", "reg/intertwine", "reg/diverse",
            "reg/nontrivial_disp", "reg/rank",
            "disc/separation", "disc/accuracy",
        ]}

        for step in range(int(cfg.num_steps)):
            rng, rng_idx = jax.random.split(rng)
            idx = jax.random.randint(
                rng_idx, (min(int(cfg.batch_size), n),), 0, n
            )
            z_batch = jnp.asarray(z_std[np.asarray(idx)], dtype=jnp.float32)

            self.gen, self.disc, self.gen_opt_state, self.disc_opt_state, rng, metrics = (
                self._step(
                    self.gen, self.disc,
                    self.gen_opt_state, self.disc_opt_state,
                    self.gen_optimizer, self.disc_optimizer,
                    z_batch, rng,
                )
            )

            for k in history:
                if k in metrics:
                    history[k].append(float(metrics[k]))

            if verbose and (
                step == 0
                or step == int(cfg.num_steps) - 1
                or (step + 1) % max(1, int(cfg.log_interval)) == 0
            ):
                print(
                    f"[liegan/{cfg.mode}] step {step + 1}/{cfg.num_steps} "
                    f"d={float(metrics['loss/disc']):.3e} "
                    f"g={float(metrics['loss/gen_total']):.3e} "
                    f"sep={float(metrics['disc/separation']):.3f}"
                )

        return history

    def get_generators(self):
        """Return discovered (L_s, L_a, W, scales[, b_s, b_a]) as numpy arrays."""
        L_a, W, _, L_s, scales = self.gen._normalized()
        out = {
            "L_s": np.asarray(L_s),       # (K, s, s)
            "L_a": np.asarray(L_a),       # (K, a, a)
            "W":   np.asarray(W),         # (a, s)
            "scales": np.asarray(scales), # (K,)
        }
        if self.gen.use_affine:
            out["b_s"] = np.asarray(self.gen.b_s_raw)  # (K, s)
            out["b_a"] = np.asarray(self.gen.b_a_raw)  # (K, a)
        return out


# ---------------------------------------------------------------------------
# Discrete signed-permutation search using discriminator as oracle
# ---------------------------------------------------------------------------

class _DiscreteParams(eqx.Module):
    state_logits:    jax.Array
    action_logits:   jax.Array
    state_sign_pre:  jax.Array
    action_sign_pre: jax.Array
    # Geometric intertwiner W: (d_a, d_s). Zero weight (mu_intertwiner=0) when disabled.
    intertwiner_W:   jax.Array


def _disc_holdout_distance(disc: Discriminator, z: np.ndarray, batch_size: int = 1024) -> float:
    """1 - mean P(real) over z.  0 = looks real (good symmetry), 1 = looks fake."""
    probs = []
    for start in range(0, z.shape[0], batch_size):
        batch = jnp.asarray(z[start:start + batch_size], dtype=jnp.float32)
        probs.append(np.asarray(jax.nn.sigmoid(disc(batch))))
    return float(1.0 - np.mean(np.concatenate(probs))) if probs else 0.5


def _disc_holdout_distance_paired(
    disc: Discriminator,
    z_real: np.ndarray,
    z_fake: np.ndarray,
    batch_size: int = 1024,
) -> float:
    """1 - mean sigmoid(D_fake_i - D_real_i) over matched pairs.

    0 = fake looks as real as real (good symmetry), 1 = discriminator fully separates.
    Identity transform gives exactly 0.5 since D_fake = D_real per pair.
    """
    scores = []
    for start in range(0, z_real.shape[0], batch_size):
        r = jnp.asarray(z_real[start:start + batch_size], dtype=jnp.float32)
        f = jnp.asarray(z_fake[start:start + batch_size], dtype=jnp.float32)
        scores.append(np.asarray(jax.nn.sigmoid(disc(f) - disc(r))))
    return float(1.0 - np.mean(np.concatenate(scores))) if scores else 0.5


def _disc_holdout_gap_mse(
    disc: Discriminator,
    z_real: np.ndarray,
    z_fake: np.ndarray,
    state_dim: int,
    action_dim: int,
    batch_size: int = 1024,
    seed: int = 0,
) -> float:
    """Gap-preservation MSE for a conditional discriminator.

    Measures E[||Δ - Δ̃||²] where
      Δ   = D(s,a,s') - D(s,a,s'_shuf)         (original gap)
      Δ̃   = D(g·s,g·a,g·s') - D(g·s,g·a,g·s'_shuf) (transformed gap)
    and s'_shuf / g·s'_shuf are within-batch shuffled.

    0 = gap perfectly preserved (good symmetry). Identity scores 0.
    """
    rng = np.random.default_rng(seed)
    mses = []
    for start in range(0, z_real.shape[0], batch_size):
        r = jnp.asarray(z_real[start:start + batch_size], dtype=jnp.float32)
        f = jnp.asarray(z_fake[start:start + batch_size], dtype=jnp.float32)
        B = r.shape[0]
        perm = rng.permutation(B)

        sa_dim = state_dim + action_dim
        # g is linear so g·s'[perm] = (g·s')[perm]
        r_mism = jnp.concatenate([r[:, :sa_dim], r[:, sa_dim:][perm]], axis=-1)
        f_mism = jnp.concatenate([f[:, :sa_dim], f[:, sa_dim:][perm]], axis=-1)

        delta   = disc(r) - disc(r_mism)
        delta_t = disc(f) - disc(f_mism)
        mses.append(float(jnp.mean((delta - delta_t) ** 2)))
    return float(np.mean(mses)) if mses else 0.0


@functools.lru_cache(maxsize=None)
def _build_discrete_step(mode: str, state_dim: int, action_dim: int):
    """Return a jit-compiled discrete Sinkhorn step using a discriminator oracle.

    lru_cache ensures the same Python function object (and its XLA compilation)
    is reused across candidate batches — prevents N compilations accumulating in
    memory when calling LieGANDiscreteSearch in a loop.
    """

    @eqx.filter_jit
    def step(params, opt_state, optimizer, disc,
             z_batch, temperature, sinkhorn_iters,
             mu_identity_repulsion, kappa_diversity, action_id_weight,
             rng_perm, lambda_sparse, mu_intertwiner):

        def loss_fn(p):
            Qs = _sinkhorn(p.state_logits,  temperature, sinkhorn_iters)  # (K, ds, ds)
            Qa = _sinkhorn(p.action_logits, temperature, sinkhorn_iters)  # (K, da, da)
            Gs = jnp.tanh(p.state_sign_pre)[:, :, None]  * Qs
            Ga = jnp.tanh(p.action_sign_pre)[:, :, None] * Qa

            s  = z_batch[:, :state_dim]
            a  = z_batch[:, state_dim:state_dim + action_dim]
            sp = z_batch[:, state_dim + action_dim:]
            K_c = Gs.shape[0]

            if mode == "conditional":
                # Conditional mode: gap preservation loss, must keep s_t/sp_t for all K.
                # Memory is O(K * B * z_dim) — acceptable for conditional mode usage.
                s_t  = jnp.einsum("bd,ked->kbe", s, Gs)
                a_t  = jnp.einsum("bd,ked->kbe", a, Ga)
                sp_t = jnp.einsum("bd,ked->kbe", sp, Gs)
                gz   = jnp.concatenate([s_t, a_t, sp_t], axis=-1)
                B, z_dim_val = s.shape[0], gz.shape[-1]
                logits_fake = disc(gz.reshape(K_c * B, z_dim_val)).reshape(K_c, B)

                perm = jax.random.permutation(rng_perm, B)
                sp_shuf    = sp[perm]
                sp_shuf_t  = sp_t[:, perm, :]
                z_mism     = jnp.concatenate([s, a, sp_shuf], axis=-1)
                gz_mism    = jnp.concatenate([s_t, a_t, sp_shuf_t], axis=-1)
                logits_real   = disc(z_batch)
                logits_mism   = disc(z_mism)
                logits_mism_k = disc(gz_mism.reshape(K_c * B, z_dim_val)).reshape(K_c, B)
                delta   = logits_real - logits_mism
                delta_t = logits_fake - logits_mism_k
                manifold = jnp.mean((delta[None] - delta_t) ** 2, axis=-1)  # (K,)
            else:
                # Marginal mode: vmap over candidates (avoids jax.lax.scan + closed-over
                # pytree inside filter_value_and_grad inside filter_jit, which triggers
                # an UnboundLocalError in JAX 0.6.x backward pass).
                def _manifold_k(Gs_k, Ga_k):
                    gz_k = jnp.concatenate(
                        [s @ Gs_k.T, a @ Ga_k.T, sp @ Gs_k.T], axis=-1
                    )
                    logits_k = disc(gz_k)
                    return -jnp.mean(jax.nn.log_sigmoid(logits_k))

                manifold = jax.vmap(_manifold_k)(Gs, Ga)  # (K,)

            eye_s = jnp.eye(state_dim,  dtype=Gs.dtype)[None]
            eye_a = jnp.eye(action_dim, dtype=Ga.dtype)[None]
            d_id  = (jnp.sum((Gs - eye_s) ** 2, axis=(1, 2))
                     + action_id_weight * jnp.sum((Ga - eye_a) ** 2, axis=(1, 2)))
            identity_barrier = jnp.sum(jnp.exp(-d_id / 0.25))

            # ||a-b||² = ||a||² + ||b||² - 2<a,b>  avoids materialising (K,K,d²) tensor.
            flat    = jnp.concatenate([Gs.reshape(K_c, -1), Ga.reshape(K_c, -1)], axis=-1)
            norm_sq = jnp.sum(flat ** 2, axis=-1)          # (K,)
            gram    = flat @ flat.T                          # (K, K)
            dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2.0 * gram + 1e-8
            mask    = jnp.triu(jnp.ones_like(dist_sq), k=1)
            diversity = jnp.sum(mask / dist_sq)

            # Geometric intertwiner: Ga W - W Gs ≈ 0, normalised by ||W||².
            W = params.intertwiner_W                         # (d_a, d_s)
            Ga_W = jnp.einsum("kij,jl->kil", Ga, W)         # (K, d_a, d_s)
            W_Gs = jnp.einsum("ij,kjl->kil", W, Gs)         # (K, d_a, d_s)
            w_nrm_sq = jnp.sum(W ** 2) + 1e-8
            intertwiner_loss = (
                jnp.sum(jnp.mean((Ga_W - W_Gs) ** 2, axis=(1, 2))) / w_nrm_sq
            )

            total = (jnp.sum(manifold)
                     + mu_identity_repulsion * identity_barrier
                     + kappa_diversity * diversity
                     + lambda_sparse * jnp.sum(jnp.exp(-manifold))
                     + mu_intertwiner * intertwiner_loss)
            return total, (jnp.mean(manifold), identity_barrier, diversity)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(params, eqx.is_array)
        )
        params_new = eqx.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    return step


class LieGANDiscreteSearch:
    """Search for discrete signed-permutation symmetries using the trained discriminator."""

    def __init__(
        self,
        disc: Discriminator,
        z_train: np.ndarray,
        z_holdout: np.ndarray,
        state_dim: int,
        action_dim: int,
        mode: str,
        config: DiscreteDiscoveryConfig,
        lambda_sparse: float = 0.0,
    ):
        self.disc          = disc
        self.z_train       = jnp.asarray(z_train,   dtype=jnp.float32)
        self.z_holdout     = z_holdout.astype(np.float32)
        self.state_dim     = int(state_dim)
        self.action_dim    = int(action_dim)
        self.mode          = str(mode)
        self.config        = config
        self.lambda_sparse = float(lambda_sparse)
        self._step_fn      = _build_discrete_step(mode, int(state_dim), int(action_dim))

        # Cap candidates at hyperoctahedral group size
        group_size = _hyperoctahedral_size(int(action_dim))
        slack = max(4, group_size // 4)
        cap = group_size + slack
        if int(config.num_candidates) > cap:
            import copy
            config = copy.copy(config)
            config.num_candidates = cap
        self.config = config

    def _init_params(self) -> _DiscreteParams:
        cfg = self.config
        rng = np.random.default_rng(int(cfg.seed))
        sl, al, ssp, asp = [], [], [], []
        for _ in range(int(cfg.num_candidates)):
            sl.append(2.0 * np.eye(self.state_dim, dtype=np.float32)[rng.permutation(self.state_dim)]
                      + 0.05 * rng.standard_normal((self.state_dim,  self.state_dim)).astype(np.float32))
            al.append(2.0 * np.eye(self.action_dim, dtype=np.float32)[rng.permutation(self.action_dim)]
                      + 0.05 * rng.standard_normal((self.action_dim, self.action_dim)).astype(np.float32))
            ssp.append(rng.choice([-2.0, 2.0], size=(self.state_dim,)).astype(np.float32))
            asp.append(rng.choice([-2.0, 2.0], size=(self.action_dim,)).astype(np.float32))
        W_raw = rng.standard_normal((self.action_dim, self.state_dim)).astype(np.float32)
        W_init = W_raw / (float(np.linalg.norm(W_raw)) + 1e-12)
        return _DiscreteParams(
            state_logits=jnp.asarray(np.stack(sl)),
            action_logits=jnp.asarray(np.stack(al)),
            state_sign_pre=jnp.asarray(np.stack(ssp)),
            action_sign_pre=jnp.asarray(np.stack(asp)),
            intertwiner_W=jnp.asarray(W_init),
        )

    def fit(self, verbose: bool = False) -> dict:
        cfg    = self.config
        params = self._init_params()
        opt    = optax.adam(float(cfg.lr))
        opt_state = opt.init(eqx.filter(params, eqx.is_array))
        history = {"loss_total": [], "loss_manifold": [], "loss_identity_push": [],
                   "diversity": [], "temperature": []}
        key = jax.random.PRNGKey(int(cfg.seed))

        # Adaptive tau: set based on identity-transform baseline.
        # Conditional (gap MSE): identity scores ~0 by construction.
        if self.mode == "conditional":
            baseline = 0.0
        else:
            baseline = _disc_holdout_distance(self.disc, self.z_holdout)
        # For LieGAN the discriminator distance is in [0,1]:
        #   dist(I) ≈ 0.5 (identity looks as real as unmodified data)
        #   dist(true G) ≈ 0.5 (true symmetry also looks real)
        #   dist(spurious G) > 0.5 (non-symmetry looks more fake)
        # effective_tau = rho * baseline separates these; rho ≈ 1.1 gives tau ≈ 0.55.
        # Do NOT apply the score-model 5x floor here — baseline=0.5 would make tau=2.5,
        # which combined with rho*baseline=1.0 accepts everything.
        tau = float(cfg.manifold_distance_tau)
        _rho = float(getattr(cfg, "rho_acceptance_threshold", 2.0))
        effective_tau = (
            min(float(tau), _rho * float(baseline))
            if baseline > 0.0
            else float(tau)
        )

        # Hoist constant Python scalars out of the loop so filter_jit
        # sees the same static args every call (avoids per-step retracing).
        _sinkhorn_iters = int(cfg.sinkhorn_iters)
        _mu_id   = float(cfg.mu_identity_repulsion)
        _kappa   = float(cfg.kappa_diversity)
        _act_idw = float(cfg.identity_repulsion_action_weight)
        _lam_s   = float(self.lambda_sparse)
        _mu_int  = float(cfg.mu_intertwiner) if getattr(cfg, 'use_intertwiner', False) else 0.0

        _raw_loss, _raw_manifold, _raw_id_push, _raw_diversity = [], [], [], []

        for step_idx in range(int(cfg.num_steps)):
            temperature = _linear_schedule(
                cfg.temperature_start, cfg.temperature_end,
                step_idx, cfg.num_steps, cfg.anneal_power,
            )
            key, subkey, rng_perm = jax.random.split(key, 3)
            batch_idx = jax.random.randint(
                subkey,
                (min(int(cfg.batch_size), int(self.z_train.shape[0])),),
                minval=0, maxval=int(self.z_train.shape[0]),
            )
            z_batch = jnp.take(self.z_train, batch_idx, axis=0)
            params, opt_state, loss, aux = self._step_fn(
                params, opt_state, opt, self.disc, z_batch,
                # Pass as JAX array so filter_jit traces it dynamically —
                # prevents a retrace per step when temperature changes.
                jnp.asarray(temperature, dtype=jnp.float32),
                _sinkhorn_iters, _mu_id, _kappa, _act_idw, rng_perm,
                _lam_s, _mu_int,
            )
            manifold, id_push, diversity = aux
            # Collect raw JAX arrays — no float() here to avoid per-step
            # GPU→CPU syncs that would stall async dispatch.
            _raw_loss.append(loss)
            _raw_manifold.append(manifold)
            _raw_id_push.append(id_push)
            _raw_diversity.append(diversity)
            history["temperature"].append(float(temperature))
            if verbose and (
                step_idx == 0
                or step_idx == int(cfg.num_steps) - 1
                or (step_idx + 1) % max(1, int(cfg.log_interval)) == 0
            ):
                print(
                    f"[discrete/liegan] step {step_idx + 1}/{cfg.num_steps} "
                    f"loss={float(loss):.4e} mfd={float(manifold):.4e} "
                    f"id_push={float(id_push):.4e}"
                )

        # Drain device buffers once after the loop instead of per-step.
        history["loss_total"]        = [float(v) for v in _raw_loss]
        history["loss_manifold"]     = [float(v) for v in _raw_manifold]
        history["loss_identity_push"]= [float(v) for v in _raw_id_push]
        history["diversity"]         = [float(v) for v in _raw_diversity]

        # Round to hard signed permutations and evaluate on holdout
        final_temp = float(_linear_schedule(
            cfg.temperature_start, cfg.temperature_end,
            int(cfg.num_steps) - 1, cfg.num_steps, cfg.anneal_power,
        ))
        Qs  = np.asarray(_sinkhorn(params.state_logits,  final_temp, int(cfg.sinkhorn_iters)))
        Qa  = np.asarray(_sinkhorn(params.action_logits, final_temp, int(cfg.sinkhorn_iters)))
        ssp = np.asarray(params.state_sign_pre)
        asp = np.asarray(params.action_sign_pre)

        eye_s = np.eye(self.state_dim,  dtype=np.float32)
        eye_a = np.eye(self.action_dim, dtype=np.float32)
        rounded = []
        for idx in range(int(cfg.num_candidates)):
            hard_s = _round_signed_permutation(Qs[idx], ssp[idx])
            hard_a = _round_signed_permutation(Qa[idx], asp[idx])
            zh = self.z_holdout
            s_h  = zh[:, :self.state_dim] @ hard_s.T
            a_h  = zh[:, self.state_dim:self.state_dim + self.action_dim] @ hard_a.T
            sp_h = zh[:, self.state_dim + self.action_dim:] @ hard_s.T
            z_trans = np.concatenate([s_h, a_h, sp_h], axis=-1).astype(np.float32)
            if self.mode == "conditional":
                dist = _disc_holdout_gap_mse(
                    self.disc, zh, z_trans, self.state_dim, self.action_dim)
            else:
                dist = _disc_holdout_distance(self.disc, z_trans)
            item = {
                "index": int(idx),
                "hard_P_s": hard_s,
                "hard_P_a": hard_a,
                "holdout_manifold_distance": float(dist),
            }
            if cfg.exclude_identity and np.allclose(hard_s, eye_s) and np.allclose(hard_a, eye_a):
                item["rejection_reason"] = "identity"
            elif dist >= effective_tau:
                item["rejection_reason"] = "manifold_distance_above_tau"
            else:
                item["rejection_reason"] = ""
            rounded.append(item)

        accepted, rejections, seen = [], [], set()
        for item in sorted(rounded, key=lambda x: x["holdout_manifold_distance"]):
            kp = _key_for_pair(item["hard_P_s"], item["hard_P_a"])
            if item["rejection_reason"]:
                rejections.append(item)
                continue
            if kp in seen:
                rejections.append({**item, "rejection_reason": "duplicate"})
                continue
            seen.add(kp)
            accepted.append(item)

        subgroup = _close_discrete_subgroup(
            accepted, self.state_dim, self.action_dim,
            max_elements=int(cfg.max_subgroup_elements),
        )
        best = accepted[0] if accepted else min(rounded, key=lambda x: x["holdout_manifold_distance"])
        best_mfd = float(best["holdout_manifold_distance"])
        # ρ_mfd = D_mfd(g_rounded) / D_mfd(I); undefined (→ NaN) when D_mfd(I) = 0 (conditional mode).
        rho_mfd = (best_mfd / float(baseline)) if baseline > 0.0 else float("nan")
        return {
            "history": history,
            "best_P_s": best["hard_P_s"],
            "best_P_a": best["hard_P_a"],
            "candidates": accepted,
            "rejections": rejections,
            "all_candidates": rounded,
            "subgroup": subgroup,
            "best_holdout_manifold_distance": best_mfd,
            "identity_manifold_distance": float(baseline),
            "rho_mfd": rho_mfd,
            "acceptance_tau": float(effective_tau),
        }


# ---------------------------------------------------------------------------
# Post-training metrics: spectral gap + Lie-bracket closure error
# ---------------------------------------------------------------------------

def _spectral_gap_from_scales(scales: np.ndarray) -> dict:
    """Compute spectral gap ratio from sorted generator scales (descending)."""
    s = np.sort(np.asarray(scales, dtype=np.float64))[::-1]
    if s.size <= 1:
        return {"gap_ratio": float("inf"), "gap_index": 0, "scales_sorted": s.tolist()}
    ratios = s[:-1] / (s[1:] + 1e-12)
    best   = int(np.argmax(ratios))
    return {"gap_ratio": float(ratios[best]), "gap_index": best, "scales_sorted": s.tolist()}


def _closure_error(L_s: np.ndarray, L_a: np.ndarray) -> float:
    """Max relative bracket closure error over all generator pairs."""
    K = L_s.shape[0]
    if K < 2:
        return 0.0

    def _pack(k):
        return np.concatenate([L_s[k].reshape(-1), L_a[k].reshape(-1)])

    def _nrm(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    basis = [_nrm(_pack(k)) for k in range(K)]
    max_err = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            # Lie bracket [Li, Lj] packed as (ds²+da²,) vector
            Ls_i, La_i = L_s[i], L_a[i]
            Ls_j, La_j = L_s[j], L_a[j]
            brk = np.concatenate([
                (Ls_i @ Ls_j - Ls_j @ Ls_i).reshape(-1),
                (La_i @ La_j - La_j @ La_i).reshape(-1),
            ])
            bn = float(np.linalg.norm(brk))
            if bn <= 1e-10:
                continue
            # Project bracket onto basis
            B   = np.stack(basis, axis=1)
            coeff, *_ = np.linalg.lstsq(B, brk, rcond=None)
            residual = brk - B @ coeff
            err = float(np.linalg.norm(residual)) / (bn + 1e-12)
            max_err = max(max_err, err)
    return max_err


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def discover_liegan_symmetries(
    states,
    actions,
    next_states,
    episode_ids,
    step_ids,
    state_dim: int,
    action_dim: int,
    device=None,
    config: Optional[LieGANConfig] = None,
    verbose: bool = False,
    only_continuous: bool = False,
):
    """Discover Lie symmetries via GAN training.

    mode="marginal"   — matches p(s, a, s')
    mode="conditional"— matches p(s' | s, a)

    Returns
    -------
    dict with keys: generators, standardizer, data, training_history
    """
    del device, episode_ids, step_ids
    cfg = config or LieGANConfig()

    states_np     = _to_numpy(states).astype(np.float32)
    actions_np    = _to_numpy(actions).astype(np.float32)
    next_states_np = _to_numpy(next_states).astype(np.float32)
    z = np.concatenate([states_np, actions_np, next_states_np], axis=-1)

    splits = _build_splits(
        num_points=int(z.shape[0]),
        validation_split=float(cfg.training.validation_split),
        holdout_split=float(cfg.training.holdout_split),
        seed=int(cfg.training.seed),
    )
    z_train_raw = z[splits["train_idx"]]
    z_mean  = z_train_raw.mean(axis=0)
    z_std_a = z_train_raw.std(axis=0)
    if cfg.training.normalize_data:
        is_const = z_std_a < 1e-6
        z_scale  = np.where(is_const, 1.0, z_std_a).astype(np.float32)
        z_shift  = np.where(is_const, 0.0, z_mean).astype(np.float32)
    else:
        z_scale = np.ones_like(z_mean, dtype=np.float32)
        z_shift = np.zeros_like(z_mean, dtype=np.float32)
    z_std    = ((z - z_shift) / z_scale).astype(np.float32)
    z_train = z_std[splits["train_idx"]]

    mode = str(cfg.training.mode)
    if verbose:
        print(f"[phase] liegan/{mode}: training GAN  "
              f"(K={cfg.generator.K}, steps={cfg.training.num_steps})")

    trainer = LieGANTrainer(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        config=cfg,
    )
    history = trainer.fit(z_train, verbose=verbose)
    gens = trainer.get_generators()

    if verbose:
        for k in range(cfg.generator.K):
            print(
                f"  generator {k}: "
                f"||L_s||={float(np.linalg.norm(gens['L_s'][k])):.3f}  "
                f"||L_a||={float(np.linalg.norm(gens['L_a'][k])):.3f}  "
                f"scale={float(gens['scales'][k]):.3f}"
            )

    # ---- spectral gap ratio and closure error from learned generators ------
    gap_info = _spectral_gap_from_scales(gens["scales"])
    clos_err = _closure_error(gens["L_s"], gens["L_a"])
    if verbose:
        print(
            f"[liegan/{mode}] spectral_gap_ratio={gap_info['gap_ratio']:.3e} "
            f"(index={gap_info['gap_index']})  closure_error={clos_err:.3e}"
        )

    # ---- discrete signed-permutation search --------------------------------
    z_holdout = z_std[splits["holdout_idx"]]
    z_val     = z_std[splits["val_idx"]]
    z_eval    = (np.concatenate([z_val, z_holdout], axis=0)
                 if z_val.shape[0] > 0 else z_holdout)
    z_eval    = (z_eval if z_eval.shape[0] > 0 else z_train).astype(np.float32)

    if only_continuous:
        if verbose:
            print(f"[phase] liegan/{mode}: discrete Sinkhorn search SKIPPED (only_continuous=True)")
        discrete_result = {
            "candidates": [],
            "rejections": [],
            "all_candidates": [],
            "best_holdout_manifold_distance": float("inf"),
            "acceptance_tau": 0.0,
            "subgroup": {"num_elements": 0, "elements": []},
            "history": {"loss_total": []},
        }
    else:
        n_candidates = int(cfg.discrete.num_candidates)
        batch_size = int(cfg.discrete.candidates_batch_size)
        if batch_size <= 0 or batch_size >= n_candidates:
            if verbose:
                print(f"[phase] liegan/{mode}: discrete Sinkhorn search  "
                      f"(candidates={n_candidates}, steps={cfg.discrete.num_steps})")
            searcher = LieGANDiscreteSearch(
                disc=trainer.disc,
                z_train=z_train,
                z_holdout=z_eval,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                mode=mode,
                config=cfg.discrete,
                lambda_sparse=float(cfg.training.lambda_scale_sparse),
            )
            discrete_result = searcher.fit(verbose=verbose)
        else:
            n_batches = math.ceil(n_candidates / batch_size)
            if verbose:
                print(f"[phase] liegan/{mode}: discrete Sinkhorn search  "
                      f"({n_candidates} candidates in {n_batches} batches of {batch_size}, "
                      f"steps={cfg.discrete.num_steps})")
            import dataclasses as _dc
            _batch_cfg = _dc.replace(cfg.discrete, num_candidates=batch_size)
            _all_cands, _all_rej, _all_loss = [], [], []
            _best_dist = float("inf")
            _best_tau  = 0.0
            _best_sub  = {"num_elements": 0, "elements": []}
            _identity_mfd = float("nan")
            for _bi in range(n_batches):
                if verbose:
                    print(f"  [discrete batch {_bi + 1}/{n_batches}]")
                _searcher = LieGANDiscreteSearch(
                    disc=trainer.disc,
                    z_train=z_train,
                    z_holdout=z_eval,
                    state_dim=int(state_dim),
                    action_dim=int(action_dim),
                    mode=mode,
                    config=_batch_cfg,
                    lambda_sparse=float(cfg.training.lambda_scale_sparse),
                )
                _r = _searcher.fit(verbose=verbose)
                _all_cands.extend(_r["candidates"])
                _all_rej.extend(_r["rejections"])
                _all_loss.extend(_r["history"]["loss_total"])
                if _r["best_holdout_manifold_distance"] < _best_dist:
                    _best_dist = _r["best_holdout_manifold_distance"]
                    _best_tau  = _r["acceptance_tau"]
                    _best_sub  = _r["subgroup"]
                if _bi == 0:
                    _identity_mfd = float(_r.get("identity_manifold_distance", float("nan")))
            _rho_mfd = (_best_dist / _identity_mfd) if _identity_mfd > 0.0 and not math.isnan(_identity_mfd) else float("nan")
            discrete_result = {
                "candidates":                   _all_cands,
                "rejections":                   _all_rej,
                "all_candidates":               _all_cands + _all_rej,
                "best_holdout_manifold_distance": _best_dist,
                "identity_manifold_distance":   _identity_mfd,
                "rho_mfd":                      _rho_mfd,
                "acceptance_tau":               _best_tau,
                "subgroup":                     _best_sub,
                "history":                      {"loss_total": _all_loss},
            }

    return {
        "generators":        gens,
        "standardizer":      {"mean": z_shift, "scale": z_scale},
        "data":              {"z_train": z_train, "z_holdout": z_holdout,
                              "z_eval": z_eval, "mode": mode},
        "training_history":  history,
        "spectral_gap_ratio": gap_info["gap_ratio"],
        "closure_error":     clos_err,
        "discrete":          discrete_result,
    }