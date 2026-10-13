"""
Action-first discrete symmetry learning.

Key insight: Learn action transformations D_a directly (physically interpretable),
then derive state transformations D_s via the intertwiner relation:
  D_a = W @ D_s @ W_pinv  =>  D_s = W_pinv @ D_a @ W

This ensures action symmetries are meaningful by construction, and state
symmetries are forced to respect them.
"""

import jax
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit
from flax import linen as nn
from jax.scipy.linalg import expm
import optax
from functools import partial
from typing import Sequence,Tuple, Dict, Any, NamedTuple

import numpy as np
import matplotlib

@jit
def apply_group_element(rho, x):
    return jnp.einsum('bij,bj->bi', rho, x)

@jit
def _normalize_generators_fro(L):
    norms = jnp.linalg.norm(L, ord='fro', axis=(1, 2), keepdims=True) + 1e-8
    return L / norms

# -----------------------------------------------------------------------
# Temperature schedule for discrete training
# -----------------------------------------------------------------------
    
def discrete_temp_schedule(step: int, config: Dict) -> float:
    """
    Anneal Gumbel temperature from temp_start to temp_end over temp_steps.
    Lower temperature = more discrete permutation.
    Start soft for exploration, end hard for clean discrete solution.
    """
    t_start = float(config.get('disc_temp_start', 1.0))
    t_end   = float(config.get('disc_temp_end',   0.1))
    t_steps = int(config.get('disc_temp_steps', 100_000))
    frac    = min(step / t_steps, 1.0)
    return t_start * (t_end / t_start) ** frac

# -----------------------------------------------------------------------
# Sparsity regularization
# -----------------------------------------------------------------------
 
def sparsity_losses(gen_params: Dict, config: Dict) -> Dict[str, jnp.ndarray]:
    """L1 sparsity on L_s and W (optional)."""
    inner = gen_params['params']
    L_s   = inner['L_s']
    W     = inner['W']
    R_sparse_Ls = jnp.mean(jnp.abs(L_s))
    R_sparse_W  = jnp.mean(jnp.abs(W))
    return {
        'R_sparse_Ls': R_sparse_Ls,
        'R_sparse_W':  R_sparse_W,
    }

# -----------------------------------------------------------------------
# Discrete symmetry module - ACTION FIRST
# -----------------------------------------------------------------------
class DiscreteSymmetriesActionFirst(nn.Module):
    """
    Learn action-space transformations D_a directly,
    then derive state-space transformations D_s via intertwiner W.
    """
    n_s:   int
    n_a:   int
    n_ops: int 

    @nn.compact
    def __call__(self):
        # Learn ACTION-side transformations directly
        if self.n_a == 1:
            # For 1D actions, just learn sign flips (±1)
            sign_logits_a = self.param(
                'sign_logits_a',
                jax.nn.initializers.zeros,
                (self.n_ops,))
            return {'sign_logits_a': sign_logits_a, 'mode': '1d'}
        else:
            # For higher-dimensional actions, learn permutation + signs
            perm_logits_a = self.param(
                'perm_logits_a',
                jax.nn.initializers.normal(0.1),
                (self.n_ops, self.n_a, self.n_a))
            sign_logits_a = self.param(
                'sign_logits_a',
                jax.nn.initializers.zeros,
                (self.n_ops, self.n_a))
            return {
                'perm_logits_a': perm_logits_a,
                'sign_logits_a': sign_logits_a,
                'mode': 'nd'
            }

    def get_intertwined_op(self, variables, gen_params, op_idx, rng, temp=1.0, hard=False):
        """
        Construct D_a (learned). Construct D_s from L_s using a deterministic alpha per op.

        NOTE: We no longer compute D_s = W_pinv @ D_a @ W (too unstable early).
        Compatibility is enforced via penalties elsewhere (e.g., ||D_a W - W D_s||^2
        and/or continuous generator intertwiner).
        """
        out = self.apply(variables)

        # 1) ACTION op D_a learned directly
        if out['mode'] == '1d':
            s_logits = jax.lax.dynamic_slice_in_dim(out['sign_logits_a'], op_idx, 1, axis=0)
            s_a = self._compute_signs(s_logits, hard)  # (1,)
            D_a = jnp.array([[s_a[0]]])
        else:
            P_logits = jax.lax.dynamic_slice_in_dim(out['perm_logits_a'], op_idx, 1, axis=0)[0]
            s_logits = jax.lax.dynamic_slice_in_dim(out['sign_logits_a'], op_idx, 1, axis=0)[0]
            P_a = self._gumbel_sinkhorn(P_logits, rng, temp, hard)
            s_a = self._compute_signs(s_logits, hard)
            D_a = P_a @ jnp.diag(s_a)

        # 2) STATE op D_s constructed from continuous L_s with a fixed alpha for each op
        #    Ensures each discrete op is a stable "direction" in the learned Lie algebra.
        K = gen_params['params']['L_s'].shape[0]
        alpha_scale = float(gen_params.get('alpha_scale', 1.0)) if isinstance(gen_params, dict) else 1.0
        alpha_scale = float(getattr(self, "alpha_scale", 1.0)) if hasattr(self, "alpha_scale") else alpha_scale
        alpha_scale = 1.0  # keep deterministic unless you explicitly want bigger steps

        alpha = jnp.zeros((K,), dtype=gen_params['params']['L_s'].dtype)
        alpha = alpha.at[jnp.mod(op_idx, K)].set(alpha_scale)

        Ls = _normalize_generators_fro(gen_params['params']['L_s'])
        Ls_comb = jnp.einsum('k,kij->ij', alpha, Ls)
        D_s = expm(Ls_comb)

        def _sanitize(x):
            x = jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x)
            return jnp.clip(x, -1e3, 1e3)

        return _sanitize(D_s), _sanitize(D_a)

    def get_transformation(self, variables, gen_params, op_idx, rng, temp=1.0, hard=False):
        """Alias for get_intertwined_op for compatibility."""
        return self.get_intertwined_op(variables, gen_params, op_idx, rng, temp, hard)

    def sample_random_op(self, variables, gen_params, rng, batch_size, temp=1.0, hard=False):
        out = self.apply(variables)
        
        if out['mode'] == '1d':
            n_ops = out['sign_logits_a'].shape[0]
        else:
            n_ops = out['perm_logits_a'].shape[0]
        
        rng, idx_rng = jax.random.split(rng)
        op_idx = jax.random.randint(idx_rng, (), 0, n_ops)
        
        D_s, D_a = self.get_transformation(variables, gen_params, op_idx, rng, temp, hard)
        
        # Broadcast to batch
        D_s_batch = jnp.broadcast_to(D_s[None], (batch_size,) + D_s.shape)
        D_a_batch = jnp.broadcast_to(D_a[None], (batch_size,) + D_a.shape)
        return D_s_batch, D_a_batch
    
    @staticmethod
    def _gumbel_sinkhorn(
        logits: jnp.ndarray,
        rng: jax.Array,
        temp: float = 1.0,
        n_iters: int = 20,
        hard: bool = False,
    ) -> jnp.ndarray:
        n = logits.shape[0]
        logits = jnp.clip(logits, -100.0, 100.0)
 
        if rng is not None and not hard:
            uniform = jax.random.uniform(
                rng, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
            gumbel  = jnp.clip(-jnp.log(-jnp.log(uniform)), -5.0, 5.0)
            logits  = logits + gumbel
 
        log_alpha = jnp.clip(logits / jnp.maximum(temp, 0.1), -15.0, 15.0)
 
        for _ in range(n_iters):
            log_alpha = log_alpha - jax.scipy.special.logsumexp(
                log_alpha, axis=1, keepdims=True)
            log_alpha = jnp.clip(log_alpha, -15.0, 15.0)
            log_alpha = log_alpha - jax.scipy.special.logsumexp(
                log_alpha, axis=0, keepdims=True)
            log_alpha = jnp.clip(log_alpha, -15.0, 15.0)
 
        soft_perm = jnp.clip(jnp.exp(log_alpha), 0.0, 1.0)
        soft_perm = soft_perm / (soft_perm.sum(axis=1, keepdims=True) + 1e-8)
        soft_perm = jnp.where(
            jnp.isnan(soft_perm) | jnp.isinf(soft_perm),
            jnp.eye(n), soft_perm)
 
        if hard:
            hard_perm = jax.nn.one_hot(jnp.argmax(soft_perm, axis=1), n)
            soft_perm = hard_perm + soft_perm - jax.lax.stop_gradient(soft_perm)
 
        return soft_perm
    
    @staticmethod
    def _compute_signs(
        sign_logits: jnp.ndarray,
        hard: bool = False,
    ) -> jnp.ndarray:
        """sigmoid -> [0,1] -> rescale to [-1,1]."""
        probs = jax.nn.sigmoid(sign_logits)
        if hard:
            hard_signs = jnp.where(probs > 0.5, 1.0, -1.0)
            probs = hard_signs + probs - jax.lax.stop_gradient(probs)
        return 2.0 * probs - 1.0

# -----------------------------------------------------------------------
# Networks (unchanged)
# -----------------------------------------------------------------------

class SNDense(nn.Module):
    features: int
    n_power_iter: int = 1

    @nn.compact
    def __call__(self, x, update_stats=True):
        in_features = x.shape[-1]
        W = self.param('kernel', nn.initializers.lecun_normal(),
                       (in_features, self.features))
        b = self.param('bias', nn.initializers.zeros, (self.features,))
        u = self.variable('sn_state', 'u',
                          lambda shape: jax.random.normal(jax.random.PRNGKey(0), shape),
                          (in_features,))
        u0 = u.value
        for _ in range(self.n_power_iter):
            v  = W.T @ u0; v  = v  / (jnp.linalg.norm(v)  + 1e-8)
            u0 = W   @ v;  u0 = u0 / (jnp.linalg.norm(u0) + 1e-8)
        if update_stats:
            u.value = u0
        sigma = u0 @ W @ v
        return x @ (W / (sigma + 1e-8)) + b


class Critic(nn.Module):
    """Full triple critic for L_s learning."""
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, s, a, s_next, update_stats=True):
        x = jnp.concatenate([s, a, s_next], axis=-1)
        for dim in self.hidden_dims:
            x = nn.relu(SNDense(dim)(x, update_stats=update_stats))
        return SNDense(1)(x, update_stats=update_stats).squeeze(-1)

class RatioNetwork(nn.Module):
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, s):
        x = s
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        return nn.softplus(nn.Dense(1)(x)).squeeze(-1)

# [MLPEncoder, MLPDecoder, EncoderState, POMDPEncoderLearner remain unchanged]
class MLPEncoder(nn.Module):
    latent_dim: int
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
        z = nn.Dense(self.latent_dim)(x)
        return z

class MLPDecoder(nn.Module):
    obs_dim: int
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = z
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
        x = nn.Dense(self.obs_dim)(x)
        return x

class EncoderState(NamedTuple):
    enc_params: Any
    dec_params: Any
    enc_opt: Any
    dec_opt: Any
    rng: jax.Array

class POMDPEncoderLearner:
    def __init__(self, obs_dim: int, latent_dim: int, lr: float = 3e-4, seed: int = 0,
                 hidden_dims=(128,128)):
        self.encoder = MLPEncoder(latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.decoder = MLPDecoder(obs_dim=obs_dim, hidden_dims=hidden_dims)

        rng = jax.random.PRNGKey(seed)
        rng, ek, dk = jax.random.split(rng, 3)
        dummy = jnp.zeros((1, obs_dim))
        enc_vars = self.encoder.init(ek, dummy)
        dec_vars = self.decoder.init(dk, jnp.zeros((1, latent_dim)))

        tx = optax.adam(lr)
        enc_params = enc_vars["params"]
        dec_params = dec_vars["params"]

        self.state = EncoderState(
            enc_params=enc_params,
            dec_params=dec_params,
            enc_opt=tx.init(enc_params),
            dec_opt=tx.init(dec_params),
            rng=rng,
        )
        self._tx = tx

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.encoder.apply({"params": self.state.enc_params}, obs)

    @jax.jit
    def update(self, state: EncoderState, obs: jnp.ndarray) -> Tuple[EncoderState, Dict[str, jnp.ndarray]]:
        def loss_fn(enc_p, dec_p):
            z = self.encoder.apply({"params": enc_p}, obs)
            recon = self.decoder.apply({"params": dec_p}, z)
            recon_loss = jnp.mean(jnp.sum((recon - obs) ** 2, axis=-1))
            z_reg = 1e-4 * jnp.mean(z**2)
            return recon_loss + z_reg, {"enc/recon_loss": recon_loss, "enc/z_reg": z_reg}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)(state.enc_params, state.dec_params)
        enc_grads, dec_grads = grads

        enc_upd, new_eopt = self._tx.update(enc_grads, state.enc_opt, state.enc_params)
        dec_upd, new_dopt = self._tx.update(dec_grads, state.dec_opt, state.dec_params)

        new_state = state._replace(
            enc_params=optax.apply_updates(state.enc_params, enc_upd),
            dec_params=optax.apply_updates(state.dec_params, dec_upd),
            enc_opt=new_eopt,
            dec_opt=new_dopt
        )
        return new_state, {"enc/ae_loss": loss, **metrics}

# -----------------------------------------------------------------------
# Generator (unchanged - continuous symmetries still L_s first)
# -----------------------------------------------------------------------

class SymmetryGenerators(nn.Module):
    n_s: int
    n_a: int
    K: int
    eps_pinv:    float = 1e-4
    alpha_scale: float = 1.0
    @nn.compact
    def __call__(self):
        # Learn both generators directly
        L_s_raw = self.param('L_s', jax.nn.initializers.normal(0.1),
                             (self.K, self.n_s, self.n_s))
        L_a_raw = self.param('L_a', jax.nn.initializers.normal(0.1),
                             (self.K, self.n_a, self.n_a))

        W = self.param('W', jax.nn.initializers.normal(0.1),
                       (self.n_a, self.n_s))

        L_s_normed = _normalize_generators_fro(L_s_raw)
        L_a_normed = _normalize_generators_fro(L_a_raw)

        # Keep W numerically stable; don't build W_pinv in the forward path.
        W_norm = jnp.maximum(jnp.linalg.norm(W, axis=1, keepdims=True), 1e-3)
        W_safe = W / W_norm

        return {
            'L_s_raw': L_s_raw,
            'L_s_normed': L_s_normed,
            'L_a_raw': L_a_raw,
            'L_a_normed': L_a_normed,
            'W': W_safe,
        }
    
    def sample_transform(self, variables, rng, batch_size):
        out   = self.apply(variables)
        alpha = jax.random.uniform(
            rng, (batch_size, self.K),
            minval=-self.alpha_scale,
            maxval=self.alpha_scale
        )
        rho_s = vmap(expm)(jnp.einsum('bk,kij->bij', alpha, out['L_s_normed']))
        rho_a = vmap(expm)(jnp.einsum('bk,kij->bij', alpha, out['L_a_normed']))
        return rho_s, rho_a
    
    def regularization_losses(self, variables, s_batch, eps_W=1e-3):
        out = self.apply(variables)
        Ls, W, La = out['L_s_normed'], out['W'], out['L_a_normed']
        K = Ls.shape[0]

        sv  = jnp.linalg.svd(W, compute_uv=False)
        R_W = jnp.maximum(eps_W - jnp.min(sv), 0.0) ** 2
        R_W = jnp.clip(R_W, 0.0, 1e4)

        col_norms = jnp.linalg.norm(W, axis=0)
        row_norms = jnp.linalg.norm(W, axis=1)
        min_norm = 0.1
        R_row_alive = jnp.mean(jnp.maximum(min_norm - row_norms, 0.0) ** 2)
        R_col_alive = jnp.mean(jnp.maximum(min_norm - col_norms, 0.0) ** 2)
        R_balance   = jnp.clip(R_row_alive + R_col_alive, 0.0, 1e4)

        # Non-triviality (same as before, on Ls)
        alpha_test   = jnp.ones((s_batch.shape[0], K)) * self.alpha_scale
        Ls_comb      = jnp.einsum('bk,kij->bij', alpha_test, Ls)
        rho_s_test   = vmap(expm)(Ls_comb)
        s_t          = apply_group_element(rho_s_test, s_batch)
        displacement = jnp.mean(jnp.sum((s_t - s_batch) ** 2, axis=-1))
        R_nontrivial = -displacement

        # Diversity on Ls (unchanged idea)
        if K > 1:
            flat  = Ls.reshape(K, -1)
            gram  = flat @ flat.T
            mask  = 1.0 - jnp.eye(K, dtype=gram.dtype)
            R_div = jnp.sum((gram * mask) ** 2) / jnp.maximum(K * (K - 1), 1)
        else:
            R_div = jnp.array(0.0, dtype=Ls.dtype)

        # Rep regularizer: disable or re-define (since we no longer push via W_pinv)
        R_rep = jnp.array(0.0, dtype=Ls.dtype)

        # Proper intertwiner residual: || W Ls - La W ||^2
        WLs = jnp.einsum('ij,kjl->kil', W, Ls)      # (K, n_a, n_s)
        LaW = jnp.einsum('kij,jl->kil', La, W)      # (K, n_a, n_s)
        R_intertwiner = jnp.mean((WLs - LaW) ** 2)
        R_intertwiner = jnp.clip(R_intertwiner, 0.0, 1e4)

        return {
            'R_W': R_W,
            'R_nontrivial': R_nontrivial,
            'R_diverse': R_div,
            'R_rep': R_rep,
            'R_balance': R_balance,
            'R_intertwiner': R_intertwiner,
        }
    

    def intertwine_residual(self, variables) -> jnp.ndarray:
        """
        Explicit intertwiner residual ||W L_s - L_a W||_F^2 averaged over K.
        Uses *live* W and L, without any stop_gradient.
        """
        out = self.apply(variables)
        L, W, L_a = out['L_s_normed'], out['W'], out['L_a_normed']
        # W:   (n_a, n_s)
        # L:   (K, n_s, n_s)
        # L_a: (K, n_a, n_a)
        WLs = jnp.einsum('ij,kjl->kil', W, L)     # (K, n_a, n_s)
        LaW = jnp.einsum('kij,jl->kil', L_a, W)   # (K, n_a, n_s)
        resid = jnp.mean((WLs - LaW) ** 2)
        return jnp.clip(resid, 0.0, 1e4)

# -----------------------------------------------------------------------
# State
# -----------------------------------------------------------------------

class DiscoveryState(NamedTuple):
    gen_params:     Any
    la_opt:         Any
    w_opt:          Any
    critic_params:  Any
    critic_sn:      Any
    critic_opt:     Any
    ratio_params:   Any
    ratio_opt:      Any
    rng:            jax.Array
    discrete_params: Any
    discrete_opt:    Any
    enc_params:     Any | None = None
    dec_params:     Any | None = None
    enc_opt:        Any | None = None
    dec_opt:        Any | None = None
    critic_target_params: Any | None = None  # NEW


# -----------------------------------------------------------------------
# Trainer (key changes in discrete_model initialization and _update)
# -----------------------------------------------------------------------

class SymmetryDiscovery:
    def __init__(self, n_s, n_a, config, seed=0, use_encoder: bool = False,
                 encoder_obs_dim: int | None = None):
        self.n_s = n_s
        self.n_a = n_a
        self.K   = int(config['K'])
        self.config = config

        rng = jax.random.PRNGKey(seed)
        rng, gk, ck, rk = jax.random.split(rng, 4)
        self.use_encoder = bool(use_encoder)

        # Encoder init 
        enc_params = dec_params = enc_opt = dec_opt = None
        if self.use_encoder:
            if encoder_obs_dim is None:
                raise ValueError("encoder_obs_dim must be provided when use_encoder=True.")
            latent_dim  = int(config.get("latent_dim", n_s))
            enc_lr      = float(config.get("enc_lr", 3e-4))
            enc_hiddens = tuple(config.get("enc_hidden_dims", (128, 128)))

            self._encoder = MLPEncoder(latent_dim=latent_dim, hidden_dims=enc_hiddens)
            self._decoder = MLPDecoder(obs_dim=encoder_obs_dim, hidden_dims=enc_hiddens)

            rng, ek, dk = jax.random.split(rng, 3)
            dummy_obs   = jnp.zeros((1, encoder_obs_dim))
            enc_vars    = self._encoder.init(ek, dummy_obs)
            dec_vars    = self._decoder.init(dk, jnp.zeros((1, latent_dim)))

            self._enc_tx = optax.adam(enc_lr)

            enc_params = enc_vars["params"]
            dec_params = dec_vars["params"]
            enc_opt    = self._enc_tx.init(enc_params)
            dec_opt    = self._enc_tx.init(dec_params)
        else:
            self._encoder = None
            self._decoder = None
            self._enc_tx  = None

        # Generator (continuous symmetries)
        self.gen_model = SymmetryGenerators(
            n_s=n_s, n_a=n_a, K=self.K,
            eps_pinv=float(config.get('eps_pinv', 1e-4)),
            alpha_scale=float(config.get('alpha_scale', 1.0)),
        )
        gen_vars = self.gen_model.init(gk)

        # Separate optimizers for L_s and W
        gen_lr = float(config['gen_lr'])
        w_lr   = float(config.get('w_lr', gen_lr * 3.0))
        clip   = float(config.get('gen_grad_clip', 1.0))

        def _la_mask(p):
            return {'params': {'L_s': False, 'L_a': True,  'W': False}}
        def _lsw_mask(p):
            return {'params': {'L_s': True, 'L_a': False, 'W': True}}

        self._la_tx = optax.chain(
            optax.clip_by_global_norm(clip),
            optax.masked(optax.adam(gen_lr), _la_mask),
        )
        self._lsw_tx = optax.chain(
            optax.clip_by_global_norm(clip),
            optax.masked(optax.adam(w_lr), _lsw_mask),
        )

        # Critics
        self.critic = Critic(
            hidden_dims=tuple(config.get('critic_hidden_dims', (256, 256))))
        dummy3 = (jnp.zeros((1,n_s)), jnp.zeros((1,n_a)), jnp.zeros((1,n_s)))
        cv = self.critic.init(ck, *dummy3, update_stats=False)

        critic_tx    = optax.chain(
            optax.clip_by_global_norm(float(config.get('critic_grad_clip', 5.0))),
            optax.adam(config['f_xi_lr']))
        self._critic_tx    = critic_tx
                
        # NEW: Target critic parameters and tau, higher tau means faster updates (more bias, less variance)
        self.critic_target_tau = float(config.get("critic_target_tau", 0.005))  # NEW

        # Ratio network
        self.ratio_network = RatioNetwork(
            hidden_dims=tuple(config.get('ratio_hidden_dims', (256, 256))))
        rv = self.ratio_network.init(rk, jnp.zeros((1, n_s)))
        ratio_tx = optax.chain(
            optax.clip_by_global_norm(float(config.get('ratio_grad_clip', 5.0))),
            optax.adam(config['v_psi_lr']))
        self._ratio_tx = ratio_tx

        # KEY CHANGE: Use DiscreteSymmetriesActionFirst
        self.discrete_model = DiscreteSymmetriesActionFirst(
            n_s=n_s, n_a=n_a,
            n_ops=int(config.get('n_discrete_ops', 2)))
        disc_vars = self.discrete_model.init(rng)
        disc_tx = optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adam(config.get('discrete_lr', 1e-3)),
        )
        self.disc_tx = disc_tx

        self.state = DiscoveryState(
            gen_params        = gen_vars,
            la_opt            = self._la_tx.init(gen_vars),
            w_opt             = self._lsw_tx.init(gen_vars),
            critic_params     = cv['params'],
            critic_sn         = {'sn_state': cv['sn_state']},
            critic_opt        = critic_tx.init(cv['params']),
            ratio_params      = rv,
            ratio_opt         = ratio_tx.init(rv),
            rng               = rng,
            discrete_params   = disc_vars['params'],
            discrete_opt      = disc_tx.init(disc_vars['params']),
            enc_params        = enc_params,
            dec_params        = dec_params,
            enc_opt           = enc_opt,
            dec_opt           = dec_opt,
            critic_target_params= cv['params']
        )

    def _encode(self, enc_params, obs: jnp.ndarray) -> jnp.ndarray:
        if self._encoder is None:
            return obs
        return self._encoder.apply({"params": enc_params}, obs)

    def _encoder_loss(self, enc_p, dec_p, obs):
        z = self._encoder.apply({"params": enc_p}, obs)
        recon = self._decoder.apply({"params": dec_p}, z)
        recon_loss = jnp.mean(jnp.sum((recon - obs) ** 2, axis=-1))
        z_reg = 1e-4 * jnp.mean(z**2)
        loss = recon_loss + z_reg
        metrics = {
            "enc/recon_loss": recon_loss,
            "enc/z_reg": z_reg,
            "enc/ae_loss": loss,
        }
        return loss, metrics
    
    # Critic helpers (unchanged)
    def _cfwd(self, cp, sn, s, a, sn_arr, mutable=True):
        vars_ = {'params': cp, **sn}
        out, new_sn = self.critic.apply(
            vars_, s, a, sn_arr,
            update_stats=mutable, mutable=['sn_state'])
        return out, new_sn

    def _cfwd_frozen(self, cp, sn, s, a, sn_arr):
        vars_ = {'params': cp, **sn}
        out, _ = self.critic.apply(
            vars_, s, a, sn_arr, update_stats=False, mutable=['sn_state'])
        return out
    
    def _update_target_critic(self, target_params, online_params):
        tau = self.critic_target_tau
        return jax.tree.map(lambda t, o: (1.0 - tau) * t + tau * o,
                            target_params, online_params)

    def _sample_combined(self, rng, gen_params, disc_params, batch_size, disc_temp):
        """Sample from union of continuous and discrete transformations."""
        rng_gate, rng_cont, rng_disc, rng_lr = jax.random.split(rng, 4)
        
        p_c_target = self.config.get('p_continuous', 0.9)
        p_d_target = self.config.get('p_discrete', 0.1)

        p_both   = p_c_target * p_d_target
        p_only_c = p_c_target * (1.0 - p_d_target)
        p_only_d = (1.0 - p_c_target) * p_d_target
        
        probs = jnp.array([p_both, p_only_c, p_only_d])
        probs /= jnp.sum(probs)
        
        mode = jax.random.choice(rng_gate, 3, p=probs)
        
        do_cont = jnp.logical_or(mode == 0, mode == 1)
        do_disc = jnp.logical_or(mode == 0, mode == 2)

        # Sample continuous transform
        rho_s_c, rho_a_c = self.gen_model.sample_transform(gen_params, rng_cont, batch_size)
        
        # Get discrete op (D_a learned, D_s derived!)
        out = self.discrete_model.apply({'params': disc_params})
        if out['mode'] == '1d':
            n_ops = out['sign_logits_a'].shape[0]
        else:
            n_ops = out['perm_logits_a'].shape[0]
        
        op_idx = jax.random.randint(rng_disc, (), 0, n_ops)
        D_s, D_a = self.discrete_model.get_intertwined_op(
            {'params': disc_params}, gen_params, op_idx, rng_disc, temp=disc_temp)
        
        # Broadcast discrete ops to batch
        D_s_b = jnp.broadcast_to(D_s, (batch_size, self.n_s, self.n_s))
        D_a_b = jnp.broadcast_to(D_a, (batch_size, self.n_a, self.n_a))
        
        # Decide left vs right application by a single scalar coin flip
        u = jax.random.uniform(rng_lr, ())
        left_first = u < 0.5  # scalar bool

        def _compose(rho_c, D_b):
            # Build either D @ rho or rho @ D
            composed = jax.lax.select(left_first, D_b @ rho_c, rho_c @ D_b)
            return composed

        I_s = jnp.broadcast_to(jnp.eye(self.n_s), (batch_size, self.n_s, self.n_s))
        I_a = jnp.broadcast_to(jnp.eye(self.n_a), (batch_size, self.n_a, self.n_a))

        # Optional discrete and continuous parts
        rho_s_disc = jax.lax.select(do_disc, D_s_b, I_s)
        rho_a_disc = jax.lax.select(do_disc, D_a_b, I_a)

        rho_s_cont = jax.lax.select(do_cont, rho_s_c, I_s)
        rho_a_cont = jax.lax.select(do_cont, rho_a_c, I_a)

        # When both present, mix left/right; when only one present, that's just identity-other
        rho_s = _compose(rho_s_cont, rho_s_disc)
        rho_a = _compose(rho_a_cont, rho_a_disc)

        return rho_s, rho_a
   
    # Loss functions (unchanged)
    def _ratio_loss(self, ratio_params, gen_params, disc_params, s, rng, disc_temp):
        rho_s, _ = self._sample_combined(rng, gen_params, disc_params, s.shape[0], disc_temp)
        s_t = jax.lax.stop_gradient(apply_group_element(rho_s, s))
        
        v_s = self.ratio_network.apply(ratio_params, s)
        v_t = self.ratio_network.apply(ratio_params, s_t)
        return 0.5 * jnp.mean(v_s**2) - jnp.mean(v_t)

    def _la_loss(self, gen_params, cp_targ, sn_targ, ratio_params, s, a, s_next, rng):
        """Update L_a only. W and L_s are stop-grad."""
        inner     = gen_params['params']
        sg_params = {
            'params': {
                **inner,
                'W':   jax.lax.stop_gradient(inner['W']),
                'L_s': jax.lax.stop_gradient(inner['L_s']),
            }
        }

        rho_s, rho_a = self.gen_model.sample_transform(sg_params, rng, s.shape[0])
        s_t  = apply_group_element(rho_s, s)
        a_t  = apply_group_element(rho_a, a)
        sn_t = apply_group_element(rho_s, s_next)

        v_s    = jax.lax.stop_gradient(self.ratio_network.apply(ratio_params, s))
        cp_sg  = jax.lax.stop_gradient(cp_targ)
        sn_sg  = jax.lax.stop_gradient(sn_targ)

        f_fake = self._cfwd_frozen(cp_sg, sn_sg, s_t, a_t, sn_t)
        f_real = self._cfwd_frozen(cp_sg, sn_sg, s,   a,   s_next)
        L_f    = jnp.mean(f_fake) - jnp.mean(f_real * v_s)

        ratio_sg  = jax.lax.stop_gradient(ratio_params)
        v_s_tilde = self.ratio_network.apply(ratio_sg, s_t)
        R_ratio   = jnp.mean((v_s_tilde - 1.0) ** 2)

        # Light regularization that does not touch W/L_s in this step
        regs = self.gen_model.regularization_losses(
            sg_params, s_batch=s,
            eps_W=float(self.config.get('eps_W', 1e-3))
        )

        L = (
            L_f
            + float(self.config.get('lambda_ratio', 1.0)) * R_ratio
            + float(self.config.get('lambda_diverse', 1.0)) * regs['R_diverse']
            + float(self.config.get('lambda_nontrivial', 0.1)) * regs['R_nontrivial']
        )
        return L, {
            'loss/L_f': L_f,
            'loss/R_ratio': R_ratio,
            'reg/R_nontrivial': regs['R_nontrivial'],
            'reg/R_diverse': regs['R_diverse'],
        }


    def _ls_loss(self, gen_params, disc_params, cp_targ, sn_targ, ratio_params, s, a, s_next, rng, disc_temp):
        """
        Update L_s and W (L_a stop-grad).
        This is the old _w_loss, but now it's the L_s objective.
        """
        inner = gen_params['params']
        sg_params = {'params': {**inner, 'L_a': jax.lax.stop_gradient(inner['L_a'])}}

        rho_s, rho_a = self._sample_combined(rng, sg_params, disc_params, s.shape[0], disc_temp)
        s_t     = apply_group_element(rho_s, s)
        a_t     = apply_group_element(rho_a, a)
        s_next_t= apply_group_element(rho_s, s_next)

        def _sanitize(x):
            x = jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x)
            return jnp.clip(x, -1e3, 1e3)

        s_t, a_t, s_next_t = _sanitize(s_t), _sanitize(a_t), _sanitize(s_next_t)

        cp_sg = jax.lax.stop_gradient(cp_targ)
        sn_sg = jax.lax.stop_gradient(sn_targ)

        f_fake = self._cfwd_frozen(cp_sg, sn_sg, s_t, a_t, s_next_t)
        f_real = self._cfwd_frozen(cp_sg, sn_sg, s,   a,   s_next)

        v_s = jax.lax.stop_gradient(self.ratio_network.apply(ratio_params, s))
        L_ipm = jnp.mean(f_fake) - jnp.mean(f_real * v_s)

        # Ratio penalty (keeps transforms from exploiting v_psi)
        ratio_sg  = jax.lax.stop_gradient(ratio_params)
        v_s_tilde = self.ratio_network.apply(ratio_sg, s_t)
        R_ratio   = jnp.mean((v_s_tilde - 1.0) ** 2)

        # Regularizers on LIVE (L_s, W). L_a is stop-grad via sg_params.
        regs = self.gen_model.regularization_losses(
            sg_params, s_batch=s, eps_W=float(self.config.get('eps_W', 1e-3))
        )
        sparse = sparsity_losses(sg_params, self.config)

        L = (
            L_ipm
            + float(self.config.get('lambda_ratio', 1.0)) * R_ratio
            + float(self.config.get('lambda_W', 1.0)) * regs['R_W']
            + float(self.config.get('lambda_balance', 0.01)) * regs['R_balance']
            + float(self.config.get('lambda_diverse', 1.0)) * regs['R_diverse']
            + float(self.config.get('lambda_nontrivial', 0.1)) * regs['R_nontrivial']
            + float(self.config.get('lambda_intertwiner', 0.1)) * regs['R_intertwiner']
            + float(self.config.get('lambda_sparse_W', 0.01)) * sparse['R_sparse_W']
            + float(self.config.get('lambda_sparse_Ls', 0.01)) * sparse['R_sparse_Ls']
        )

        return L, {
            'loss/L_ls': L,
            'loss/L_ipm_ls': L_ipm,
            'loss/R_ratio_ls': R_ratio,
            'reg/R_W': regs['R_W'],
            'reg/R_balance': regs['R_balance'],
            'reg/R_nontrivial': regs['R_nontrivial'],
            'reg/R_diverse': regs['R_diverse'],
            'reg/R_intertwiner': regs['R_intertwiner'],
            'reg/R_sparse_W': sparse['R_sparse_W'],
            'reg/R_sparse_Ls': sparse['R_sparse_Ls'],
        }
    
    # def _w_loss(self, gen_params, disc_params, cp_targ, sn_targ, ratio_params, s, a, s_next, rng, disc_temp):

    #     """W learns from union of continuous and discrete pairs."""
    #     inner = gen_params['params']
    #     # sg_params = {'params': {**inner, 'L_s_normed': jax.lax.stop_gradient(inner['L_s_normed'])}}
    #     # sg_params = {'params': {**inner, 'L_a_normed': jax.lax.stop_gradient(inner['L_a_normed'])}}
    #     sg_params = {'params': {**inner, 'L_a': jax.lax.stop_gradient(inner['L_a'])}}

    #     rho_s, rho_a = self._sample_combined(rng, sg_params, disc_params, s.shape[0], disc_temp)
    #     # s_t, a_t = apply_group_element(rho_s, s), apply_group_element(rho_a, a)
    #     s_t, a_t, s_next_t = apply_group_element(rho_s, s), apply_group_element(rho_a, a), apply_group_element(rho_s, s_next)

    #     def _sanitize(x):
    #         x = jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x)
    #         return jnp.clip(x, -1e3, 1e3)

    #     s_t = _sanitize(s_t)
    #     s_next_t = _sanitize(s_next_t)
    #     a_t = _sanitize(a_t)

    #     cp_sg = jax.lax.stop_gradient(cp_targ)
    #     sn_sg = jax.lax.stop_gradient(sn_targ)

    #     f_fake = self._cfwd_frozen(cp_sg, sn_sg, s_t, a_t, s_next_t)
    #     f_real = self._cfwd_frozen(cp_sg, sn_sg, s,   a,   s_next)

    #     # use same baseline factor v_s as in _la_loss
    #     v_s = jax.lax.stop_gradient(self.ratio_network.apply(ratio_params, s))

    #     L_w = jnp.mean(f_fake) - jnp.mean(f_real * v_s)
    #     # old uncorrected version without baseline:
    #     # L_w = jnp.mean(f_fake) - jnp.mean(f_real)

    #     # sparsity on W
    #     sparse = sparsity_losses(gen_params, self.config)
    #     L_w += float(self.config.get('lambda_sparse_W', 0.01)) * sparse['R_sparse_W']

    #     # W-related regs on *live* params
    #     regs_live = self.gen_model.regularization_losses(
    #         gen_params, s_batch=s, eps_W=float(self.config.get('eps_W', 1e-3))
    #     )

    #     # If you want R_rep to also act through W:

    #     # Intertwiner residual that sees W
    #     R_intertwine_W = self.gen_model.intertwine_residual(gen_params)


    #     L_w += float(self.config.get('lambda_W',       1.0))  * regs_live['R_W']
    #     L_w += float(self.config.get('lambda_balance', 0.01)) * regs_live['R_balance']
    #     L_w += float(self.config.get('lambda_rep',     0.01))  * regs_live['R_rep']
    #     L_w += float(self.config.get('lambda_intertwine_W', 0.1)) * R_intertwine_W

    #     return L_w, {
    #         'loss/L_w':            L_w,
    #         'reg/R_sparse_W':      sparse['R_sparse_W'],
    #         'reg/R_W_live':        regs_live['R_W'],
    #         'reg/R_balance_live':  regs_live['R_balance'],
    #         'reg/R_rep_live':      regs_live['R_rep'],
    #         'reg/R_intertwine_W':  R_intertwine_W,
    #     }
    
    # Main update function
    @partial(jit, static_argnames=['self', 'n_critic_steps', 'use_encoder'])
    def _update(self, state, s, a, s_next, n_critic_steps, disc_temp, use_encoder: bool):

        rng = state.rng
        # rng, r_rng, c_rng, sa_rng, ls_rng, w_rng, d_rng, e_rng = jax.random.split(rng, 8)
        rng, r_rng, c_rng, ls_rng, w_rng, d_rng, e_rng = jax.random.split(rng, 7)

        # Encoder update (if enabled)
        enc_metrics = {}
        new_enc_params = state.enc_params
        new_dec_params = state.dec_params
        new_enc_opt    = state.enc_opt
        new_dec_opt    = state.dec_opt

        if use_encoder:
            def _enc_loss(enc_p, dec_p):
                return self._encoder_loss(enc_p, dec_p, s)

            (e_loss, e_m), grads = jax.value_and_grad(_enc_loss, argnums=(0, 1), has_aux=True)(
                state.enc_params,
                state.dec_params,
            )
            enc_grads, dec_grads = grads
            
            enc_upd, new_enc_opt = self._enc_tx.update(enc_grads, state.enc_opt, state.enc_params)
            dec_upd, new_dec_opt = self._enc_tx.update(dec_grads, state.dec_opt, state.dec_params)

            new_enc_params = optax.apply_updates(state.enc_params, enc_upd)
            new_dec_params = optax.apply_updates(state.dec_params, dec_upd)
            enc_metrics = e_m

            s      = self._encoder.apply({"params": new_enc_params}, s)
            s_next = self._encoder.apply({"params": new_enc_params}, s_next)

        # 1) Ratio Update
        r_loss, r_grads = value_and_grad(self._ratio_loss)(
            state.ratio_params, state.gen_params, state.discrete_params, s, r_rng, disc_temp)
        r_upd, new_ratio_opt = self._ratio_tx.update(
            r_grads, state.ratio_opt, state.ratio_params)
        new_ratio_params = optax.apply_updates(state.ratio_params, r_upd)
        
        def _get_baseline(s_):
            return jax.lax.stop_gradient(
                self.ratio_network.apply(new_ratio_params, s_))

        # 2) Full triple critic
        def _critic_step(carry, _):
            cp, sn, opt, key = carry
            k1, k2 = jax.random.split(key)
            rho_s, rho_a = self._sample_combined(k1, state.gen_params, state.discrete_params, s.shape[0], disc_temp)
            s_t, a_t, sn_t = apply_group_element(rho_s, s), apply_group_element(rho_a, a), apply_group_element(rho_s, s_next)
            
            def _loss(cp_):
                v_s = _get_baseline(s)
                f_fake, sn1 = self._cfwd(cp_, sn, s_t, a_t, sn_t)
                f_real, sn2 = self._cfwd(cp_, sn1, s, a, s_next)
                return -(jnp.mean(f_fake) - jnp.mean(f_real * v_s)), sn2
            (l, n_sn), grads = value_and_grad(_loss, has_aux=True)(cp)
            return (optax.apply_updates(cp, self._critic_tx.update(grads, opt, cp)[0]), n_sn, opt, k2), l

        (new_cp, new_sn, new_copt, _), c_losses = jax.lax.scan(_critic_step, 
            (state.critic_params, state.critic_sn, state.critic_opt, c_rng), None, length=n_critic_steps)

        # # 3) SA critic
        # def _sa_critic_step(carry, _):
        #     cp, sn, opt, key = carry
        #     k1, k2 = jax.random.split(key)
        #     rho_s, rho_a = self._sample_combined(k1, state.gen_params, state.discrete_params, s.shape[0], disc_temp)
        #     def _loss(cp_):
        #         f_fake, sn1 = self._safwd(cp_, sn, apply_group_element(rho_s, s), apply_group_element(rho_a, a))
        #         f_real, sn2 = self._safwd(cp_, sn1, s, a)
        #         return -(jnp.mean(f_fake) - jnp.mean(f_real)), sn2
        #     (l, n_sn), grads = value_and_grad(_loss, has_aux=True)(cp)
        #     return (optax.apply_updates(cp, self._sa_critic_tx.update(grads, opt, cp)[0]), n_sn, opt, k2), l

        # (new_sacp, new_sasn, new_sacopt, _), _ = jax.lax.scan(_sa_critic_step, 
        #     (state.sa_critic_params, state.sa_critic_sn, state.sa_critic_opt, sa_rng), None, length=n_critic_steps)
        

        # 4) Discrete critic (train on discrete ops)
        def _disc_critic_step(carry, _):
            cp, sn, opt, key = carry
            key, ck = jax.random.split(key)
            def _loss(cp_):
                disc_vars = {'params': state.discrete_params}
                D_s, D_a = self.discrete_model.sample_random_op(
                    disc_vars, state.gen_params, ck, s.shape[0], temp=disc_temp)
                    
                s_t  = jnp.einsum('bij,bj->bi', D_s, s)
                a_t  = jnp.einsum('bij,bj->bi', D_a, a)
                sn_t = jnp.einsum('bij,bj->bi', D_s, s_next)
                v_s  = _get_baseline(s)
                f_fake, sn1 = self._cfwd(cp_, sn, s_t, a_t, sn_t)
                f_real, sn2 = self._cfwd(cp_, sn1, s, a, s_next)
                return -(jnp.mean(f_fake) - jnp.mean(f_real * v_s)), sn2
            (c_loss, new_sn), c_grads = value_and_grad(_loss, has_aux=True)(cp)
            upd, new_opt = self._critic_tx.update(c_grads, opt, cp)
            return (optax.apply_updates(cp, upd), new_sn, new_opt, key), c_loss

        rng, dc_rng = jax.random.split(rng)
        (new_cp, new_sn, new_copt, _), dc_losses = jax.lax.scan(
            _disc_critic_step,
            (new_cp, new_sn, new_copt, dc_rng),
            None, length=n_critic_steps)

        # Update target critic (EMA)
        new_target_cp = self._update_target_critic(state.critic_target_params, new_cp)  # NEW

        # 5) L_s update
        # (ls_loss, ls_metrics), ls_grads = value_and_grad(self._ls_loss, has_aux=True)(
        #     state.gen_params, new_cp, new_sn, new_ratio_params, s, a, s_next, ls_rng)
        # (la_loss, la_metrics), ls_grads = value_and_grad(self._la_loss, has_aux=True)(
        #     state.gen_params, new_target_cp, new_sn, new_ratio_params, s, a, s_next, ls_rng)
        # la_upd, new_la_opt = self._ls_tx.update(
        #     ls_grads, state.la_opt, state.gen_params)
        # new_gp = optax.apply_updates(state.gen_params, la_upd)
        (la_loss, la_metrics), la_grads = value_and_grad(self._la_loss, has_aux=True)(
            state.gen_params, new_target_cp, new_sn, new_ratio_params, s, a, s_next, ls_rng)
        la_upd, new_la_opt = self._la_tx.update(la_grads, state.la_opt, state.gen_params)
        new_gp = optax.apply_updates(state.gen_params, la_upd)

        # 6) W update
        # # (w_loss, w_metrics), w_grads = value_and_grad(self._w_loss, has_aux=True)(
        # #     new_gp, state.discrete_params, new_sacp, new_sasn, s, a, s_next, w_rng, disc_temp)
        # (w_loss, w_metrics), w_grads = value_and_grad(self._w_loss, has_aux=True)(
        #     new_gp, state.discrete_params, new_target_cp, new_sn, new_ratio_params, s, a, s_next, w_rng, disc_temp)
        # w_upd, new_w_opt = self._w_tx.update(
        #     w_grads, state.w_opt, new_gp)
        # new_gp = optax.apply_updates(new_gp, w_upd)
        # # new_gp = _renorm_Ls_in_vars(new_gp)
        # new_gp = _renorm_La_in_vars(new_gp)  # renorm L_a to prevent explosion after W update

        # 6) L_s + W update (formerly _w_loss)
        (ls_loss, ls_metrics), ls_grads = value_and_grad(self._ls_loss, has_aux=True)(
            new_gp, state.discrete_params, new_target_cp, new_sn, new_ratio_params, s, a, s_next, w_rng, disc_temp)
        ls_upd, new_lsw_opt = self._lsw_tx.update(ls_grads, state.w_opt, new_gp)
        new_gp = optax.apply_updates(new_gp, ls_upd)
        new_gp = _renorm_Ls_in_vars(new_gp)  # keep L_s stable
        new_gp = _renorm_La_in_vars(new_gp)  # optional


        # 7) Discrete ops update (D_a learned directly, D_s derived!)
        def discrete_diversity_loss(disc_params, temp, rng):
            """Penalize discrete ops that are too similar."""
            out = self.discrete_model.apply({'params': disc_params})
            
            if out['mode'] == '1d':
                n_ops = out['sign_logits_a'].shape[0]
            else:
                n_ops = out['perm_logits_a'].shape[0]
            
            if n_ops <= 1:
                return 0.0
            
            ops_s = []
            ops_a = []
            for i in range(n_ops):
                key_i = jax.random.fold_in(rng, i)
                D_s, D_a = self.discrete_model.get_transformation(
                    {'params': disc_params}, new_gp, i, key_i, temp, hard=False
                )
                def _sanitize(x):
                    x = jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x)
                    return jnp.clip(x, -1e3, 1e3)

                ops_s.append(_sanitize(D_s))
                ops_a.append(_sanitize(D_a))
            
            D_s_all = jnp.stack(ops_s)
            D_a_all = jnp.stack(ops_a)
            
            def pair_loss(i, j):
                diff_s = D_s_all[i] - D_s_all[j]
                diff_a = D_a_all[i] - D_a_all[j]
                dist_s = jnp.linalg.norm(diff_s, ord='fro')
                dist_a = jnp.linalg.norm(diff_a, ord='fro')
                min_dist = 2.0
                ls = jnp.maximum(min_dist - dist_s, 0.0) ** 2
                la = jnp.maximum(min_dist - dist_a, 0.0) ** 2
                return ls + la

            idx_i, idx_j = jnp.triu_indices(n_ops, k=1)
            pair_losses = jax.vmap(lambda i, j: pair_loss(i, j))(idx_i, idx_j)
            denom = n_ops * (n_ops - 1)
            return jnp.sum(pair_losses) / denom

        def _disc_gen_loss_fn(dp):
            out = self.discrete_model.apply({'params': dp})
            if out['mode'] == '1d':
                n_ops = out['sign_logits_a'].shape[0]
            else:
                n_ops = out['perm_logits_a'].shape[0]

            # get current safe W
            W_live = self.gen_model.apply(new_gp)['W']

            total_l, total_r, total_int = 0.0, 0.0, 0.0
            for i in range(n_ops):
                k_i = jax.random.fold_in(d_rng, i)
                D_s, D_a = self.discrete_model.get_intertwined_op(
                    {'params': dp}, new_gp, i, k_i, temp=disc_temp
                )

                # IPM part (unchanged)
                s_t  = jnp.einsum('ij,bj->bi', D_s, s)
                a_t  = jnp.einsum('ij,bj->bi', D_a, a)
                sn_t = jnp.einsum('ij,bj->bi', D_s, s_next)

                def _sanitize(x):
                    x = jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x)
                    return jnp.clip(x, -1e3, 1e3)

                s_t, a_t, sn_t = _sanitize(s_t), _sanitize(a_t), _sanitize(sn_t)

                f_fake = self._cfwd_frozen(new_target_cp, new_sn, s_t, a_t, sn_t)
                f_real = self._cfwd_frozen(new_target_cp, new_sn, s, a, s_next)

                v_s = _get_baseline(s)
                v_t = self.ratio_network.apply(jax.lax.stop_gradient(new_ratio_params), s_t)

                total_l += jnp.mean(f_fake) - jnp.mean(f_real * v_s)
                total_r += jnp.mean((v_t - 1.0) ** 2)

                # NEW: discrete intertwiner penalty
                resid = D_a @ W_live - W_live @ D_s
                total_int += jnp.mean(resid ** 2)

            ipm_loss = (total_l + total_r) / n_ops
            int_loss = total_int / n_ops

            div_loss = discrete_diversity_loss(dp, disc_temp, d_rng)
            lambda_diversity = float(self.config.get('lambda_discrete_diversity', 0.1))
            lambda_int = float(self.config.get('lambda_discrete_intertwiner', 0.1))

            loss = ipm_loss + lambda_diversity * div_loss + lambda_int * int_loss
            return loss, {
                'loss/L_f_disc': total_l / n_ops,
                'loss/R_ratio_disc': total_r / n_ops,
                'reg/discrete_diversity': div_loss,
                'reg/discrete_intertwiner': int_loss,
            }

        (d_loss, d_metrics), d_grads = value_and_grad(_disc_gen_loss_fn, has_aux=True)(state.discrete_params)

        d_upd, new_disc_opt = self.disc_tx.update(
            d_grads, state.discrete_opt, state.discrete_params)
        new_disc_params = optax.apply_updates(state.discrete_params, d_upd)

        new_state = DiscoveryState(
            gen_params              = new_gp,
            la_opt                  = new_la_opt,
            w_opt                   = new_lsw_opt,
            critic_params           = new_cp,
            critic_sn               = new_sn,
            critic_opt              = new_copt,
            # sa_critic_params = new_sacp,
            # sa_critic_sn     = new_sasn,
            # sa_critic_opt    = new_sacopt,
            ratio_params            = new_ratio_params,
            ratio_opt               = new_ratio_opt,
            rng                     = rng,
            discrete_params         = new_disc_params,
            discrete_opt            = new_disc_opt,
            enc_params              = new_enc_params,
            dec_params              = new_dec_params,
            enc_opt                 = new_enc_opt,
            dec_opt                 = new_dec_opt,
            critic_target_params    = new_target_cp,  # NEW
        )
        metrics = {
            'loss/ratio':        r_loss,
            'loss/critic':       jnp.mean(c_losses),
            'loss/critic_disc':  jnp.mean(dc_losses),
            # 'loss/L_w':          w_metrics['loss/L_w'],
            **la_metrics,
            **d_metrics,
            **ls_metrics,      
            **enc_metrics,
            }
        return new_state, metrics
    
    def train_step(self, batch, step: int) -> Dict:
        """Main training step."""
        temp = discrete_temp_schedule(step, self.config)

        self.state, metrics = self._update(
            self.state,
            batch.observations,
            batch.actions,
            batch.next_observations,
            int(self.config.get('n_critic_steps', 5)),
            float(temp),
            self.use_encoder,
        )
        metrics['disc/temperature'] = temp
        return {k: float(v) for k, v in metrics.items()}

    def get_learned(self):
        out = self.gen_model.apply(self.state.gen_params)
        return out['L_s_normed'], out['W'], out['L_a_normed']

    def get_discrete_ops(
        self,
        hard: bool = True,
        temp: float | None = None,
        rng: jax.Array | None = None,
    ):
        """Materialize all discrete ops as concrete matrices."""
        if temp is None:
            temp = float(self.config.get("disc_temp_end", 0.1))
        if rng is None:
            rng = self.state.rng

        out = self.discrete_model.apply({'params': self.state.discrete_params})
        if out['mode'] == '1d':
            n_ops = out['sign_logits_a'].shape[0]
        else:
            n_ops = out['perm_logits_a'].shape[0]

        vars_ = {"params": self.state.discrete_params}

        ops = []
        for op_idx in range(n_ops):
            key = jax.random.fold_in(rng, op_idx)
            D_s, D_a = self.discrete_model.get_transformation(
                vars_, self.state.gen_params, op_idx=op_idx, rng=key, temp=temp, hard=hard
            )
            ops.append(
                {
                    "D_s": np.array(D_s),
                    "D_a": np.array(D_a),
                }
            )
        return ops

@jit
def _renorm_Ls_in_vars(gen_vars):
    inner = gen_vars['params']
    return {'params': {**inner, 'L_s': _normalize_generators_fro(inner['L_s'])}}

@jit
def _renorm_La_in_vars(gen_vars):
    inner = gen_vars['params']
    return {'params': {**inner, 'L_a': _normalize_generators_fro(inner['L_a'])}}



# -----------------------------------------------------------------------
# Improved logging
# -----------------------------------------------------------------------

def log_all_plots(
    learner,
    step: int,
    batch,
    summary_writer,
    use_wandb: bool = False,
    plot_interval: int = 10_000,
):
    """
    Log generator heatmaps, scatter, and discrete op heatmaps.
    """
    if step % plot_interval != 0:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    def fig_to_pil(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def pil_to_tb(img):
        arr = np.array(img).astype(np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None]

    # ---- continuous generators ----
    L_s, W, L_a = learner.get_learned()
    L_s_np = np.array(L_s)
    W_np   = np.array(W)
    L_a_np = np.array(L_a)
    K      = L_s_np.shape[0]

    fig, axes = plt.subplots(K, 3, figsize=(12, 3.5 * K), squeeze=False)
    fig.suptitle(f'Learned generators — step {step}')
    vmax = max(np.abs(L_s_np).max(), np.abs(L_a_np).max(), 1e-8)
    for k in range(K):
        for ax, mat, title in zip(
            axes[k],
            [L_s_np[k], W_np, L_a_np[k]],
            [f'$L_s^{k}$ ({L_s_np.shape[-1]}×{L_s_np.shape[-1]})',
             f'$W$ ({W_np.shape[0]}×{W_np.shape[1]})',
             f'$L_a^{k}$ ({L_a_np.shape[-1]}×{L_a_np.shape[-1]})']):
            vm = max(np.abs(mat).max(), 1e-8)
            im = ax.imshow(mat, cmap='RdBu', vmin=-vm, vmax=vm)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    gen_img = fig_to_pil(fig)
    plt.close(fig)

    # ---- state scatter ----
    s     = batch.observations
    plot_rng = jax.random.PRNGKey(int(step))
    disc_temp = float(discrete_temp_schedule(step, learner.config))

    rho_s, _ = learner._sample_combined(
        plot_rng , learner.state.gen_params, learner.state.discrete_params,  s.shape[0], disc_temp)

    s_t   = np.array(jnp.einsum('bij,bj->bi', rho_s, s))
    s_np  = np.array(s)
    n_plt = min(200, s_np.shape[0])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(s_np[:n_plt,0], s_np[:n_plt,1], s=8, alpha=0.4,
               label='original', color='steelblue')
    ax.scatter(s_t[:n_plt,0],  s_t[:n_plt,1],  s=8, alpha=0.4,
               label='transformed', color='tomato')
    for i in range(min(200, n_plt)):
        ax.annotate('', xy=(s_t[i,0], s_t[i,1]),
                    xytext=(s_np[i,0], s_np[i,1]),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))
    ax.set_title(f'State transform — step {step}')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    scatter_img = fig_to_pil(fig)
    plt.close(fig)

    # ---- discrete ops ----
    n_ops   = int(learner.config.get('n_discrete_ops', 2))
    temp    = discrete_temp_schedule(step, learner.config)
    ops     = learner.get_discrete_ops(hard=True, temp=temp)
    fig, axes = plt.subplots(n_ops, 2,
                             figsize=(8, 3.5 * n_ops), squeeze=False)
    fig.suptitle(f'Discrete ops (hard) — step {step}')
    for i, op in enumerate(ops):
        for ax, mat, title in zip(
            axes[i],
            [op['D_s'], op['D_a']],
            [f'op {i}: $D_s$', f'op {i}: $D_a$']):
            im = ax.imshow(mat, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    disc_img = fig_to_pil(fig)
    plt.close(fig)

    # ---- v_psi on batch ----
    v_vals = np.array(learner.ratio_network.apply(
        learner.state.ratio_params, s))
    
    # Sanitize and handle near-degenerate ranges
    v_vals = v_vals[~np.isnan(v_vals)]
    v_vals = v_vals[np.isfinite(v_vals)]
    if v_vals.size == 0:
        v_vals = np.array([0.0], dtype=np.float32)

    v_min, v_max = float(v_vals.min()), float(v_vals.max())
    eps_range = 1e-6
    if v_max - v_min < eps_range:
        # All values (almost) identical: plot a single bar at that value
        ax.hist(v_vals, bins=1, color='steelblue', alpha=0.7)
    else:
        ax.hist(v_vals, bins=40, color='steelblue', alpha=0.7)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axvline(1.0, color='tomato', lw=1.5, label='v=1')
    ax.set_xlabel('v(s)')
    ax.set_title(f'Ratio network output — step {step}')
    # set log scale if values are very spread out
    if v_vals.max() / (v_vals.min() + 1e-8) > 100:
        ax.set_yscale('log')
    ax.legend()
    ratio_img = fig_to_pil(fig)
    plt.close(fig)

    # ---- log to TB + wandb ----
    for key, img in [
        ('generators/heatmap',  gen_img),
        ('generators/scatter',  scatter_img),
        ('generators/discrete', disc_img),
        ('ratio/histogram',     ratio_img),
    ]:
        summary_writer.add_images(key, pil_to_tb(img), step)

    if use_wandb:
        import wandb
        wandb.log({
            'generators/heatmap':  wandb.Image(gen_img),
            'generators/scatter':  wandb.Image(scatter_img),
            'generators/discrete': wandb.Image(disc_img),
            'ratio/histogram':     wandb.Image(ratio_img),
        }, step=step)

    # ---- scalar diagnostics ----
    out  = learner.gen_model.apply(learner.state.gen_params)
    W    = np.array(out['W'])
    L_s  = np.array(out['L_s_normed'])
    L_a  = np.array(out['L_a_normed'])
    sv_W = np.linalg.svd(W, compute_uv=False)

    scalars = {
        'gen/W_min_sv':    float(sv_W.min()),
        'gen/W_condition': float(sv_W.max() / (sv_W.min() + 1e-8)),
        'disc/temperature': temp,
    }
    for k in range(L_s.shape[0]):
        scalars[f'gen/L_s_{k}_fro'] = float(np.linalg.norm(L_s[k], 'fro'))
        scalars[f'gen/L_a_{k}_fro'] = float(np.linalg.norm(L_a[k], 'fro'))
        resid = W @ L_s[k] - L_a[k] @ W
        scalars[f'gen/intertwine_residual_{k}'] = float(np.linalg.norm(resid, 'fro'))

    for k, v in scalars.items():
        summary_writer.add_scalar(k, v, step)
    if use_wandb:
        import wandb
        wandb.log(scalars, step=step)

