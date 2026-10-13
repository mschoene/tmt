"""Train a plain-MLP forward dynamics model f(s,a) -> delta_s on narrow-theta1
Reacher transitions, with an optional finite-rotation equivariance penalty
that supervises held-out (rotated) predictions using the environment's own
EXACT dynamics symmetry (verified separately, to roundoff):
    f(exp(eps*G_s).s, exp(eps*G_a).a) = exp(eps*G_s) . f(s,a)
No EMLP, no bilinear -- plain MLP throughout. eq_lambda=0 (or no G) reduces
to a plain dynamics-model fit.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from jax_rl.networks.common import MLP
from jax_rl.agents.bc.actor import _taylor_expm


class DynamicsMLP(nn.Module):
    hidden_dims: tuple
    obs_dim: int

    @nn.compact
    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True)(x)
        return nn.Dense(self.obs_dim)(x)


def load_transitions(path, val_frac=0.1):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    eps = d['episodes']
    n_val = int(len(eps) * val_frac)
    train_eps, val_eps = eps[:-n_val], eps[-n_val:]

    def flatten(episodes):
        s, a, ds = [], [], []
        for ep in episodes:
            obs = np.asarray(ep['observations'])
            act = np.asarray(ep['actions'])
            s.append(obs[:-1])
            a.append(act)
            ds.append(obs[1:] - obs[:-1])
        return (np.concatenate(s).astype(np.float32),
                np.concatenate(a).astype(np.float32),
                np.concatenate(ds).astype(np.float32))

    return flatten(train_eps), flatten(val_eps)


def _eq_violation(model_def, params, s, a, G_s, G_a, eps, ds_pred=None):
    if ds_pred is None:
        ds_pred = model_def.apply({'params': params}, s, a)

    def per_generator(k_eps, Gs_k, Ga_k):
        A_s = k_eps[:, None, None] * Gs_k[None, :, :]
        exp_s = _taylor_expm(A_s)
        s_rot = jnp.einsum('bij,bj->bi', exp_s, s)

        A_a = k_eps[:, None, None] * Ga_k[None, :, :]
        exp_a = _taylor_expm(A_a)
        a_rot = jnp.einsum('bij,bj->bi', exp_a, a)

        f_rot_pred = model_def.apply({'params': params}, s_rot, a_rot)
        f_rot_expected = jnp.einsum('bij,bj->bi', exp_s, ds_pred)

        return jnp.mean(jnp.sum((f_rot_pred - f_rot_expected) ** 2, axis=-1))

    return jnp.mean(jax.vmap(per_generator)(eps, G_s, G_a))


def build_train_step(model_def, tx, G_s, G_a, eq_lambda, eps_min, eps_max):
    def loss_fn(params, s, a, ds_true, key):
        pred = model_def.apply({'params': params}, s, a)
        reg_loss = jnp.mean(jnp.sum((pred - ds_true) ** 2, axis=-1))

        eq_loss = jnp.array(0.0)
        if G_s is not None:
            batch_size = s.shape[0]
            K = G_s.shape[0]
            eps = jax.random.uniform(key, (K, batch_size), minval=eps_min, maxval=eps_max)
            eq_loss = _eq_violation(model_def, params, s, a, G_s, G_a, eps, ds_pred=pred)

        total = reg_loss + eq_lambda * eq_loss
        return total, {'reg_loss': reg_loss, 'eq_loss': eq_loss, 'total': total}

    @jax.jit
    def train_step(params, opt_state, s, a, ds_true, key):
        (total, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, s, a, ds_true, key)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, info

    return train_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/reacher_expert.pkl')
    parser.add_argument('--eq_symmetry_file', default='')
    parser.add_argument('--eq_lambda', type=float, default=0.0)
    parser.add_argument('--eq_eps_min', type=float, default=0.0)
    parser.add_argument('--eq_eps_max', type=float, default=2 * np.pi)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dims', type=str, default='256,256')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='')
    args = parser.parse_args()

    (train_s, train_a, train_ds), (val_s, val_a, val_ds) = load_transitions(args.dataset_path)
    print(f"train N={len(train_s)}  val N={len(val_s)}")

    G_s, G_a = None, None
    if args.eq_lambda > 0 and args.eq_symmetry_file:
        meta = json.load(open(args.eq_symmetry_file))
        npz = np.load(meta['npz_path'])
        G_s, G_a = jnp.asarray(npz['L_s'].astype(np.float32)), jnp.asarray(npz['L_a'].astype(np.float32))
        print(f"[eq penalty] G_s {G_s.shape} G_a {G_a.shape} from {args.eq_symmetry_file}, "
              f"lambda={args.eq_lambda}, eps_range=[{args.eq_eps_min},{args.eq_eps_max}]")

        # Safety guard: _taylor_expm (jax_rl/agents/bc/actor.py) is a fixed
        # 30-term truncated Taylor series, documented as "comfortably
        # converged" only for ||eps*G|| well under ~7. An un-normalized
        # generator fed here can push eps_max*||G_s|| far past that envelope
        # and silently produce huge-but-finite (then eventually non-finite,
        # in downstream rollout compounding) garbage rather than an error.
        # Reject loudly at load time instead of failing silently mid-training.
        _TAYLOR_EXPM_SAFE_NORM = 7.0
        for _name, _G in [("G_s", G_s), ("G_a", G_a)]:
            _G_np = np.asarray(_G)
            for _k in range(_G_np.shape[0]):
                _norm = float(np.linalg.norm(_G_np[_k]))
                _max_arg_norm = float(args.eq_eps_max) * _norm
                if _max_arg_norm >= _TAYLOR_EXPM_SAFE_NORM:
                    raise SystemExit(
                        f"[eq penalty] REFUSING to train: {_name}[{_k}] has ||.||_F={_norm:.4f}, "
                        f"so eps_max*||{_name}||={_max_arg_norm:.2f} at eps_max={args.eq_eps_max:.4f} -- "
                        f"outside _taylor_expm's documented safe envelope (well under "
                        f"{_TAYLOR_EXPM_SAFE_NORM:.0f}). This generator is not jointly normalized; "
                        f"re-derive it via score_symmetry_discovery._normalize_generator on the "
                        f"flattened [L_s,L_a] vector before use. Refusing rather than silently "
                        f"producing an unstable downstream run.")
        print(f"[eq penalty] norm guard OK: eps_max*||G_s||={float(args.eq_eps_max) * float(np.linalg.norm(np.asarray(G_s)[0])):.4f}, "
              f"eps_max*||G_a||={float(args.eq_eps_max) * float(np.linalg.norm(np.asarray(G_a)[0])):.4f} "
              f"(both < {_TAYLOR_EXPM_SAFE_NORM:.0f})")

    hidden_dims = tuple(int(h) for h in args.hidden_dims.split(','))
    obs_dim, act_dim = train_s.shape[-1], train_a.shape[-1]

    key = jax.random.PRNGKey(args.seed)
    model_def = DynamicsMLP(hidden_dims, obs_dim)
    params = model_def.init(key, train_s[:1], train_a[:1])['params']
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    train_step = build_train_step(model_def, tx, G_s, G_a, args.eq_lambda, args.eq_eps_min, args.eq_eps_max)

    rng = np.random.default_rng(args.seed)
    n = len(train_s)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n, size=args.batch_size)
        key, subkey = jax.random.split(key)
        params, opt_state, info = train_step(
            params, opt_state, train_s[idx], train_a[idx], train_ds[idx], subkey)
        if step % 2000 == 0 or step == args.steps:
            print(f"step {step}: reg_loss={float(info['reg_loss']):.6f} eq_loss={float(info['eq_loss']):.6f}")

    # Validation metrics (narrow val split only -- never touches held-out phi)
    val_pred = model_def.apply({'params': params}, val_s, val_a)
    val_mse = float(jnp.mean(jnp.sum((val_pred - val_ds) ** 2, axis=-1)))
    print(f"VAL_MSE={val_mse:.6f}")

    if G_s is not None:
        key, subkey = jax.random.split(key)
        val_eps = jax.random.uniform(subkey, (G_s.shape[0], len(val_s)),
                                      minval=args.eq_eps_min, maxval=args.eq_eps_max)
        val_eq = float(_eq_violation(model_def, params, jnp.asarray(val_s), jnp.asarray(val_a),
                                      G_s, G_a, val_eps))
        print(f"VAL_EQ={val_eq:.6f}")

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, 'wb') as f:
            flat, tree_def = jax.tree_util.tree_flatten(params)
            pickle.dump({'params_arrays': [np.asarray(p) for p in flat], 'params_tree_def': tree_def,
                         'hidden_dims': hidden_dims, 'obs_dim': obs_dim, 'act_dim': act_dim}, f)
        print(f"Saved to {args.output_path}")


if __name__ == '__main__':
    main()
