"""KNN-based TMT symmetry discovery (TMT-KNN).

Drop-in alternative to TMT-NN (score/denoiser pipeline).  Instead of training a
sigma-conditioned denoiser to learn the transition manifold, we fit a k-nearest-
neighbour tangent-space projector:

  1. Standardise z = [s; a; s'] coordinate-wise.
  2. For a set of reference points, compute the local normal frame via PCA of
     their k-nearest neighbours.
  3. Build a KNNManifoldProjector (equinox module) that, given a query z, finds
     the nearest reference and projects onto its local tangent plane.
  4. Because the Jacobian of this projection is I - N_i N_i^T, the existing
     `_compute_normal_frames` (jacrev path) automatically recovers the KNN
     normal frame — no bespoke frame-extraction code needed.
  5. Continuous and discrete symmetry search are identical to TMT-NN 

Public API mirrors discover_score_symmetries:
    result = discover_knn_symmetries(states, actions, next_states,
                                     episode_ids, step_ids,
                                     state_dim, action_dim, config=...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.networks.score_symmetry_discovery import (
    ContinuousDiscoveryConfig,
    DiscreteDiscoveryConfig,
    GroupAssemblyConfig,
    _assemble_group,
    _batched_manifold_rmse,
    _build_splits,
    _check_policy_symmetry,
    _ctilde,
    _empty_continuous_result,
    _empty_discrete_result,
    _merge_discrete_results,
    _to_numpy,
    discover_continuous_symmetries,
    MultiDiscreteSymmetryDiscovery,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KNNTangentConfig:
    """Hyperparameters for the KNN tangent-space estimator."""
    k: int = 20
    num_ref_points: int = 2000
    validation_split: float = 0.1
    holdout_split: float = 0.1
    seed: int = 0
    verbose: bool = False
    standardize_eps: float = 1e-6
    normalize_data: bool = True
    # Proxy sigma passed to downstream code that expects a training config.
    # The KNN projector ignores sigma; this value is only used to initialise
    # manifold_distance_tau in the discrete search (5 * train_rmse).
    sigma_eval: float = 0.0
    # If non-empty, build one projector per k value and average trace-normalised M matrices.
    # Analogous to multiscale_sigmas in TMT-NN. Empty list = single-k mode (use k above).
    multiscale_k: list = field(default_factory=list)


@dataclass
class KNNSymmetryDiscoveryConfig:
    tangent: KNNTangentConfig = field(default_factory=KNNTangentConfig)
    continuous: ContinuousDiscoveryConfig = field(default_factory=ContinuousDiscoveryConfig)
    discrete: DiscreteDiscoveryConfig = field(default_factory=DiscreteDiscoveryConfig)
    group: GroupAssemblyConfig = field(default_factory=GroupAssemblyConfig)


# ---------------------------------------------------------------------------
# KNN manifold projector (equinox module — JAX-compatible drop-in for denoiser)
# ---------------------------------------------------------------------------

class KNNManifoldProjector(eqx.Module):
    """Projects a query z onto the local tangent plane of the nearest reference point.

    The projection is  proj(z) = z - N_i N_i^T (z - z_ref_i)  where i is the
    index of the nearest reference point and N_i is its normal frame.

    Jacobian w.r.t. z (treating i as constant) = I - N_i N_i^T = tangent projector.

    The `sigma` argument is accepted for API compatibility with the denoiser
    but is ignored.
    """
    z_ref: jnp.ndarray        # (M, d)
    normal_bases: jnp.ndarray  # (M, d, n_normals)
    data_dim: int = eqx.field(static=True)

    def __call__(self, gz, sigma=None):
        single = gz.ndim == 1
        if single:
            gz = gz[None]

        z_ref = self.z_ref
        nb = self.normal_bases

        def _proj_one(y):
            # Memory-safe NN search via lax.scan — avoids the O(N×M×d) intermediate
            # that jnp.argmin(vmap over N) would materialise, which OOMs at M=16k.
            # Memory cost: O(N × d × n_normals) instead of O(N × M × d).
            def scan_fn(carry, refs):
                best_dist, best_z_r, best_n = carry
                z_r, n = refs
                dist = jnp.sum((y - z_r) ** 2)
                better = dist < best_dist
                return (jnp.where(better, dist, best_dist),
                        jnp.where(better, z_r, best_z_r),
                        jnp.where(better, n, best_n)), None

            (_, z_nn, n_nn), _ = jax.lax.scan(
                scan_fn,
                (jnp.array(jnp.inf), z_ref[0], nb[0]),
                (z_ref, nb),
            )
            delta = y - z_nn
            normal_comp = n_nn.T @ delta    # (n_normals,)
            return y - n_nn @ normal_comp   # (d,)

        out = jax.vmap(_proj_one)(gz)
        return out[0] if single else out

    def get_normal_frames(self, z_points: np.ndarray) -> np.ndarray:
        """Return precomputed normal frames for each query point via brute-force NN.

        Bypasses jacrev entirely — avoids XLA compilation of dynamic-slice
        gradients over the large reference array, which hangs for hundreds of
        reference points.

        Returns
        -------
        frames : np.ndarray  shape (N, d, n_normals)
            Normal frame (already orthonormal from PCA/QR) for each query.
        """
        z_ref_np = np.asarray(self.z_ref)      # (M, d)
        nb_np = np.asarray(self.normal_bases)  # (M, d, n_normals)
        z_np = np.asarray(z_points, dtype=np.float32)  # (N, d)

        # Compute nearest-neighbor indices in query-point batches to avoid
        # allocating an (N, M, d) intermediate (can be multi-GB for large N, M, d).
        batch = 256
        nn_idx = np.empty(z_np.shape[0], dtype=np.int64)
        for s in range(0, z_np.shape[0], batch):
            chunk = z_np[s:s + batch]                               # (b, d)
            diff = chunk[:, None, :] - z_ref_np[None, :, :]        # (b, M, d)
            nn_idx[s:s + batch] = np.argmin(np.sum(diff ** 2, axis=-1), axis=-1)
        return nb_np[nn_idx]                             # (N, d, n_normals)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_knn_estimator(
    z_train: np.ndarray,
    z_val: np.ndarray,
    z_holdout: np.ndarray,
    state_dim: int,
    action_dim: int,
    config: KNNTangentConfig,
) -> tuple[KNNManifoldProjector, dict]:
    """Build KNNManifoldProjector from training data.

    Returns
    -------
    projector : KNNManifoldProjector
    history   : dict  — mirrors the training_history dict of the score pipeline
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError("scikit-learn is required for TMT-KNN") from e

    data_dim = int(z_train.shape[1])
    n_normals = int(state_dim)          # normal-space dim = state_dim
    tangent_dim = data_dim - n_normals  # = state_dim + action_dim

    # --- Select reference points ---
    num_ref = min(int(config.num_ref_points), int(z_train.shape[0]))
    rng = np.random.default_rng(int(config.seed))
    ref_idx = rng.choice(z_train.shape[0], size=num_ref, replace=False)
    z_ref = z_train[ref_idx].astype(np.float32)

    if config.verbose:
        print(
            f"[tmt-knn] k={config.k}, num_ref={num_ref}, "
            f"data_dim={data_dim}, n_normals={n_normals}"
        )

    # --- Fit KNN on full training set ---
    nn_model = NearestNeighbors(n_neighbors=config.k + 1, algorithm="auto", n_jobs=-1)
    nn_model.fit(z_train)

    # --- Compute normal frames at reference points ---
    normal_bases = np.zeros((num_ref, data_dim, n_normals), dtype=np.float32)
    proj_errors_sq = []  # squared projection residuals (on neighbours)

    for idx in range(num_ref):
        z_i = z_ref[idx]
        nbr_raw = nn_model.kneighbors(
            [z_i], n_neighbors=config.k + 1, return_distance=False
        )[0]
        # Exclude exact self-matches to avoid zero vectors in centred matrix.
        nbr_idx = [j for j in nbr_raw if not np.allclose(z_train[j], z_i, atol=1e-6)][
            : config.k
        ]
        if len(nbr_idx) == 0:
            nbr_idx = nbr_raw[: config.k].tolist()

        neighbors = z_train[nbr_idx].astype(np.float64)  # (k, d)
        centered = neighbors - z_i.astype(np.float64)    # (k, d)

        # SVD: rows of Vt are right singular vectors.
        # Tangent space  = top tangent_dim right singular vectors.
        # Normal space   = bottom n_normals right singular vectors.
        _, _, Vt = np.linalg.svd(centered, full_matrices=True)
        normal_frame_raw = Vt[tangent_dim:].T.astype(np.float32)  # (d, n_normals)
        q, _ = np.linalg.qr(normal_frame_raw)
        normal_bases[idx] = q[:, :n_normals]

        # Residual of neighbours projected onto the fitted normal frame
        # (= distance of each neighbour from the local tangent plane).
        for nbr in neighbors[:5]:
            delta = (nbr - z_i).astype(np.float32)
            nc = normal_bases[idx].T @ delta
            proj_errors_sq.append(float(np.sum(nc ** 2)))

    # --- Build equinox module ---
    projector = KNNManifoldProjector(
        z_ref=jnp.asarray(z_ref),
        normal_bases=jnp.asarray(normal_bases),
        data_dim=data_dim,
    )

    # --- Compute manifold RMSE diagnostics ---
    train_rmse = float(np.sqrt(np.mean(proj_errors_sq))) if proj_errors_sq else 0.0

    holdout_rmse = 0.0
    if z_holdout.shape[0] > 0:
        holdout_rmse = float(
            _batched_manifold_rmse(projector, jnp.asarray(z_holdout), 0.0)
        )

    val_rmse = 0.0
    if z_val.shape[0] > 0:
        val_rmse = float(
            _batched_manifold_rmse(projector, jnp.asarray(z_val), 0.0)
        )

    if config.verbose:
        print(
            f"[tmt-knn] train_proj_rmse={train_rmse:.4e} "
            f"val_rmse={val_rmse:.4e} holdout_rmse={holdout_rmse:.4e}"
        )

    history = {
        "train_total_loss": [],   # no training loop
        "val_total_loss": [],
        "diagnostics": {
            "full_manifold_rmse": train_rmse,
            "train_manifold_rmse": train_rmse,
            "holdout_manifold_rmse": holdout_rmse,
            "sigma_eval": 0.0,
        },
    }
    return projector, history


# ---------------------------------------------------------------------------
# Multi-scale M computation for KNN
# ---------------------------------------------------------------------------

def _compute_knn_M(
    projector: KNNManifoldProjector,
    z_subset: np.ndarray,
    state_dim: int,
    action_dim: int,
    affine: bool = False,
) -> np.ndarray:
    """Compute the symmetry constraint matrix M for a single KNN projector.

    Uses the raw_P formula: A_i = (N_i N_i^T) @ C_i, M = mean(A_i^T A_i).
    This matches what discover_continuous_symmetries does for the raw_P variant
    but avoids going through jacrev by using the projector's precomputed normal frames.
    """
    operator_dim = state_dim ** 2 + action_dim ** 2 + (state_dim + action_dim if affine else 0)
    n_frames = z_subset.shape[0]
    frames = projector.get_normal_frames(z_subset)  # (N, d, n_normals)
    M = np.zeros((operator_dim, operator_dim), dtype=np.float64)
    for i in range(n_frames):
        N_i = frames[i].astype(np.float64)           # (d, n_normals)
        normal_proj = N_i @ N_i.T                    # (d, d)  =  I - P_i
        C_i = _ctilde(z_subset[i], state_dim, action_dim, affine).astype(np.float64)
        A_i = normal_proj @ C_i                      # (d, operator_dim)
        M += A_i.T @ A_i
    if n_frames > 0:
        M /= float(n_frames)
    return M


# ---------------------------------------------------------------------------
# Proxy training-config dataclass (satisfies discover_continuous_symmetries API)
# ---------------------------------------------------------------------------

@dataclass
class _KNNTrainingCfgProxy:
    """Minimal stand-in for ScoreTrainingConfig passed to discover_continuous_symmetries."""
    sigma_min: float = 1.0
    sigma_max: float = 1.0
    seed: int = 0
    sigma_eval: float = 0.0  # not used by KNN projector


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def discover_knn_symmetries(
    states,
    actions,
    next_states,
    episode_ids,
    step_ids,
    state_dim: int,
    action_dim: int,
    device=None,
    config: Optional[KNNSymmetryDiscoveryConfig] = None,
    verbose: bool = False,
    discrete_log_interval: int = 250,
    only_discrete: bool = False,
    only_continuous: bool = False,
):
    """TMT-KNN symmetry discovery.

    Mirrors the interface of discover_score_symmetries.  The only difference is
    that the denoiser is replaced by a KNN tangent-space projector.  Continuous
    and discrete discovery, group assembly, and policy symmetry checks are
    identical.

    Returns the same result dict structure so that logging/saving helpers can
    be shared across TMT-NN and TMT-KNN.
    """
    del device, episode_ids, step_ids
    cfg = config or KNNSymmetryDiscoveryConfig()
    cfg.discrete.log_interval = int(discrete_log_interval)

    states_np = _to_numpy(states).astype(np.float32)
    actions_np = _to_numpy(actions).astype(np.float32)
    next_states_np = _to_numpy(next_states).astype(np.float32)
    z = np.concatenate([states_np, actions_np, next_states_np], axis=-1).astype(np.float32)

    # --- Standardise ---
    initial_splits = _build_splits(
        num_points=int(z.shape[0]),
        validation_split=float(cfg.tangent.validation_split),
        holdout_split=float(cfg.tangent.holdout_split),
        seed=int(cfg.tangent.seed),
    )
    train_idx = initial_splits["train_idx"]
    z_train_raw = z[train_idx]
    z_mean = z_train_raw.mean(axis=0)
    z_std = z_train_raw.std(axis=0)
    if cfg.tangent.normalize_data:
        is_const  = z_std < float(cfg.tangent.standardize_eps)
        z_scale   = np.where(is_const, 1.0, z_std).astype(np.float32)
        z_shift   = np.where(is_const, 0.0, z_mean).astype(np.float32)
    else:
        z_scale = np.ones_like(z_mean, dtype=np.float32)
        z_shift = np.zeros_like(z_mean, dtype=np.float32)
    z_std_all = ((z - z_shift) / z_scale).astype(np.float32)

    z_train_std = z_std_all[initial_splits["train_idx"]]
    z_val_std = z_std_all[initial_splits["val_idx"]]
    z_holdout_std = z_std_all[initial_splits["holdout_idx"]]

    # --- Fit KNN projector ---
    if verbose:
        print("[phase] tmt-knn: fitting KNN tangent estimator")
    projector, knn_history = _fit_knn_estimator(
        z_train=z_train_std,
        z_val=z_val_std,
        z_holdout=z_holdout_std,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        config=cfg.tangent,
    )
    # Attach split info so downstream code (discrete search tau calibration) works.
    knn_history["split"] = initial_splits

    # Combine val + holdout for a larger discrete eval set (same as TMT-NN).
    z_eval_std = (
        np.concatenate([z_val_std, z_holdout_std], axis=0)
        if z_val_std.shape[0] > 0
        else z_holdout_std
    )

    # --- Continuous symmetry discovery ---
    if verbose:
        diag = knn_history["diagnostics"]
        print(
            f"[phase] knn projector fitted "
            f"train_proj_rmse={diag['train_manifold_rmse']:.4e} "
            f"holdout_rmse={diag['holdout_manifold_rmse']:.4e}"
        )
        print("[phase] continuous knn symmetry discovery start")

    training_cfg_proxy = _KNNTrainingCfgProxy(seed=int(cfg.tangent.seed))
    # Ensure sigma_eval is explicitly set so discover_continuous_symmetries does
    # not try to derive it from sigma_min/sigma_max (irrelevant for KNN).
    cfg.continuous.sigma_eval = 0.0

    # Anchor-centric M assembly: evaluate both N_j and ctilde at the same ref
    # points rather than sampling a separate (larger) set of frame points whose
    # normal frames are then borrowed from whichever ref is nearest.
    z_anchor_std = np.asarray(projector.z_ref)          # already standardised
    cfg.continuous.num_frame_points = z_anchor_std.shape[0]  # use every anchor

    # Multi-scale M: build projectors for each k in multiscale_k, compute M for
    # each, trace-normalise, average — analogous to multiscale_sigmas in TMT-NN.
    multiscale_ks = list(getattr(cfg.tangent, "multiscale_k", []))
    precomputed_M = None
    if len(multiscale_ks) > 1:
        affine = bool(getattr(cfg.continuous, "affine", False))
        operator_dim = (state_dim ** 2 + action_dim ** 2
                        + (state_dim + action_dim if affine else 0))
        M_acc = np.zeros((operator_dim, operator_dim), dtype=np.float64)
        if verbose:
            print(f"[tmt-knn] multiscale M: k_list={multiscale_ks}")
        for k_i in multiscale_ks:
            cfg_k = KNNTangentConfig(
                k=k_i,
                num_ref_points=cfg.tangent.num_ref_points,
                validation_split=cfg.tangent.validation_split,
                holdout_split=cfg.tangent.holdout_split,
                seed=cfg.tangent.seed,
                verbose=cfg.tangent.verbose,
                standardize_eps=cfg.tangent.standardize_eps,
                sigma_eval=cfg.tangent.sigma_eval,
            )
            proj_k, _ = _fit_knn_estimator(
                z_train=z_train_std,
                z_val=z_val_std,
                z_holdout=z_holdout_std,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                config=cfg_k,
            )
            M_k = _compute_knn_M(proj_k, np.asarray(proj_k.z_ref), state_dim, action_dim, affine)
            tr = float(np.trace(M_k))
            if tr > 1e-30:
                M_k /= tr
            M_acc += M_k
            if verbose:
                print(f"[tmt-knn] k={k_i}: trace(M)={tr:.3e}")
        precomputed_M = M_acc / len(multiscale_ks)

    if only_discrete:
        if verbose:
            print("[phase] continuous knn symmetry discovery SKIPPED (only_discrete=True)")
        continuous = _empty_continuous_result(int(state_dim), int(action_dim))
    else:
        continuous = discover_continuous_symmetries(
            model=projector,
            z_train_std=z_anchor_std,
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            training_cfg=training_cfg_proxy,
            config=cfg.continuous,
            verbose=verbose,
            precomputed_M=precomputed_M,
        )

    if only_continuous:
        if verbose:
            print("[phase] discrete knn symmetry discovery SKIPPED (only_continuous=True)")
        discrete = _empty_discrete_result(int(state_dim), int(action_dim))
    else:
        n_candidates = int(cfg.discrete.num_candidates)
        cand_batch = int(cfg.discrete.candidates_batch_size)
        if cand_batch <= 0 or cand_batch >= n_candidates:
            if verbose:
                print("[phase] discrete knn symmetry discovery start")
            discrete = MultiDiscreteSymmetryDiscovery(
                model=projector,
                z_train_std=z_train_std,
                z_holdout_std=z_holdout_std,
                z_val_std=z_val_std,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                sigma_eval=0.0,
                config=cfg.discrete,
                training_history=knn_history,
            ).fit(verbose=verbose)
        else:
            n_batches = math.ceil(n_candidates / cand_batch)
            if verbose:
                print(f"[phase] discrete knn symmetry discovery start "
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
                    training_history=knn_history,
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

    # --- Policy symmetry check ---
    s_train_std = z_train_std[:, : int(state_dim)]
    a_train_std = z_train_std[:, int(state_dim): int(state_dim) + int(action_dim)]
    if verbose:
        print("[phase] policy symmetry check start")
    policy_check = _check_policy_symmetry(
        states_std=s_train_std,
        actions_std=a_train_std,
        continuous_generators=continuous["generators"],
        discrete_candidates=discrete["candidates"],
        seed=int(cfg.tangent.seed),
        verbose=verbose,
    )

    return {
        "model": projector,
        "training_history": knn_history,
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
