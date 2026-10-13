"""LieGAN training driver.

Two modes (--mode flag):
  marginal    — matches p(s, a, s'):     fake = (g_s·s, g_a·a, g_s·s')
  conditional — matches p(s' | s, a):   fake = (s,     a,     g_s·s')
"""

import os
import random
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
from absl import app, flags
from tensorboardX import SummaryWriter

from dataset_loader import build_offline_arrays
from jax_rl.envs import simpleEnvs  # noqa: F401
from representations import (get_ground_truth_generators, compute_space_error, compute_orth_error,
                             compute_sin_theta_max, compute_sin_theta_avg,
                             get_ground_truth_discrete_symmetries, compute_discrete_symmetry_error)
from jax_rl.networks.liegan_symmetry_discovery import (
    LieGANConfig,
    LieGANGeneratorConfig,
    LieGANDiscriminatorConfig,
    LieGANTrainingConfig,
    discover_liegan_symmetries,
)
from jax_rl.networks.score_symmetry_discovery import DiscreteDiscoveryConfig
from jax_rl.utils import make_env

try:
    import wandb
except Exception:
    wandb = None

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "ParticleCircle2D-v0", "Environment name.")
flags.DEFINE_string('dataset_path', 'path/to/your/dataset.pkl', 'Path to dataset (.pkl) or Minari ID.')

flags.DEFINE_string("mode", "marginal", "GAN mode: 'marginal' or 'conditional'.")

flags.DEFINE_string("gen_loss_flavor", "mse", "generator loss flavor: 'mse' or 'adversarial'.")

flags.DEFINE_string("save_dir", "/path/to/logs/liegan/", "Logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_points", 100000, "Maximum number of transitions to use.")
flags.DEFINE_boolean("verbose", True, "Print phase progress.")

# Generator
flags.DEFINE_integer("K", 4, "Number of Lie algebra generators.")
flags.DEFINE_multi_integer("gen_hidden_dims", [256, 256], "Generator hidden dims (unused — W/L are direct params).")
flags.DEFINE_float("init_scale", 0.8, "Initial sampling std per generator.")
flags.DEFINE_float("min_scale", 0.2, "Minimum sampling std floor.")
flags.DEFINE_boolean("fixed_scale", False, "Sample alpha ~ Uniform(-0.5,0.5) instead of Normal*scale.")
flags.DEFINE_float("eps_pinv", 1e-3, "Regularisation for W pseudoinverse.")
flags.DEFINE_boolean("use_intertwiner", False, "Couple L_s and L_a via intertwiner W (L_s = W^+ L_a W).")
flags.DEFINE_float("mu_intertwiner", 1.0, "Weight for intertwiner penalty in discrete search.")

# Discriminator
flags.DEFINE_multi_integer("disc_hidden_dims", [256, 256], "Discriminator MLP hidden dims.")

# Training
flags.DEFINE_integer("num_steps", 50000, "Number of GAN training steps.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")
flags.DEFINE_float("gen_lr", 2e-4, "Generator learning rate.")
flags.DEFINE_float("disc_lr", 4e-4, "Discriminator learning rate.")
flags.DEFINE_float("grad_clip", 1.0, "Gradient clipping norm.")
flags.DEFINE_float("validation_split", 0.1, "Validation split fraction.")
flags.DEFINE_float("holdout_split", 0.1, "Holdout split fraction.")
flags.DEFINE_integer("log_interval", 1000, "Print every N steps.")
flags.DEFINE_integer("n_disc_steps", 2, "Discriminator updates per generator update.")

# Regularisation
flags.DEFINE_float("lambda_isometry", 0.0, "Weight for skew-symmetry regularisation.")
flags.DEFINE_float("lambda_nontrivial", 5.0, "Weight for non-triviality hinge.")
flags.DEFINE_float("nontrivial_margin", 0.05, "Minimum displacement for non-triviality.")
flags.DEFINE_float("lambda_diverse", 1.0, "Weight for diversity regularisation.")
flags.DEFINE_float("lambda_intertwine", 0.5, "Weight for intertwiner constraint.")
flags.DEFINE_float("lambda_scale_floor", 0.5, "Weight for scale floor regularisation.")
flags.DEFINE_float("lambda_scale_sparse", 0.0, "L1 weight on (scale - min_scale): pushes generators without discriminator support to the floor.")
flags.DEFINE_float("lambda_rank", 0.1, "Weight for rank regularisation (log-barrier on singular values).")

# Discrete Sinkhorn search
flags.DEFINE_integer("discrete_num_candidates", 64,   "Number of signed-permutation candidates.")
flags.DEFINE_integer("discrete_num_steps",      1000, "Sinkhorn optimisation steps.")
flags.DEFINE_float("discrete_lr",               7e-3, "Learning rate for Sinkhorn search.")
flags.DEFINE_float("discrete_tau",              0.5,  "Acceptance threshold for holdout discriminator distance.")
flags.DEFINE_integer("discrete_sinkhorn_iters", 100,  "Sinkhorn normalisation iterations.")

flags.DEFINE_boolean("lifted_dimension", False, "Append constant-1 dim to state and action for SE(n) symmetry discovery.")
flags.DEFINE_boolean("only_continuous", False, "Skip discrete discovery phase, run GAN only.")
flags.DEFINE_boolean("only_discrete", False, "Reduce GAN training to only_discrete_num_steps (discriminator still needed for discrete search).")
flags.DEFINE_integer("only_discrete_num_steps", 10000, "GAN steps when only_discrete=True (fewer suffice since we just need a working discriminator).")
flags.DEFINE_integer("discrete_candidates_batch_size", -1, "Run discrete search in batches of this many candidates to cap peak memory (-1 = all at once).")
flags.DEFINE_float("discrete_rho_acceptance", 1.15, "Accept G if disc_dist(G) < rho * disc_dist(I). Should be close to 1.0 for LieGAN (baseline≈0.5).")
flags.DEFINE_float("disc_mu_identity_repulsion", 0.1, "Weight on identity barrier in discrete Sinkhorn search.")
flags.DEFINE_float("disc_identity_action_weight", 0.0, "Action-space weight in identity barrier (0=state-only; 1=joint state+action barrier).")

flags.DEFINE_boolean("normalize_data", True, "Standardize z=[s;a;s'] to zero mean / unit std before training.")

# W&B
flags.DEFINE_boolean("use_wandb", True, "Enable Weights & Biases logging.")
flags.DEFINE_string("wandb_project", "tmt", "W&B project.")
flags.DEFINE_string("wandb_entity", "", "W&B entity.")
flags.DEFINE_string("wandb_run_name", "", "Optional run name.")
flags.DEFINE_string("wandb_group", "", "Optional run group.")
flags.DEFINE_string("wandb_mode", "online", "wandb mode.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_matrix(mat, title, path):
    if plt is None:
        return
    mat = np.asarray(mat, dtype=np.float64)
    vabs = max(float(np.max(np.abs(mat))), 1e-8)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mat, cmap="RdBu", vmin=-vabs, vmax=vabs)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_curve(values, title, ylabel, path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _log_scalar(sw, tag, value, step, *, use_wandb, step_key):
    sw.add_scalar(tag, float(value), step)
    if use_wandb and wandb is not None:
        wandb.log({tag: float(value), step_key: int(step)})


def _wandb_matrix(tag, mat):
    if wandb is None or plt is None:
        return
    mat = np.asarray(mat, dtype=np.float64)
    vabs = max(float(np.max(np.abs(mat))), 1e-8)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mat, cmap="RdBu", vmin=-vabs, vmax=vabs)
    ax.set_title(tag)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)


def _wandb_vector(tag, vec):
    if wandb is None or plt is None:
        return
    vec = np.asarray(vec, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=(max(3, len(vec) * 0.4), 3))
    ax.bar(range(len(vec)), vec)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_title(tag)
    ax.set_xlabel("dim")
    fig.tight_layout()
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config():
    return LieGANConfig(
        generator=LieGANGeneratorConfig(
            K=FLAGS.K,
            init_scale=FLAGS.init_scale,
            min_scale=FLAGS.min_scale,
            eps_pinv=FLAGS.eps_pinv,
            use_intertwiner=FLAGS.use_intertwiner,
            fixed_scale=FLAGS.fixed_scale,
        ),
        discriminator=LieGANDiscriminatorConfig(
            hidden_dims=tuple(FLAGS.disc_hidden_dims),
        ),
        training=LieGANTrainingConfig(
            mode=FLAGS.mode,
            gen_loss_flavor=FLAGS.gen_loss_flavor,
            num_steps=FLAGS.only_discrete_num_steps if FLAGS.only_discrete else FLAGS.num_steps,
            normalize_data=FLAGS.normalize_data,
            batch_size=FLAGS.batch_size,
            gen_lr=FLAGS.gen_lr,
            disc_lr=FLAGS.disc_lr,
            grad_clip=FLAGS.grad_clip,
            n_disc_steps=FLAGS.n_disc_steps,
            validation_split=FLAGS.validation_split,
            holdout_split=FLAGS.holdout_split,
            seed=FLAGS.seed,
            log_interval=FLAGS.log_interval,
            lambda_isometry=FLAGS.lambda_isometry,
            lambda_nontrivial=FLAGS.lambda_nontrivial,
            nontrivial_margin=FLAGS.nontrivial_margin,
            lambda_diverse=FLAGS.lambda_diverse,
            lambda_intertwine=FLAGS.lambda_intertwine,
            lambda_scale_floor=FLAGS.lambda_scale_floor,
            lambda_scale_sparse=FLAGS.lambda_scale_sparse,
            lambda_rank=FLAGS.lambda_rank,
        ),
        discrete=DiscreteDiscoveryConfig(
            num_candidates=FLAGS.discrete_num_candidates,
            num_steps=FLAGS.discrete_num_steps,
            lr=FLAGS.discrete_lr,
            manifold_distance_tau=FLAGS.discrete_tau,
            sinkhorn_iters=FLAGS.discrete_sinkhorn_iters,
            candidates_batch_size=FLAGS.discrete_candidates_batch_size,
            use_intertwiner=FLAGS.use_intertwiner,
            mu_intertwiner=FLAGS.mu_intertwiner,
            rho_acceptance_threshold=FLAGS.discrete_rho_acceptance,
            mu_identity_repulsion=FLAGS.disc_mu_identity_repulsion,
            identity_repulsion_action_weight=FLAGS.disc_identity_action_weight,
        ),
    )


# ---------------------------------------------------------------------------
# Save / log
# ---------------------------------------------------------------------------

def _save_results(exp_dir, result):
    snap_dir = os.path.join(exp_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    gens = result["generators"]
    np.save(os.path.join(snap_dir, "z_mean.npy"),    result["standardizer"]["mean"])
    np.save(os.path.join(snap_dir, "z_scale.npy"),   result["standardizer"]["scale"])
    np.save(os.path.join(snap_dir, "W.npy"),         gens["W"])
    np.save(os.path.join(snap_dir, "scales.npy"),    gens["scales"])
    for k in range(gens["L_s"].shape[0]):
        np.save(os.path.join(snap_dir, f"L_s_{k}.npy"), gens["L_s"][k])
        np.save(os.path.join(snap_dir, f"L_a_{k}.npy"), gens["L_a"][k])
        _plot_matrix(gens["L_s"][k], f"L_s[{k}]", os.path.join(snap_dir, f"L_s_{k}.png"))
        _plot_matrix(gens["L_a"][k], f"L_a[{k}]", os.path.join(snap_dir, f"L_a_{k}.png"))
    if gens.get("b_s") is not None:
        np.save(os.path.join(snap_dir, "b_s.npy"), gens["b_s"])
    if gens.get("b_a") is not None:
        np.save(os.path.join(snap_dir, "b_a.npy"), gens["b_a"])
    _plot_matrix(gens["W"], "W (intertwiner)", os.path.join(snap_dir, "W.png"))

    h = result["training_history"]
    _plot_curve(h["loss/disc"],      "Discriminator Loss", "loss", os.path.join(snap_dir, "loss_disc.png"))
    _plot_curve(h["loss/gen_total"], "Generator Loss",     "loss", os.path.join(snap_dir, "loss_gen.png"))
    _plot_curve(h["disc/separation"],"D separation",       "val",  os.path.join(snap_dir, "disc_sep.png"))

    torch.save({
        "generators": gens,
        "standardizer": result["standardizer"],
        "data": result["data"],
        "training_history": h,
    }, os.path.join(snap_dir, "liegan_results.pt"))


def _log_results(result, sw, *, use_wandb):
    h = result["training_history"]
    for step, v in enumerate(h["loss/disc"]):
        _log_scalar(sw, "loss/disc",      v, step, use_wandb=use_wandb, step_key="step")
    for step, v in enumerate(h["loss/gen_total"]):
        _log_scalar(sw, "loss/gen_total", v, step, use_wandb=use_wandb, step_key="step")
    for step, v in enumerate(h["disc/separation"]):
        _log_scalar(sw, "disc/separation",v, step, use_wandb=use_wandb, step_key="step")
    for step, v in enumerate(h.get("reg/rank", [])):
        _log_scalar(sw, "reg/rank", v, step, use_wandb=use_wandb, step_key="step")

    # Spectral gap and closure error
    if "spectral_gap_ratio" in result:
        _log_scalar(sw, "eval/spectral_gap_ratio", result["spectral_gap_ratio"],
                    0, use_wandb=use_wandb, step_key="step")
    if "closure_error" in result:
        _log_scalar(sw, "eval/closure_error", result["closure_error"],
                    0, use_wandb=use_wandb, step_key="step")

    # Discrete search summary
    if "discrete" in result:
        disc_res = result["discrete"]
        _log_scalar(sw, "discrete/num_accepted",
                    len(disc_res["candidates"]),
                    0, use_wandb=use_wandb, step_key="step")
        _log_scalar(sw, "discrete/best_holdout_distance",
                    disc_res["best_holdout_manifold_distance"],
                    0, use_wandb=use_wandb, step_key="step")
        _log_scalar(sw, "discrete/subgroup_size",
                    disc_res["subgroup"]["num_elements"],
                    0, use_wandb=use_wandb, step_key="step")
        for step, v in enumerate(disc_res["history"]["loss_total"]):
            _log_scalar(sw, "discrete/loss_total", v, step, use_wandb=use_wandb, step_key="discrete_step")

    gens = result["generators"]
    if use_wandb and wandb is not None:
        _wandb_matrix("generators/W", gens["W"])
        for k in range(gens["L_s"].shape[0]):
            _wandb_matrix(f"generators/L_s_{k:02d}", gens["L_s"][k])
            _wandb_matrix(f"generators/L_a_{k:02d}", gens["L_a"][k])
            if gens.get("b_s") is not None:
                _wandb_vector(f"generators/b_s_{k:02d}", gens["b_s"][k])
            if gens.get("b_a") is not None:
                _wandb_vector(f"generators/b_a_{k:02d}", gens["b_a"][k])


def _log_gt_metrics(result, sw, env_name, n_s, n_a, *, use_wandb, min_scale=0.2):
    # Discrete metrics are independent of continuous generators — compute first
    # so they are logged even for purely-discrete envs (Z(2): Swimmer, Walker).
    gt_disc = get_ground_truth_discrete_symmetries(env_name, n_s, n_a)
    if gt_disc is not None and "discrete" in result:
        candidates = result["discrete"]["candidates"]
        disc_metrics = compute_discrete_symmetry_error(candidates, gt_disc)
        for tag, val in disc_metrics.items():
            _log_scalar(sw, tag, float(val), 0, use_wandb=use_wandb, step_key="step")
        print("  GT discrete: " +
              " | ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in disc_metrics.items()))

    L_s_gt, L_a_gt = get_ground_truth_generators(env_name, n_s, n_a)
    if L_s_gt is None:
        return
    L_s = result["generators"]["L_s"]
    L_a = result["generators"]["L_a"]
    scales = np.asarray(result["generators"]["scales"])
    num_found = float(int(np.sum(scales > min_scale * 1.5)))
    num_gt    = float(L_s_gt.shape[0])
    w_s = n_s / (n_s + n_a)
    w_a = n_a / (n_s + n_a)
    space_s = compute_space_error(L_s, L_s_gt)
    space_a = compute_space_error(L_a, L_a_gt)
    orth_s  = compute_orth_error(L_s)
    orth_a  = compute_orth_error(L_a)
    gt_metrics = {
        "eval/space_error_state":     space_s,
        "eval/space_error_action":    space_a,
        "eval/orth_error_state":      orth_s,
        "eval/orth_error_action":     orth_a,
        "eval/num_found_generators":  num_found,
        "eval/num_gt_generators":     num_gt,
        "eval/combined_error":        w_s * (space_s + orth_s) + w_a * (space_a + orth_a),
        "eval/combined_space_error":  w_s * space_s + w_a * space_a,
        "eval/sin_theta_max_state":   compute_sin_theta_max(L_s, L_s_gt),
        "eval/sin_theta_avg_state":   compute_sin_theta_avg(L_s, L_s_gt),
        "eval/sin_theta_max_action":  compute_sin_theta_max(L_a, L_a_gt),
        "eval/sin_theta_avg_action":  compute_sin_theta_avg(L_a, L_a_gt),
        "eval/generator_count_error": abs(num_found - num_gt),
    }
    for tag, val in gt_metrics.items():
        _log_scalar(sw, tag, float(val), 0, use_wandb=use_wandb, step_key="step")
    print("  GT metrics: " +
          " | ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in gt_metrics.items()))


def _print_summary(result):
    gens = result["generators"]
    mode = result["data"]["mode"]
    K = gens["L_s"].shape[0]
    print(f"\n=== liegan/{mode} summary ===")
    print(f"  {'k':>2}  {'||L_s||':>9}  {'||L_a||':>9}  {'scale':>8}")
    for k in range(K):
        print(
            f"  {k:>2}  {float(np.linalg.norm(gens['L_s'][k])):>9.3f}  "
            f"{float(np.linalg.norm(gens['L_a'][k])):>9.3f}  "
            f"{float(gens['scales'][k]):>8.4f}"
        )
    # Summary line in consistent format across all discovery methods.
    gap_idx = result.get("spectral_gap_index", K - 1)
    picked_k = gap_idx + 1
    gap_ratio = result.get("spectral_gap_ratio", float("nan"))
    clos_err  = result.get("closure_error",      float("nan"))
    print(
        f"\ncontinuous: picked_k={picked_k} after_closure={picked_k} "
        f"gap_ratio={gap_ratio:.3e} "
        f"closure_error={clos_err:.3e}"
    )
    if "discrete" in result:
        dr = result["discrete"]
        rho = dr.get("rho_mfd", float("nan"))
        print(
            f"\ndiscrete: accepted={len(dr['candidates'])} "
            f"rejected={len(dr.get('rejections', []))} "
            f"best_holdout_mfd={dr['best_holdout_manifold_distance']:.3e} "
            f"identity_mfd={dr.get('identity_manifold_distance', float('nan')):.3e} "
            f"rho_mfd={rho:.3e} "
            f"tau={dr.get('acceptance_tau', 0.0):.3e}"
        )
        print(f"\ngroup: num_elements={dr['subgroup']['num_elements']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    mode = FLAGS.mode
    assert mode in ("marginal", "conditional"), f"Unknown mode {mode!r}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"liegan_{mode}_{FLAGS.env_name}_seed{FLAGS.seed}_{timestamp}"
    exp_dir  = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    sw = SummaryWriter(exp_dir)

    env = make_env(FLAGS.env_name, seed=FLAGS.seed, save_folder=None)
    state_dim  = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    states_np, actions_np, next_states_np, episode_ids_np, step_ids_np = (
        build_offline_arrays(FLAGS.dataset_path)
    )
    n_total = states_np.shape[0]
    n_use = min(int(FLAGS.num_points), n_total)
    rng = np.random.default_rng(FLAGS.seed)
    idx = rng.choice(n_total, size=n_use, replace=False) if n_use < n_total else np.arange(n_total)
    idx = np.asarray(
        sorted(idx.tolist(), key=lambda i: (int(episode_ids_np[i]), int(step_ids_np[i]))),
        dtype=np.int64,
    )

    states      = torch.as_tensor(states_np[idx],      dtype=torch.float32)
    actions     = torch.as_tensor(actions_np[idx],     dtype=torch.float32)
    next_states = torch.as_tensor(next_states_np[idx], dtype=torch.float32)
    episode_ids = torch.as_tensor(episode_ids_np[idx], dtype=torch.long)
    step_ids    = torch.as_tensor(step_ids_np[idx],    dtype=torch.long)

    if state_dim  != states.shape[-1]:  state_dim  = int(states.shape[-1])
    if action_dim != actions.shape[-1]: action_dim = int(actions.shape[-1])

    if FLAGS.lifted_dimension:
        ones_s = torch.ones(states.shape[0], 1, dtype=torch.float32)
        ones_a = torch.ones(actions.shape[0], 1, dtype=torch.float32)
        states = torch.cat([states, ones_s], dim=-1)
        next_states = torch.cat([next_states, ones_s], dim=-1)
        actions = torch.cat([actions, ones_a], dim=-1)
        state_dim += 1
        action_dim += 1

    config = _build_config()

    use_wb = bool(FLAGS.use_wandb)
    if use_wb:
        if wandb is None:
            raise RuntimeError("wandb not installed but --use_wandb=True")
        os.environ["WANDB_MODE"] = "disabled" if FLAGS.wandb_mode == "disabled" else FLAGS.wandb_mode
        wandb.init(
            project=FLAGS.wandb_project,
            entity=(FLAGS.wandb_entity or None),
            group=(FLAGS.wandb_group or FLAGS.env_name),
            name=(FLAGS.wandb_run_name or exp_name),
            config={
                "method": f"liegan/{mode}",
                "gen_loss_flavor": FLAGS.gen_loss_flavor,
                "env_name": FLAGS.env_name,
                "dataset_path": FLAGS.dataset_path,
                "seed": FLAGS.seed,
                "num_points": n_use,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "K": FLAGS.K,
                "num_steps": FLAGS.num_steps,
                "gen_lr": FLAGS.gen_lr,
                "disc_lr": FLAGS.disc_lr,
                "lambda_isometry": FLAGS.lambda_isometry,
                "lambda_intertwine": FLAGS.lambda_intertwine,
                "init_scale": FLAGS.init_scale,
                "min_scale": FLAGS.min_scale,
                "fixed_scale": FLAGS.fixed_scale,
                "lambda_scale_floor": FLAGS.lambda_scale_floor,
                "lambda_scale_sparse": FLAGS.lambda_scale_sparse,
                "lambda_nontrivial": FLAGS.lambda_nontrivial,
                "nontrivial_margin": FLAGS.nontrivial_margin,
                "lambda_diverse": FLAGS.lambda_diverse,
                "lambda_rank": FLAGS.lambda_rank,
                "lifted_dimension": FLAGS.lifted_dimension,
            },
        )
        wandb.define_metric("step")
        wandb.define_metric("loss/*",   step_metric="step")
        wandb.define_metric("disc/*",   step_metric="step")
        wandb.define_metric("reg/*",    step_metric="step")
        wandb.define_metric("eval/*",   step_metric="step")

    result = discover_liegan_symmetries(
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        verbose=FLAGS.verbose,
        only_continuous=FLAGS.only_continuous,
    )

    _log_results(result, sw, use_wandb=use_wb)
    _log_gt_metrics(result, sw, FLAGS.env_name, state_dim, action_dim,
                    use_wandb=use_wb, min_scale=FLAGS.min_scale)
    _save_results(exp_dir, result)
    if FLAGS.verbose:
        _print_summary(result)

    if use_wb:
        wandb.finish()
    sw.flush()
    sw.close()
    print(f"[run] saved liegan/{mode} artifacts to {exp_dir}")


if __name__ == "__main__":
    app.run(main)
