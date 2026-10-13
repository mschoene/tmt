"""LieSD training driver: Jacobian-based tangent-space symmetry discovery."""

import os
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app, flags
from tensorboardX import SummaryWriter

from dataset_loader import build_offline_arrays
from jax_rl.envs import simpleEnvs  # noqa: F401
from representations import (get_ground_truth_generators, compute_space_error,
                              compute_orth_error, compute_sin_theta_max, compute_sin_theta_avg,
                              save_discovered_symmetry,
                              get_ground_truth_discrete_symmetries,
                              compute_discrete_symmetry_error,
                              save_discovered_discrete_symmetry)
from jax_rl.networks.liesd_symmetry_discovery import (
    ContinuousDiscoveryConfig,
    DiscreteDiscoveryConfig,
    DynamicsTrainingConfig,
    GroupAssemblyConfig,
    LieSDConfig,
    discover_liesd_symmetries,
)
from jax_rl.utils import make_env

try:
    import wandb
except Exception:
    wandb = None

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "ParticleCircle2D-v0", "Environment name.")
flags.DEFINE_string('dataset_path', 'path/to/your/dataset.pkl', 'Path to dataset (.pkl) or Minari ID.')

flags.DEFINE_string("save_dir", "/path/to/logs/liesd/", "Logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_points", 100000, "Maximum number of transitions to use.")
flags.DEFINE_boolean("verbose", True, "Print phase progress and summaries.")

# Dynamics model
flags.DEFINE_multi_integer("hidden_dims", [256, 256], "Hidden dims for dynamics MLP.")
flags.DEFINE_float("dynamics_lr", 1e-3, "Dynamics model learning rate.")
flags.DEFINE_float("dynamics_weight_decay", 1e-6, "Dynamics model weight decay.")
flags.DEFINE_integer("dynamics_batch_size", 2048, "Dynamics model batch size.")
flags.DEFINE_integer("dynamics_num_epochs", 2000, "Dynamics model training epochs.")
flags.DEFINE_float("validation_split", 0.1, "Validation split fraction.")
flags.DEFINE_float("holdout_split", 0.1, "Holdout split fraction.")
flags.DEFINE_integer("dynamics_log_interval", 50, "Print every N epochs during dynamics training.")
flags.DEFINE_string("dynamics_lr_schedule", "cosine", "LR schedule: constant|cosine|linear.")
flags.DEFINE_float("dynamics_lr_min_factor", 0.01, "Minimum LR as fraction of initial LR.")
flags.DEFINE_integer("dynamics_lr_warmup_epochs", 0, "Warmup epochs before LR decay.")

# Continuous discovery
flags.DEFINE_integer("continuous_max_generators", 8, "Maximum initial continuous generators.")
flags.DEFINE_integer("continuous_num_frame_points", 10000, "Points used to estimate normal frames.")
flags.DEFINE_integer("continuous_frame_batch_size", 256, "Batch size for Jacobian frame estimation.")
flags.DEFINE_float("continuous_spectral_gap_min_ratio", 1.2, "Minimum spectral-gap ratio.")
flags.DEFINE_float("continuous_abs_threshold_frac", 1e-1, "Absolute eigenvalue threshold.")
flags.DEFINE_float("continuous_closure_tol", 5e-2, "Lie-closure tolerance.")
flags.DEFINE_integer("continuous_closure_max_generators", 16, "Maximum generators after closure.")
flags.DEFINE_integer("continuous_polish_steps", 2, "SGD polish steps per generator.")
flags.DEFINE_float("continuous_polish_lr", 0.1, "Learning rate for generator polish.")
flags.DEFINE_integer("continuous_polish_batch_size", 256, "Batch size for generator polish.")

# Discrete discovery
flags.DEFINE_integer("discrete_num_candidates", 64, "Number of simultaneous discrete candidates.")
flags.DEFINE_float("discrete_lr", 1e-2, "Learning rate for discrete search.")
flags.DEFINE_integer("discrete_num_steps", 1000, "Optimisation steps for discrete search.")
flags.DEFINE_integer("discrete_batch_size", 1024, "Batch size for discrete search.")
flags.DEFINE_integer("sinkhorn_iters", 200, "Sinkhorn iterations.")
flags.DEFINE_float("disc_temp_start", 5.0, "Initial Sinkhorn temperature.")
flags.DEFINE_float("disc_temp_end", 0.015, "Final Sinkhorn temperature.")
flags.DEFINE_float("disc_anneal_power", 1.0, "Temperature anneal power.")
flags.DEFINE_float("disc_mu_identity_repulsion", 0.05, "Weight on non-identity barrier term.")
flags.DEFINE_float("disc_identity_action_weight", 0.0, "Action-space weight in identity barrier.")
flags.DEFINE_float("disc_kappa_diversity", 0.5, "Weight on pairwise candidate diversity.")
flags.DEFINE_float("disc_manifold_distance_tau", 0.25, "Holdout distance threshold for acceptance.")
flags.DEFINE_integer("disc_max_subgroup_elements", 64, "Maximum elements when closing discrete subgroup.")
flags.DEFINE_integer("discrete_log_interval", 50, "Print every N steps during discrete search.")
flags.DEFINE_boolean("only_discrete", False, "Skip continuous discovery phase, run discrete only.")
flags.DEFINE_boolean("only_continuous", False, "Skip discrete discovery phase, run continuous only.")
flags.DEFINE_integer("discrete_candidates_batch_size", -1, "Run discrete search in batches of this many candidates to cap peak memory (-1 = all at once).")

flags.DEFINE_boolean("use_intertwiner", True, "Learn shared W coupling G_s<->G_a in discrete search.")
flags.DEFINE_float("mu_intertwiner", 1.0, "Weight for intertwiner penalty.")
flags.DEFINE_float("max_intertwiner_residual", 1e9, "Reject continuous generators above this residual.")
flags.DEFINE_boolean("fix_L_a_from_intertwiner", False, "Replace L_a with pseudoinverse Sylvester solution.")

# Group assembly
flags.DEFINE_float("orbit_theta", 0.5, "Theta for exponentiating continuous generators.")
flags.DEFINE_integer("group_max_elements", 128, "Maximum total group elements to keep.")

flags.DEFINE_boolean("lifted_dimension", False, "Append constant-1 dim to state and action for SE(n) symmetry discovery.")

flags.DEFINE_boolean("normalize_data", True, "Standardize z=[s;a;s'] to zero mean / unit std before training.")

# Logging
flags.DEFINE_boolean("use_wandb", True, "Enable Weights & Biases logging.")
flags.DEFINE_string("wandb_project", "tmt", "Weights & Biases project.")
flags.DEFINE_string("wandb_entity", "", "Weights & Biases entity.")
flags.DEFINE_string("wandb_run_name", "", "Optional run name.")
flags.DEFINE_string("wandb_group", "", "Optional run group.")
flags.DEFINE_string("wandb_mode", "online", "wandb mode: online/offline/disabled.")


# ---------------------------------------------------------------------------
# Plotting helpers (identical to train_tmt_nn.py)
# ---------------------------------------------------------------------------

def _plot_matrix(matrix: np.ndarray, title: str, path: str):
    mat = np.asarray(matrix, dtype=np.float64)
    vabs = float(np.max(np.abs(mat))) if mat.size else 1.0
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mat, cmap="RdBu", vmin=-max(vabs, 1.0), vmax=max(vabs, 1.0))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_curve(values, title: str, ylabel: str, path: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_discovered_config(snap_dir, result, env_name, n_s, n_a):
    gens = result["continuous"]["generators"]
    if not gens:
        return
    L_s = np.stack([np.asarray(g["L_s"]) for g in gens])
    L_a = np.stack([np.asarray(g["L_a"]) for g in gens])
    W = (np.stack([np.asarray(g["intertwiner_W"]) for g in gens])
         if all("intertwiner_W" in g for g in gens) else None)
    save_discovered_symmetry(snap_dir, step=0, env_name=env_name,
                             n_s=n_s, n_a=n_a, L_s=L_s, L_a=L_a, W=W)


def _log_gt_metrics(result, sw, env_name, n_s, n_a, *, use_wandb):
    gens = result["continuous"]["generators"]
    L_s_gt, L_a_gt = get_ground_truth_generators(env_name, n_s, n_a)
    if gens and L_s_gt is not None:
        L_s = np.stack([np.asarray(g["L_s"]) for g in gens])
        L_a = np.stack([np.asarray(g["L_a"]) for g in gens])
        w_s = n_s / (n_s + n_a)
        w_a = n_a / (n_s + n_a)
        space_s  = compute_space_error(L_s, L_s_gt)
        space_a  = compute_space_error(L_a, L_a_gt)
        orth_s   = compute_orth_error(L_s)
        orth_a   = compute_orth_error(L_a)
        num_found = float(len(gens))
        num_gt    = float(L_s_gt.shape[0])
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
            sw.add_scalar(tag, float(val), 0)
        if use_wandb and wandb is not None:
            wandb.log(gt_metrics)
        print("  GT metrics: " +
              " | ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in gt_metrics.items()))

    # Discrete ground-truth evaluation (runs even when only_discrete=True)
    gt_disc = get_ground_truth_discrete_symmetries(env_name, n_s, n_a)
    if gt_disc is not None:
        candidates = result.get("discrete", {}).get("candidates", [])
        disc_metrics = compute_discrete_symmetry_error(candidates, gt_disc)
        for tag, val in disc_metrics.items():
            sw.add_scalar(tag, float(val), 0)
        if use_wandb and wandb is not None:
            wandb.log(disc_metrics)
        print("  Discrete GT: " +
              " | ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in disc_metrics.items()))


def _log_scalar(sw, tag, value, step, *, use_wandb, step_key):
    sw.add_scalar(tag, value, step)
    if use_wandb and wandb is not None:
        wandb.log({tag: value, step_key: int(step)})


def _wandb_log_matrix_image(tag, mat):
    if wandb is None:
        return
    mat = np.asarray(mat, dtype=np.float64)
    vabs = float(np.max(np.abs(mat))) if mat.size else 1.0
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mat, cmap="RdBu", vmin=-max(vabs, 1.0), vmax=max(vabs, 1.0))
    ax.set_title(tag)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config():
    return LieSDConfig(
        training=DynamicsTrainingConfig(
            hidden_dims=tuple(FLAGS.hidden_dims),
            lr=FLAGS.dynamics_lr,
            normalize_data=FLAGS.normalize_data,
            weight_decay=FLAGS.dynamics_weight_decay,
            batch_size=FLAGS.dynamics_batch_size,
            num_epochs=FLAGS.dynamics_num_epochs,
            validation_split=FLAGS.validation_split,
            holdout_split=FLAGS.holdout_split,
            log_interval=FLAGS.dynamics_log_interval,
            seed=FLAGS.seed,
            lr_schedule=FLAGS.dynamics_lr_schedule,
            lr_min_factor=FLAGS.dynamics_lr_min_factor,
            lr_warmup_epochs=FLAGS.dynamics_lr_warmup_epochs,
        ),
        continuous=ContinuousDiscoveryConfig(
            max_generators=FLAGS.continuous_max_generators,
            num_frame_points=FLAGS.continuous_num_frame_points,
            frame_batch_size=FLAGS.continuous_frame_batch_size,
            spectral_gap_min_ratio=FLAGS.continuous_spectral_gap_min_ratio,
            closure_tol=FLAGS.continuous_closure_tol,
            closure_max_generators=FLAGS.continuous_closure_max_generators,
            polish_steps=FLAGS.continuous_polish_steps,
            polish_lr=FLAGS.continuous_polish_lr,
            polish_batch_size=FLAGS.continuous_polish_batch_size,
            abs_threshold_frac=FLAGS.continuous_abs_threshold_frac,
            max_intertwiner_residual=FLAGS.max_intertwiner_residual,
            fix_L_a_from_intertwiner=FLAGS.fix_L_a_from_intertwiner,
        ),
        discrete=DiscreteDiscoveryConfig(
            num_candidates=FLAGS.discrete_num_candidates,
            lr=FLAGS.discrete_lr,
            num_steps=FLAGS.discrete_num_steps,
            batch_size=FLAGS.discrete_batch_size,
            sinkhorn_iters=FLAGS.sinkhorn_iters,
            temperature_start=FLAGS.disc_temp_start,
            temperature_end=FLAGS.disc_temp_end,
            anneal_power=FLAGS.disc_anneal_power,
            mu_identity_repulsion=FLAGS.disc_mu_identity_repulsion,
            identity_repulsion_action_weight=FLAGS.disc_identity_action_weight,
            kappa_diversity=FLAGS.disc_kappa_diversity,
            manifold_distance_tau=FLAGS.disc_manifold_distance_tau,
            max_subgroup_elements=FLAGS.disc_max_subgroup_elements,
            log_interval=FLAGS.discrete_log_interval,
            seed=FLAGS.seed,
            use_intertwiner=FLAGS.use_intertwiner,
            mu_intertwiner=FLAGS.mu_intertwiner,
            candidates_batch_size=FLAGS.discrete_candidates_batch_size,
        ),
        group=GroupAssemblyConfig(
            validation_theta=FLAGS.orbit_theta,
            max_elements=FLAGS.group_max_elements,
        ),
    )


# ---------------------------------------------------------------------------
# Save / log results
# ---------------------------------------------------------------------------

_MAX_WANDB_CANDIDATES = 5


def _save_results(exp_dir, result, env_name="", n_s=0, n_a=0):
    snap_dir = os.path.join(exp_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    try:
        import equinox as eqx
        eqx.tree_serialise_leaves(
            os.path.join(snap_dir, "liesd_projector.eqx"), result["model"]
        )
    except Exception:
        pass

    summary = {
        "training_history": result["training_history"],
        "standardizer": result["standardizer"],
        "data": result["data"],
        "continuous": result["continuous"],
        "discrete": result["discrete"],
        "group": result["group"],
    }
    torch.save(summary, os.path.join(snap_dir, "liesd_results.pt"))

    np.save(os.path.join(snap_dir, "z_mean.npy"), result["standardizer"]["mean"])
    np.save(os.path.join(snap_dir, "z_scale.npy"), result["standardizer"]["scale"])
    np.save(os.path.join(snap_dir, "continuous_spectrum.npy"), result["continuous"]["spectrum"])
    np.save(os.path.join(snap_dir, "continuous_matrix_M.npy"), result["continuous"]["matrix_M"])
    np.save(os.path.join(snap_dir, "discrete_soft_P_s.npy"), result["discrete"]["soft_P_s"])
    np.save(os.path.join(snap_dir, "discrete_soft_P_a.npy"), result["discrete"]["soft_P_a"])
    np.save(os.path.join(snap_dir, "discrete_hard_P_s.npy"), result["discrete"]["hard_P_s"])
    np.save(os.path.join(snap_dir, "discrete_hard_P_a.npy"), result["discrete"]["hard_P_a"])
    for idx, gen in enumerate(result["continuous"]["generators"]):
        np.save(os.path.join(snap_dir, f"L_s_{idx}.npy"), gen["L_s"])
        np.save(os.path.join(snap_dir, f"L_a_{idx}.npy"), gen["L_a"])
    if result["discrete"].get("subgroup", {}).get("elements"):
        save_discovered_discrete_symmetry(
            snap_dir, result["discrete"]["subgroup"],
            env_name=FLAGS.env_name, n_s=result["discrete"]["hard_P_s"].shape[0],
            n_a=result["discrete"]["hard_P_a"].shape[0],
        )

    th = result["training_history"]
    if th["train_total_loss"]:
        _plot_curve(th["train_total_loss"], "Train Dynamics MSE", "mse",
                    os.path.join(snap_dir, "train_dynamics_mse.png"))
    if th["val_total_loss"]:
        _plot_curve(th["val_total_loss"], "Val Dynamics MSE", "mse",
                    os.path.join(snap_dir, "val_dynamics_mse.png"))
    _plot_curve(result["discrete"]["history"]["loss_total"], "Discrete Total Loss", "loss",
                os.path.join(snap_dir, "discrete_total_loss.png"))
    _plot_curve(result["continuous"]["spectrum"].tolist(), "Continuous Spectrum", "eigenvalue",
                os.path.join(snap_dir, "continuous_spectrum.png"))
    _plot_matrix(result["continuous"]["matrix_M"], "Continuous M",
                 os.path.join(snap_dir, "continuous_M.png"))
    _plot_matrix(result["discrete"]["soft_P_s"], "Soft P_s",
                 os.path.join(snap_dir, "soft_P_s.png"))
    _plot_matrix(result["discrete"]["soft_P_a"], "Soft P_a",
                 os.path.join(snap_dir, "soft_P_a.png"))
    _plot_matrix(result["discrete"]["hard_P_s"], "Hard P_s",
                 os.path.join(snap_dir, "hard_P_s.png"))
    _plot_matrix(result["discrete"]["hard_P_a"], "Hard P_a",
                 os.path.join(snap_dir, "hard_P_a.png"))
    for idx, gen in enumerate(result["continuous"]["generators"]):
        _plot_matrix(gen["L_s"], f"L_s[{idx}]", os.path.join(snap_dir, f"L_s_{idx}.png"))
        _plot_matrix(gen["L_a"], f"L_a[{idx}]", os.path.join(snap_dir, f"L_a_{idx}.png"))

    if env_name and n_s > 0 and n_a > 0:
        _save_discovered_config(snap_dir, result, env_name, n_s, n_a)


def _log_results(result, sw, *, use_wandb, discrete_log_interval=1):
    th = result["training_history"]
    for ep, v in enumerate(th["train_total_loss"]):
        _log_scalar(sw, "dynamics/train_mse", float(v), ep, use_wandb=use_wandb, step_key="epoch")
    for ep, v in enumerate(th["val_total_loss"]):
        if not (isinstance(v, float) and v != v):  # skip nan
            _log_scalar(sw, "dynamics/val_mse", float(v), ep, use_wandb=use_wandb, step_key="epoch")

    diag = th["diagnostics"]
    for k in ["train_manifold_rmse", "holdout_manifold_rmse"]:
        _log_scalar(sw, f"dynamics/{k}", float(diag[k]), 0, use_wandb=use_wandb, step_key="epoch")

    stride = max(1, int(discrete_log_interval))
    disc_hist = result["discrete"]["history"]
    for step, v in enumerate(disc_hist["loss_total"]):
        if step % stride == 0 or step == len(disc_hist["loss_total"]) - 1:
            _log_scalar(sw, "discrete/loss_total", float(v), step, use_wandb=use_wandb, step_key="disc_step")

    for idx, eig in enumerate(result["continuous"]["spectrum"]):
        _log_scalar(sw, "continuous/eigenvalue", float(eig), idx, use_wandb=use_wandb, step_key="spec_idx")

    one_offs = {
        "continuous/num_generators": float(len(result["continuous"]["generators"])),
        "continuous/picked_k": float(result["continuous"]["picked_k"]),
        "continuous/closure_error": float(result["continuous"]["closure_error"]),
        "continuous/spectral_gap_ratio": float(result["continuous"]["spectral_gap_ratio"]),
        "discrete/num_accepted": float(len(result["discrete"]["candidates"])),
        "discrete/num_rejected": float(len(result["discrete"]["rejections"])),
        "discrete/best_holdout_manifold_distance": float(result["discrete"]["best_holdout_manifold_distance"]),
        "group/num_elements": float(result["group"]["num_elements"]),
    }
    for tag, value in one_offs.items():
        _log_scalar(sw, tag, value, 0, use_wandb=use_wandb, step_key="epoch")

    if use_wandb and wandb is not None:
        _wandb_log_matrix_image("continuous/M", result["continuous"]["matrix_M"])
        _wandb_log_matrix_image("discrete/soft_P_s", result["discrete"]["soft_P_s"])
        _wandb_log_matrix_image("discrete/soft_P_a", result["discrete"]["soft_P_a"])
        _wandb_log_matrix_image("discrete/hard_P_s", result["discrete"]["hard_P_s"])
        _wandb_log_matrix_image("discrete/hard_P_a", result["discrete"]["hard_P_a"])
        for gen in result["continuous"]["generators"]:
            idx = int(gen["index"])
            _wandb_log_matrix_image(f"continuous/L_s_{idx:02d}", gen["L_s"])
            _wandb_log_matrix_image(f"continuous/L_a_{idx:02d}", gen["L_a"])
        for cand in result["discrete"]["candidates"][:_MAX_WANDB_CANDIDATES]:
            idx = int(cand["index"])
            _wandb_log_matrix_image(f"discrete/accepted_{idx:02d}_hard_P_s", cand["hard_P_s"])
            _wandb_log_matrix_image(f"discrete/accepted_{idx:02d}_hard_P_a", cand["hard_P_a"])


def _print_summary(result):
    diag = result["training_history"]["diagnostics"]
    cnt = result["continuous"]
    dsc = result["discrete"]
    grp = result["group"]

    print("\n=== liesd run summary ===")
    print(
        f"dynamics: train_mse={diag['train_manifold_rmse']:.3e} "
        f"holdout_mse={diag['holdout_manifold_rmse']:.3e}"
    )
    print(
        f"\ncontinuous: picked_k={cnt['picked_k']} "
        f"after_closure={len(cnt['generators'])} "
        f"gap_ratio={cnt['spectral_gap_ratio']:.3e} "
        f"closure_error={cnt['closure_error']:.3e}"
    )
    if cnt["generators"]:
        print(f"  {'i':>2}  {'obj':>10}  {'||L_s||':>9}  {'||L_a||':>9}  {'closure?':>8}")
        for gen in cnt["generators"]:
            print(
                f"  {int(gen['index']):>2}  {float(gen['objective']):>10.3e}  "
                f"{float(gen['state_norm']):>9.3f}  {float(gen['action_norm']):>9.3f}  "
                f"{str(bool(gen['from_closure'])):>8}"
            )
    _identity_mfd = dsc.get("identity_manifold_distance", float("nan"))
    _rho_mfd = dsc.get("rho_mfd", float("nan"))
    print(
        f"\ndiscrete: accepted={len(dsc['candidates'])} "
        f"rejected={len(dsc['rejections'])} "
        f"best_holdout_mfd={dsc['best_holdout_manifold_distance']:.3e} "
        f"identity_mfd={_identity_mfd:.3e} "
        f"rho_mfd={_rho_mfd:.3e} "
        f"tau={dsc['acceptance_tau']:.3e}"
    )
    print(f"\ngroup: num_elements={grp['num_elements']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"liesd_{FLAGS.env_name}_seed{FLAGS.seed}_{timestamp}"
    exp_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    sw = SummaryWriter(exp_dir)

    env = make_env(FLAGS.env_name, seed=FLAGS.seed, save_folder=None)
    state_dim = int(np.prod(env.observation_space.shape))
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

    states = torch.as_tensor(states_np[idx], dtype=torch.float32)
    actions = torch.as_tensor(actions_np[idx], dtype=torch.float32)
    next_states = torch.as_tensor(next_states_np[idx], dtype=torch.float32)
    episode_ids = torch.as_tensor(episode_ids_np[idx], dtype=torch.long)
    step_ids = torch.as_tensor(step_ids_np[idx], dtype=torch.long)

    if state_dim != states.shape[-1]:
        state_dim = int(states.shape[-1])
    if action_dim != actions.shape[-1]:
        action_dim = int(actions.shape[-1])

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
                "method": "liesd",
                "env_name": FLAGS.env_name,
                "dataset_path": FLAGS.dataset_path,
                "seed": FLAGS.seed,
                "num_points": n_use,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dims": FLAGS.hidden_dims,
                "dynamics_lr": FLAGS.dynamics_lr,
                "dynamics_num_epochs": FLAGS.dynamics_num_epochs,
                "continuous_num_frame_points": FLAGS.continuous_num_frame_points,
                "discrete_num_candidates": FLAGS.discrete_num_candidates,
                "discrete_num_steps": FLAGS.discrete_num_steps,
                "sinkhorn_iters": FLAGS.sinkhorn_iters,
                "lifted_dimension": FLAGS.lifted_dimension,
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("dynamics/*", step_metric="epoch")
        wandb.define_metric("disc_step")
        wandb.define_metric("discrete/*", step_metric="disc_step")
        wandb.define_metric("spec_idx")
        wandb.define_metric("continuous/eigenvalue", step_metric="spec_idx")

    result = discover_liesd_symmetries(
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        verbose=FLAGS.verbose,
        discrete_log_interval=FLAGS.discrete_log_interval,
        only_discrete=FLAGS.only_discrete,
        only_continuous=FLAGS.only_continuous,
    )

    _log_results(result, sw, use_wandb=use_wb,
                 discrete_log_interval=FLAGS.discrete_log_interval)
    _log_gt_metrics(result, sw, FLAGS.env_name, state_dim, action_dim, use_wandb=use_wb)
    _save_results(exp_dir, result, env_name=FLAGS.env_name, n_s=state_dim, n_a=action_dim)
    if FLAGS.verbose:
        _print_summary(result)

    if use_wb:
        wandb.finish()
    sw.flush()
    sw.close()
    print(f"[run] saved liesd artifacts to {exp_dir}")


if __name__ == "__main__":
    app.run(main)
