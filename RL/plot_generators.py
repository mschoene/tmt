#!/usr/bin/env python3
"""Generator comparison figures.

Produces:
  - particle2d_generators.pdf/png  — GT + 4 models side by side, state row + action row
  - reacher_generators.pdf/png     — same for Reacher
  - particle3d_generators_state/action.pdf/png
  - colorbar.pdf/png               — standalone ±1 colorbar

Usage:
    python plot_generators.py --runs_dir /path/to/experiment/runs [--out figures/generators]

Expected layout under --runs_dir (searched recursively):
    {runs_dir}/**/tmt_nn/**/{env}**/snapshots/discovered_symmetry_step0.json
    {runs_dir}/**/tmt_knn/**/{env}**/snapshots/discovered_symmetry_step0.json
    {runs_dir}/**/liesd/**/{env}**/snapshots/discovered_symmetry_step0.json
    {runs_dir}/**/liegan/**/{env}**/snapshots/   (containing L_s_0.npy, L_a_0.npy)
"""
import argparse
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from representations import get_ground_truth_generators

ENV_DIM = {
    "ParticleCircle2D-v0": (4, 2),
    "ParticleCircle3D-v0": (6, 3),
    "Reacher-v5":           (10, 2),
}

# Short index labels per environment (None = just show 0,1,2,...)
# State labels after state_transform is applied (transformed coord order)
ENV_STATE_LABELS = {
    "ParticleCircle2D-v0": ["x", "y", "vx", "vy"],
    "ParticleCircle3D-v0": ["x", "y", "z", "vx", "vy", "vz"],
    "Reacher-v5": ["cq0", "cq1", "sq0", "sq1", "tx", "ty", "dq0", "dq1", "ex", "ey"],
}
ENV_ACTION_LABELS = {
    "ParticleCircle2D-v0": ["fx", "fy"],
    "ParticleCircle3D-v0": ["fx", "fy", "fz"],
    "Reacher-v5":           ["τ0", "τ1"],
}


# ── Path discovery ─────────────────────────────────────────────────────────────

def _env_in_path(path_str: str, env_name: str) -> bool:
    norm = env_name.replace("-", "").replace("_", "").lower()
    return norm in path_str.replace("-", "").replace("_", "").lower()


def discover_paths(runs_dir: Path, env_name: str) -> dict:
    """Auto-discover snapshot paths for each method under runs_dir.

    Searches recursively for discovered_symmetry_step0.json (tmt_nn/knn/liesd)
    and snapshots/ dirs containing L_s_0.npy (liegan).
    Returns {display_name: path_str}; missing methods are omitted.
    """
    paths = {}
    all_json = glob.glob(str(runs_dir / "**" / "discovered_symmetry_step0.json"), recursive=True)
    for method_key, display_name in [
        ("tmt_nn",  "TMT-NN"),
        ("tmt_knn", "TMT-KNN"),
        ("liesd",   "LieSD"),
    ]:
        matches = [f for f in all_json if method_key in f and _env_in_path(f, env_name)]
        if matches:
            paths[display_name] = matches[0]

    liegan_dirs = [
        p for p in glob.glob(str(runs_dir / "**" / "snapshots"), recursive=True)
        if "liegan" in p and _env_in_path(p, env_name)
        and (Path(p) / "L_s_0.npy").exists()
    ]
    if liegan_dirs:
        paths["LieGAN"] = liegan_dirs[0]

    return paths


# ── Loader helpers ─────────────────────────────────────────────────────────────
def load_npz(json_path):
    """Load L_s, L_a from discovered_symmetry_step0.json (tmt_nn/knn/liesd)."""
    meta = json.loads(Path(json_path).read_text())
    npz  = np.load(meta["npz_path"])
    return npz["L_s"].astype(float), npz["L_a"].astype(float)


def load_liegan(snap_dir):
    """Load L_s, L_a from LieGAN snapshot dir (saves individual L_s_k.npy files)."""
    snap = Path(snap_dir)
    ls_files = sorted(snap.glob("L_s_*.npy"),
                      key=lambda p: int(p.stem.split("_")[-1]))
    la_files = sorted(snap.glob("L_a_*.npy"),
                      key=lambda p: int(p.stem.split("_")[-1]))
    L_s = np.stack([np.load(f) for f in ls_files]).astype(float)
    L_a = np.stack([np.load(f) for f in la_files]).astype(float)
    return L_s, L_a


def load_model(path):
    p = Path(path)
    if p.suffix == ".json":
        return load_npz(p)
    else:  # directory = LieGAN
        return load_liegan(p)


def first_gen(L_s, L_a):
    """Return the first generator (index 0) for state and action."""
    return L_s[0], L_a[0]


def normalize(M, scale=None):
    """Normalize to [-1,1] using given scale, or M's own max if scale is None."""
    v = scale if scale is not None else np.abs(M).max()
    return M / v if v > 1e-12 else M


# ── Figure builders ────────────────────────────────────────────────────────────
def _set_index_ticks(ax, M, s_labels, a_labels, space, col):
    """Add index tick labels; only show y-axis on leftmost column."""
    n = M.shape[0]
    labels = s_labels if space == "state" else a_labels
    ticks = list(range(n))

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if labels else [str(i) for i in ticks],
                       fontsize=5, rotation=90, ha="center")
    if col == 0:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels if labels else [str(i) for i in ticks],
                           fontsize=5)
    else:
        ax.set_yticks([])
    ax.tick_params(length=2, pad=1)


def make_comparison_figure(env, paths, out_path, size_in=1.4):
    """One figure: rows=state/action, cols=GT + each model. First gen only."""
    n_s, n_a = ENV_DIM[env]
    gt_Ls, gt_La = get_ground_truth_generators(env, n_s, n_a)

    s_labels = ENV_STATE_LABELS.get(env)
    a_labels = ENV_ACTION_LABELS.get(env)

    model_names = list(paths.keys())
    col_labels = ["Ground Truth"] + model_names
    n_cols = len(col_labels)

    # Collect first generators per column.
    # Normalize action by the SAME scale as its state generator so near-zero
    # action generators (e.g. discovered on invariant torques) appear near-zero.
    state_gens, action_gens = [], []
    s_scale = np.abs(gt_Ls[0]).max()
    a_scale = np.abs(gt_La[0]).max()
    state_gens.append(normalize(gt_Ls[0], s_scale))
    action_gens.append(normalize(gt_La[0], a_scale if a_scale > 1e-12 else 1.0))
    for _, path in paths.items():
        L_s, L_a = load_model(path)
        sc = np.abs(L_s[0]).max()
        state_gens.append(normalize(L_s[0], sc))
        action_gens.append(normalize(L_a[0], sc))  # same scale as state

    # Extra height for tick labels
    fig, axes = plt.subplots(2, n_cols, figsize=(size_in * n_cols, size_in * 2 + 0.8))

    for col, (lbl, Ms, Ma) in enumerate(zip(col_labels, state_gens, action_gens)):
        for row, (ax, M, space) in enumerate(zip(axes[:, col], [Ms, Ma], ["state", "action"])):
            ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            _set_index_ticks(ax, M, s_labels, a_labels, space, col)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"{lbl}\n$L^s_0$", fontsize=8, pad=3)
            else:
                ax.set_title(f"$L^a_0$", fontsize=8, pad=3)

    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}.pdf/png")


def save_colorbar(path, height_in=2.0):
    """Standalone vertical colorbar for [-1, 1]."""
    fig, ax = plt.subplots(figsize=(0.3, height_in))
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm   = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["-1", "0", "1"])
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {path}.pdf/png")


def make_stacked_figure(env, paths, out_path, n_show=3, space="state", size_in=1.2):
    """Stacked multi-generator figure: rows=generators (0..n_show-1), cols=GT+models.

    One plot for state, one for action (call twice with space='state'/'action').
    Action is normalized by the same scale as its state generator so near-zero
    action generators appear near-zero.
    """
    n_s, n_a = ENV_DIM[env]
    s_labels = ENV_STATE_LABELS.get(env)
    a_labels = ENV_ACTION_LABELS.get(env)
    labels = s_labels if space == "state" else a_labels
    n_dim = n_s if space == "state" else n_a

    gt_Ls, gt_La = get_ground_truth_generators(env, n_s, n_a)

    model_names = list(paths.keys())
    col_labels  = ["Ground Truth"] + model_names
    n_cols = len(col_labels)

    # Load all generators; normalize action by state scale
    all_Ls, all_La = [gt_Ls], [gt_La]
    for path in paths.values():
        Ls, La = load_model(path)
        all_Ls.append(Ls)
        all_La.append(La)

    fig, axes = plt.subplots(n_show, n_cols,
                             figsize=(size_in * n_cols, size_in * n_show + 0.5))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for col, (lbl, Ls, La) in enumerate(zip(col_labels, all_Ls, all_La)):
        L = Ls if space == "state" else La
        sc_s = np.abs(Ls[0]).max()
        sc = sc_s  # normalize action by state scale
        for row in range(n_show):
            ax = axes[row, col]
            if row < L.shape[0]:
                M = normalize(L[row], sc)
            else:
                M = np.zeros((n_dim, n_dim))
            ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ticks = list(range(n_dim))
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels if labels else [str(i) for i in ticks],
                               fontsize=5, rotation=90, ha="center")
            if col == 0:
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels if labels else [str(i) for i in ticks],
                                   fontsize=5)
            else:
                ax.set_yticks([])
            ax.tick_params(length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"{lbl}\n$L^{space[0]}_0$", fontsize=8, pad=3)
            else:
                ax.set_ylabel(f"$L^{space[0]}_{row}$", fontsize=7, rotation=0,
                              labelpad=18, va="center")

    fig.suptitle(f"{env} — {space} generators (top {n_show})", fontsize=9, y=1.01)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}.pdf/png")


def save_generators(env, paths, out_dir):
    """Save raw L_s and L_a arrays for each method as .npy files."""
    n_s, n_a = ENV_DIM[env]
    gt_Ls, gt_La = get_ground_truth_generators(env, n_s, n_a)
    np.save(out_dir / "GT_L_s.npy", gt_Ls)
    np.save(out_dir / "GT_L_a.npy", gt_La)
    print(f"  GT: L_s {gt_Ls.shape}, L_a {gt_La.shape}")
    for name, path in paths.items():
        L_s, L_a = load_model(path)
        np.save(out_dir / f"{name}_L_s.npy", L_s)
        np.save(out_dir / f"{name}_L_a.npy", L_a)
        print(f"  {name}: L_s {L_s.shape}  L_a max={np.abs(L_a[0]).max():.4e}  L_s max={np.abs(L_s[0]).max():.4e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", type=Path, required=True,
                        help="Root directory of experiment runs (searched recursively).")
    parser.add_argument("--out", default="figures/generators")
    parser.add_argument("--size", type=float, default=1.4)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = args.runs_dir.resolve()

    for env, label, out_stem, extra in [
        ("ParticleCircle2D-v0", "ParticleCircle2D", "particle2d_generators", None),
        ("Reacher-v5",          "Reacher",          "reacher_generators",    "save"),
        ("ParticleCircle3D-v0", "ParticleCircle3D", None,                    "stacked"),
    ]:
        print(f"=== {label} ===")
        paths = discover_paths(runs_dir, env)
        if not paths:
            print("  No snapshots found — skipping.")
            continue
        if extra == "stacked":
            for space in ("state", "action"):
                make_stacked_figure(env, paths,
                                    out / f"particle3d_generators_{space}",
                                    n_show=3, space=space, size_in=args.size)
        else:
            make_comparison_figure(env, paths, out / out_stem, args.size)
            if extra == "save":
                print(f"--- Saving {label} generators ---")
                save_generators(env, paths, out)

    print("=== Colorbar ===")
    save_colorbar(out / "colorbar")


if __name__ == "__main__":
    main()
