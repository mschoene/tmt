"""
Plot eigenvalue spectra of discovered Lie algebra generators.

Compares TMT-NN, TMT-KNN, and LieSD on:
  - ParticleCircle3D-v0 (6D state, SO(3) true group)
  - Reacher-v5 (10D state)

For each (env, method) panel: scatter eigenvalues of L_s_0 (K=1 generator)
in the complex plane across available seeds.

Usage:
    python plot_eigenspectra.py --runs_dir /path/to/experiment/runs [--out figures/]

Expected layout under --runs_dir (searched recursively):
    {runs_dir}/**/tmt_nn/**/{env}**/seed{N}/**/snapshots/L_s_0.npy
    {runs_dir}/**/tmt_knn/**/{env}**/seed{N}/**/snapshots/L_s_0.npy
    {runs_dir}/**/liesd/**/{env}**/seed{N}/**/snapshots/L_s_0.npy
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METHOD_STYLE = {
    "tmt_nn":  dict(color="#4CAF50", label="TMT-NN",  marker="o", zorder=3),
    "tmt_knn": dict(color="#FF9800", label="TMT-KNN", marker="s", zorder=3),
    "liesd":   dict(color="#9C27B0", label="LieSD",   marker="^", zorder=3),
}

SEED_ALPHA = [1.0, 0.65, 0.35]
SEEDS = [42, 43, 44]
METHODS = list(METHOD_STYLE.keys())

ENVS = {
    "particle3d": {
        "env_name": "ParticleCircle3D-v0",
        "title":    "ParticleCircle3D-v0  (SO(3), 6D state)",
        "n_s":      6,
        # Standard SO(3) generator in 6D (block-diag 2×3D): eigenvalues 0,0,±i,±i
        "gt_eigs":  [0+0j, 0+0j, 0+1j, 0+1j, 0-1j, 0-1j],
    },
    "reacher": {
        "env_name": "Reacher-v5",
        "title":    "Reacher-v5  (10D state)",
        "n_s":      10,
        "gt_eigs":  None,
    },
}


def _env_in_path(path: str, env_name: str) -> bool:
    norm = env_name.replace("-", "").replace("_", "").lower()
    return norm in path.replace("-", "").replace("_", "").lower()


def load_generator(runs_dir, method, env_name, seed, gen_idx=0, which="L_s"):
    """Return array for L_{which}_{gen_idx} or None if not found."""
    pattern = os.path.join(runs_dir, "**", "snapshots", f"{which}_{gen_idx}.npy")
    files = [
        f for f in glob.glob(pattern, recursive=True)
        if method in f and _env_in_path(f, env_name) and f"seed{seed}" in f
    ]
    return np.load(files[0]) if files else None


def load_all_generators(runs_dir, method, env_name, seed, which="L_s"):
    """Return list of all L_{which}_k matrices for a seed (sorted by index)."""
    pattern = os.path.join(runs_dir, "**", "snapshots", f"{which}_*.npy")
    files = [
        f for f in glob.glob(pattern, recursive=True)
        if method in f and _env_in_path(f, env_name) and f"seed{seed}" in f
        and os.path.basename(f).split("_")[2].split(".")[0].isdigit()
    ]
    files = sorted(files, key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))
    return [np.load(f) for f in files]


def eigenvalues(L):
    return np.linalg.eigvals(L)


def add_axes(ax):
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Re(λ)", fontsize=9)
    ax.set_ylabel("Im(λ)", fontsize=9)
    ax.tick_params(labelsize=8)


def plot_gt_eigs(ax, gt_eigs, scale=1.0):
    if gt_eigs is None:
        return
    gt = np.array(gt_eigs) * scale
    ax.scatter(gt.real, gt.imag, marker="x", s=80, color="black",
               linewidths=2, zorder=5, label="GT SO(3)")


def fig_complex_plane(runs_dir, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5),
                             sharex="row", sharey="row")
    fig.suptitle("Eigenvalue spectra of K=1 state generator (L_s_0)",
                 fontsize=12, y=1.01)

    for row, (env_key, env_cfg) in enumerate(ENVS.items()):
        env_name = env_cfg["env_name"]
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            add_axes(ax)
            st = METHOD_STYLE[method]

            plotted_any = False
            for si, seed in enumerate(SEEDS):
                L = load_generator(runs_dir, method, env_name, seed, gen_idx=0)
                if L is None:
                    continue
                eigs = eigenvalues(L)
                ax.scatter(eigs.real, eigs.imag,
                           c=st["color"], marker=st["marker"],
                           s=55, alpha=SEED_ALPHA[si], edgecolors="none",
                           label=f"seed {seed}" if si == 0 else None)
                plotted_any = True

            if plotted_any and env_cfg["gt_eigs"] is not None:
                all_eigs = []
                for seed in SEEDS:
                    L = load_generator(runs_dir, method, env_name, seed, gen_idx=0)
                    if L is not None:
                        all_eigs.extend(eigenvalues(L))
                if all_eigs:
                    max_imag = max(abs(e.imag) for e in all_eigs)
                    plot_gt_eigs(ax, env_cfg["gt_eigs"], scale=max_imag)

            if row == 0:
                ax.set_title(st["label"], fontsize=11, color=st["color"])
            if col == 0:
                ax.set_ylabel(f"{env_cfg['title'].split('(')[0].strip()}\nIm(λ)", fontsize=9)

    seed_handles = [
        mpatches.Patch(color="gray", alpha=a, label=f"seed {s}")
        for s, a in zip(SEEDS, SEED_ALPHA)
    ]
    gt_handle = mpatches.Patch(color="black", label="GT SO(3)")
    fig.legend(handles=seed_handles + [gt_handle],
               loc="upper right", fontsize=8, ncol=1,
               bbox_to_anchor=(1.02, 0.98))

    fig.tight_layout()
    fname = "eigenspectra_complex_plane"
    fig.savefig(os.path.join(out_dir, f"{fname}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {fname}.pdf/png")
    plt.close(fig)


def fig_spectrum_bars(runs_dir, out_dir):
    """Sorted max|Im(λ)| per generator across seeds, as bar chart."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    fig.suptitle("Sorted |Im(λ)| of all discovered generators (state)",
                 fontsize=12, y=1.01)

    for row, (env_key, env_cfg) in enumerate(ENVS.items()):
        env_name = env_cfg["env_name"]
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            st = METHOD_STYLE[method]

            gen_scores_per_seed = {}
            for seed in SEEDS:
                Ls = load_all_generators(runs_dir, method, env_name, seed)
                for idx, L in enumerate(Ls):
                    eigs = eigenvalues(L)
                    score = np.max(np.abs(eigs.imag))
                    gen_scores_per_seed.setdefault(idx, []).append(score)

            if not gen_scores_per_seed:
                ax.set_visible(False)
                continue

            indices = sorted(gen_scores_per_seed.keys())
            means = [np.mean(gen_scores_per_seed[i]) for i in indices]
            stds  = [np.std(gen_scores_per_seed[i])  for i in indices]
            order = np.argsort(means)[::-1]
            means = [means[i] for i in order]
            stds  = [stds[i]  for i in order]
            xs = np.arange(len(means))

            ax.bar(xs, means, color=st["color"], alpha=0.8, width=0.7)
            ax.errorbar(xs, means, yerr=stds, fmt="none", color="black",
                        capsize=3, lw=1)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"G{i+1}" for i in xs], fontsize=7, rotation=45)
            ax.set_ylabel("max |Im(λ)|", fontsize=9)
            ax.tick_params(labelsize=8)

            if row == 0:
                ax.set_title(st["label"], fontsize=11, color=st["color"])
            if col == 0:
                ax.set_ylabel(f"{env_cfg['title'].split('(')[0].strip()}\nmax |Im(λ)|",
                              fontsize=9)

    fig.tight_layout()
    fname = "eigenspectra_bars"
    fig.savefig(os.path.join(out_dir, f"{fname}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {fname}.pdf/png")
    plt.close(fig)


def fig_purity(runs_dir, out_dir):
    """Rotation purity: |Im| vs |Re| of K=1 generator eigenvalues."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("K=1 generator: rotation purity  (|Im| vs |Re|  of eigenvalues)",
                 fontsize=11)

    for col, (env_key, env_cfg) in enumerate(ENVS.items()):
        env_name = env_cfg["env_name"]
        ax = axes[col]
        ax.set_title(env_cfg["title"].split("(")[0].strip(), fontsize=10)
        ax.set_xlabel("|Re(λ)|", fontsize=9)
        ax.set_ylabel("|Im(λ)|", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

        for method in METHODS:
            st = METHOD_STYLE[method]
            all_re, all_im = [], []
            for seed in SEEDS:
                L = load_generator(runs_dir, method, env_name, seed, gen_idx=0)
                if L is None:
                    continue
                eigs = eigenvalues(L)
                all_re.extend(np.abs(eigs.real))
                all_im.extend(np.abs(eigs.imag))
            if all_re:
                ax.scatter(all_re, all_im,
                           c=st["color"], marker=st["marker"],
                           s=40, alpha=0.7, edgecolors="none",
                           label=st["label"])
        ax.legend(fontsize=8)

    fig.tight_layout()
    fname = "eigenspectra_purity"
    fig.savefig(os.path.join(out_dir, f"{fname}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {fname}.pdf/png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", required=True,
                        help="Root directory of experiment runs (searched recursively).")
    parser.add_argument("--out", default="figures/")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=== Generator availability ===")
    for env_key, env_cfg in ENVS.items():
        env_name = env_cfg["env_name"]
        for method in METHODS:
            for seed in SEEDS:
                L = load_generator(args.runs_dir, method, env_name, seed)
                all_Ls = load_all_generators(args.runs_dir, method, env_name, seed)
                status = f"{len(all_Ls)} generators" if L is not None else "MISSING"
                print(f"  {env_key}/{method}/seed{seed}: {status}")

    print("\nGenerating figures...")
    fig_complex_plane(args.runs_dir, args.out)
    fig_spectrum_bars(args.runs_dir, args.out)
    fig_purity(args.runs_dir, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
