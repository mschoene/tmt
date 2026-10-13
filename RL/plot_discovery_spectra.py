"""
Plot sorted eigenvalue spectra of the continuous symmetry-discovery matrix (M_TMT).

One figure per environment: spectrum (top row) + ratio profile (bottom row).

Two gap-detection modes, each generating its own set of figures:
  baseline  — fixed thresholds: primary=2.0×, fallback=1.5×
  adaptive  — data-driven threshold via MAD of log-ratios

Usage:
    python plot_discovery_spectra.py --runs_dir /path/to/experiment/runs [--out figures/] [--mode baseline|adaptive|both]

Expected layout under --runs_dir (searched recursively):
    {runs_dir}/**/tmt_nn/**/{env}**/seed{N}/**/snapshots/continuous_spectrum.npy
    {runs_dir}/**/tmt_knn/**/{env}**/seed{N}/**/snapshots/continuous_spectrum.npy
    {runs_dir}/**/liesd/**/{env}**/seed{N}/**/snapshots/continuous_spectrum.npy
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

METHOD_STYLE = {
    "tmt_nn":  dict(color="#4CAF50", label="TMT-NN"),
    "tmt_knn": dict(color="#FF9800", label="TMT-KNN"),
    "liesd":   dict(color="#9C27B0", label="LieSD"),
}
SEEDS      = [42, 43, 44]
SEED_ALPHA = [0.9, 0.60, 0.35]

MODES = {
    "baseline": dict(
        primary_thresh  = 2.0,
        fallback_thresh = 1.5,
    ),
    "adaptive": dict(
        k_primary   = 3.5,  _floor_primary   = 1.5,
        k_fallback  = 2.5,  _floor_fallback  = 1.3,
    ),
}

ENVS = [
    {
        "key":      "particle2d",
        "env_name": "ParticleCircle2D-v0",
        "title":    "Particle2D  (SO(2))",
    },
    {
        "key":      "particle3d",
        "env_name": "ParticleCircle3D-v0",
        "title":    "Particle3D  (SO(3))",
    },
    {
        "key":      "reacher",
        "env_name": "Reacher-v5",
        "title":    "Reacher-v5",
    },
]


# ── gap finding ───────────────────────────────────────────────────────────────

def _adaptive_threshold(ratios, k_sigma, abs_floor):
    lr  = np.log(np.maximum(ratios, 1e-9))
    med = float(np.median(lr))
    mad = float(np.median(np.abs(lr - med))) * 1.4826
    return max(float(np.exp(med + k_sigma * mad)), abs_floor)


def _scan_gap(ratios, threshold):
    for i, r in enumerate(ratios):
        if r >= threshold:
            return i + 1, float(r)
    return 0, float(ratios.max())


def find_first_gap_k(spec, mode="adaptive"):
    spec   = np.asarray(spec, dtype=float)
    if len(spec) < 2:
        return 0, 0.0, False
    ratios = spec[1:] / np.maximum(spec[:-1], 1e-9)
    cfg    = MODES[mode]

    if mode == "baseline":
        k, r = _scan_gap(ratios, cfg["primary_thresh"])
        return k, r, False

    tp  = _adaptive_threshold(ratios, cfg["k_primary"],  cfg["_floor_primary"])
    tfb = _adaptive_threshold(ratios, cfg["k_fallback"], cfg["_floor_fallback"])

    k, r = _scan_gap(ratios, tp)
    if k > 0:
        return k, r, False
    k, r = _scan_gap(ratios, tfb)
    return k, r, (k > 0)


def gap_thresholds(spec, mode):
    cfg    = MODES[mode]
    ratios = np.asarray(spec, dtype=float)
    ratios = ratios[1:] / np.maximum(ratios[:-1], 1e-9)
    if mode == "baseline":
        return cfg["primary_thresh"], None
    return (
        _adaptive_threshold(ratios, cfg["k_primary"],  cfg["_floor_primary"]),
        _adaptive_threshold(ratios, cfg["k_fallback"], cfg["_floor_fallback"]),
    )


def all_ratios(spec):
    spec = np.asarray(spec, dtype=float)
    return spec[1:] / np.maximum(spec[:-1], 1e-9)


# ── data loading ──────────────────────────────────────────────────────────────

def _env_in_path(path: str, env_name: str) -> bool:
    norm = env_name.replace("-", "").replace("_", "").lower()
    return norm in path.replace("-", "").replace("_", "").lower()


def _glob_spectra(runs_dir, method, env_name, seed):
    pattern = os.path.join(runs_dir, "**", "snapshots", "continuous_spectrum.npy")
    return [
        f for f in glob.glob(pattern, recursive=True)
        if method in f and _env_in_path(f, env_name) and f"seed{seed}" in f
    ]


def load_all_seed_spectra(runs_dir, method, env_name, mode):
    results = []
    for seed in SEEDS:
        files = _glob_spectra(runs_dir, method, env_name, seed)
        if not files:
            results.append(None)
            continue
        best, best_r = None, -1.0
        for f in files:
            spec = np.load(f)
            k, r, fb = find_first_gap_k(spec, mode)
            if r > best_r:
                best_r = r
                best   = (spec, k, r, fb)
        results.append(best)
    return results


# ── panel plotting ────────────────────────────────────────────────────────────

def _best_item(seed_data):
    valid = [x for x in seed_data if x is not None]
    return max(valid, key=lambda x: x[2]) if valid else None


def plot_spectrum_panel(ax, seed_data, color):
    for i, item in enumerate(seed_data):
        if item is None:
            continue
        spec, k, _r, fb = item
        ax.semilogy(np.arange(len(spec)), np.maximum(spec, 1e-9),
                    color=color, alpha=SEED_ALPHA[i], lw=1.5)
        if k > 0:
            ax.axvline(k - 0.5, color=color,
                       alpha=SEED_ALPHA[i] * (0.5 if fb else 0.8),
                       ls="-." if fb else "--", lw=1.0)

    best = _best_item(seed_data)
    if best is None:
        return
    spec_b, k_b, r_b, fb_b = best
    if 0 < k_b <= len(spec_b):
        ax.axvspan(0, k_b - 0.5, color=color, alpha=0.07 if fb_b else 0.12)
        ratio_str = f"{r_b:.0f}×" if r_b >= 10 else f"{r_b:.1f}×"
        ax.text(k_b + 0.3, 0.92,
                f"K={k_b}\n({ratio_str})" + (" *" if fb_b else ""),
                transform=ax.get_xaxis_transform(),
                color=color, fontsize=8, va="top", ha="left",
                fontweight="bold" if r_b >= 5.0 else "normal",
                fontstyle="italic" if fb_b else "normal")

    ax.set_ylabel("Eigenvalue", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, which="both")
    ax.set_xlim(left=-0.5)


def plot_ratio_panel(ax, seed_data, color, mode):
    thresh_lines_drawn = set()

    for i, item in enumerate(seed_data):
        if item is None:
            continue
        spec, k, _r, fb = item
        ratios = all_ratios(spec)
        ax.semilogy(np.arange(len(ratios)), ratios,
                    color=color, alpha=SEED_ALPHA[i], lw=1.2)
        if k > 0:
            ax.axvline(k - 0.5, color=color,
                       alpha=SEED_ALPHA[i] * (0.5 if fb else 0.8),
                       ls="-." if fb else "--", lw=1.0)

        t_p, t_fb = gap_thresholds(spec, mode)
        for t, ls in [(t_p, "--"), (t_fb, "-.")]:
            if t is None:
                continue
            key = round(t, 2)
            if key not in thresh_lines_drawn:
                ax.axhline(t, color="black", ls=ls, lw=0.8, alpha=0.5)
                thresh_lines_drawn.add(key)

    ax.set_ylabel("Ratio", fontsize=9)
    ax.set_xlabel("Sorted index", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, which="both")
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=1.05)


# ── per-environment figure ────────────────────────────────────────────────────

def make_env_figure(runs_dir, env, out_dir, mode):
    methods = list(METHOD_STYLE.keys())
    n_cols  = len(methods)
    env_name = env["env_name"]

    fig = plt.figure(figsize=(4.5 * n_cols, 7.5))
    gs  = gridspec.GridSpec(2, n_cols,
                            height_ratios=[3, 2],
                            hspace=0.08, wspace=0.32)

    ax_specs, ax_ratios = [], []
    for col in range(n_cols):
        ax_s = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col], sharex=ax_s,
                               **({"sharey": ax_ratios[0]} if col > 0 else {}))
        ax_specs.append(ax_s)
        ax_ratios.append(ax_r)

    for col, method in enumerate(methods):
        st        = METHOD_STYLE[method]
        seed_data = load_all_seed_spectra(runs_dir, method, env_name, mode)

        for si, item in enumerate(seed_data):
            if item is not None:
                _, k, r, fb = item
                print(f"  {env['key']}/{method}/seed{SEEDS[si]}: "
                      f"K={k}, ratio={r:.1f}×" + (" (fallback)" if fb else ""))

        plot_spectrum_panel(ax_specs[col], seed_data, st["color"])
        plot_ratio_panel   (ax_ratios[col], seed_data, st["color"], mode)

        ax_specs[col].set_title(st["label"], fontsize=12, color=st["color"])
        plt.setp(ax_specs[col].get_xticklabels(), visible=False)
        if col > 0:
            ax_specs[col].set_ylabel("")
            ax_ratios[col].set_ylabel("")

    ax_ratios[0].set_ylim(bottom=1.05)

    seed_handles = [
        Line2D([0], [0], color="gray", alpha=a, lw=1.5, label=f"seed {s}")
        for s, a in zip(SEEDS, SEED_ALPHA)
    ] + [
        Line2D([0], [0], color="black", ls="--", lw=1.0, label="gap"),
    ]
    if mode != "baseline":
        seed_handles += [
            Line2D([0], [0], color="black", ls="-.", lw=0.9, label="gap (fallback)"),
        ]
    ax_specs[-1].legend(handles=seed_handles, fontsize=7.5, loc="lower right")

    fig.suptitle(f"{env['title']}  [{mode}]", fontsize=12, y=1.01)
    fname = f"discovery_spectra_{env['key']}_{mode}"
    fig.savefig(os.path.join(out_dir, f"{fname}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {fname}.pdf/png")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", required=True,
                        help="Root directory of experiment runs (searched recursively).")
    parser.add_argument("--out",  default="figures/")
    parser.add_argument("--mode", choices=["baseline", "adaptive", "both"],
                        default="baseline")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    modes = list(MODES.keys()) if args.mode == "both" else [args.mode]
    for mode in modes:
        for env in ENVS:
            print(f"\n=== {env['key']} [{mode}] ===")
            make_env_figure(args.runs_dir, env, args.out, mode)


if __name__ == "__main__":
    main()
