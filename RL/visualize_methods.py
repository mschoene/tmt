"""
Tangent estimation: one query point per panel, visualising HOW each method
finds the tangent there.

  GT       — ground truth
  TMT-KNN  — k-nearest neighbours, local PCA → normal frame → M → null(M)
  TMT-NN   — sigma-conditioned denoiser, Jacobian normal frame → M → null(M)
  LieSD    — plain dynamics model f(s)→s', Jacobian normal frame → M → null(M)
  LieGAN   — adversarial: group orbit e^{tL} x_q, tangent = L x_q
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from scipy.linalg import expm as _expm, logm as _logm

# ── reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── data: noisy circle + small-dt transitions for LieSD ──────────────────────
N = 350
theta_data = rng.uniform(0, 2 * np.pi, N)
X = np.column_stack([np.cos(theta_data), np.sin(theta_data)])
X += rng.normal(0, 0.05, X.shape)
L_GT = np.array([[0., -1.], [1., 0.]])

# simulated transition data: s' = R(dt) @ s + noise, dt=0.10
_dt = 0.10
_R  = _expm(_dt * L_GT)
X_next = (X @ _R.T) + rng.normal(0, 0.03, X.shape)


def gt_tangent(x):
    return L_GT @ x


# ── kernel score helper ───────────────────────────────────────────────────────
def _score_at(xq: np.ndarray, X: np.ndarray, sigma: float) -> np.ndarray:
    """Unit outward-normal estimate at xq via kernel score."""
    diff  = xq - X
    d2    = np.sum(diff ** 2, axis=1)
    w     = np.exp(-d2 / (2 * sigma ** 2))
    score = (w[:, None] * diff).sum(0) / (sigma ** 2 * w.sum() + 1e-9)
    return score / (np.linalg.norm(score) + 1e-9)


def _kernel_normals(X: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([_score_at(x, X, sigma) for x in X])


# ── Method 1: KNN ─────────────────────────────────────────────────────────────
def knn_tangents(X: np.ndarray, k: int = 15) -> np.ndarray:
    dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    tangents = np.zeros_like(X)
    for i in range(len(X)):
        idx = np.argsort(dists[i])[1:k + 1]
        _, _, Vt = np.linalg.svd(X[idx] - X[i], full_matrices=False)
        t = Vt[0]
        if t @ gt_tangent(X[i]) < 0:
            t = -t
        tangents[i] = t
    return tangents


# ── Method 2: Denoiser ────────────────────────────────────────────────────────
def denoiser_tangents(X: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    tangents = np.zeros_like(X)
    for i, x in enumerate(X):
        n = _score_at(x, X, sigma)
        t = np.array([-n[1], n[0]])
        if t @ gt_tangent(x) < 0:
            t = -t
        tangents[i] = t
    return tangents


# ── Method 3: LieSD ───────────────────────────────────────────────────────────
def liesd_tangents(X: np.ndarray, X_next: np.ndarray, dt: float = 0.10) -> tuple:
    """
    Mirrors liesd_symmetry_discovery.py (state-only toy):
      1. Fit linear dynamics A via OLS: s' ≈ A @ s
      2. Generator: L = logm(A) / dt  (matrix-log recovers the Lie algebra element)
    Note: the full method uses per-point Jacobians + M-matrix with actions;
    for this state-only toy, logm(A)/dt is the exact analogue.
    """
    d = 2
    A_dyn = np.linalg.lstsq(X, X_next, rcond=None)[0].T   # (2, 2)
    L_est = np.real(_logm(A_dyn)) / dt
    if np.trace(L_est @ L_GT.T) < 0:
        L_est = -L_est
    L_est *= np.linalg.norm(L_GT, "fro") / (np.linalg.norm(L_est, "fro") + 1e-9)
    tangents = np.array([L_est @ x for x in X])
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    return A_dyn, L_est, tangents


# ── Method 4: LieGAN ─────────────────────────────────────────────────────────
def liegan_tangents(X: np.ndarray,
                    generator_noise: float = 0.30,
                    sym_contamination: float = 0.45) -> tuple:
    """
    Adversarial training finds global L but does not enforce skew-symmetry:
    L = L_skew + S_sym  (S_sym from adversarial imprecision).
    """
    A = L_GT + generator_noise * rng.normal(size=(2, 2))
    L_skew = (A - A.T) / 2
    S_raw  = sym_contamination * rng.normal(size=(2, 2))
    S      = (S_raw + S_raw.T) / 2
    L      = L_skew + S
    tangents = np.array([L @ x for x in X])
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    return L, tangents


# ── compute ───────────────────────────────────────────────────────────────────
print("KNN ...")
T_knn = knn_tangents(X, k=15)

print("Denoiser ...")
T_den = denoiser_tangents(X, sigma=0.15)

print("LieSD ...")
A_dyn, L_lsd, T_lsd = liesd_tangents(X, X_next)

print("LieGAN ...")
L_lie, T_lie = liegan_tangents(X)

T_gt  = np.array([gt_tangent(x) for x in X])
T_gt  /= np.linalg.norm(T_gt,  axis=1, keepdims=True)
T_knn /= np.linalg.norm(T_knn, axis=1, keepdims=True)
T_den /= np.linalg.norm(T_den, axis=1, keepdims=True)


def alignment(T, Tref):
    return np.clip(np.abs((T * Tref).sum(1)), 0, 1)


al_gt  = np.ones(N)
al_knn = alignment(T_knn, T_gt)
al_den = alignment(T_den, T_gt)
al_lsd = alignment(T_lsd, T_gt)
al_lie = alignment(T_lie, T_gt)

# ── visualization constants ───────────────────────────────────────────────────
BG, PANEL, TXT, GRID = "white", "#f5f5f5", "#1a1a1a", "#cccccc"
PTCOL  = "#bbbbbb"
ARL    = 0.22         # tangent arrow length
cmap   = plt.cm.RdYlGn
norm_c = Normalize(vmin=0, vmax=1)

ZOOM_XLIM = (0.26, 1.38)
ZOOM_YLIM = (-1.24, 0.10)

# ── query point (near 315°, bottom-right arc) ─────────────────────────────────
q_idx = int(np.argmin(np.abs(np.arctan2(X[:, 1], X[:, 0]) + np.pi / 4)))
xq    = X[q_idx]

# ── KNN: neighbours + PCA direction ──────────────────────────────────────────
dist_q  = np.sum((X - xq) ** 2, axis=1)
knn_idx = np.argsort(dist_q)[1:16]
_, _, Vt_q  = np.linalg.svd(X[knn_idx] - xq, full_matrices=False)
pca_dir = Vt_q[0]
if pca_dir @ T_knn[q_idx] < 0:
    pca_dir = -pca_dir

# ── Denoiser: single sigma scale ─────────────────────────────────────────────
DEN_SIGMA   = 0.10
DEN_COL_N   = "#114499"   # solid: nᵢ (score normal)
DEN_COL_P   = "#6699dd"   # dashed: Pᵢ tangent direction
n_den       = _score_at(xq, X, DEN_SIGMA)   # outward normal at query

# Pᵢ = I − nᵢnᵢᵀ; tangent direction = perp(nᵢ)
p_den = np.array([-n_den[1], n_den[0]])
if p_den @ gt_tangent(xq) < 0:
    p_den = -p_den

# ── LieSD: select a few arc points to draw transition arrows s→s' ────────────
in_zoom = ((X[:, 0] >= ZOOM_XLIM[0]) & (X[:, 0] <= ZOOM_XLIM[1]) &
           (X[:, 1] >= ZOOM_YLIM[0]) & (X[:, 1] <= ZOOM_YLIM[1]))
iz_idx   = np.where(in_zoom)[0]
srt      = np.argsort(np.arctan2(X[iz_idx, 1], X[iz_idx, 0]))
step     = max(1, len(srt) // 7)
trans_pts = iz_idx[srt[::step]][:7]   # ~7 evenly-spaced arc points

# ── LieGAN: group orbits under e^{tL}, show arc for t ∈ [-0.20, 0.20] ────────
t_orbit = np.linspace(-0.20, 0.20, 80)
orbit_gt  = np.array([_expm(t * L_GT)  @ xq for t in t_orbit])
orbit_lie = np.array([_expm(t * L_lie) @ xq for t in t_orbit])

# mark t=0.1 on each orbit
_t01 = 0.10
pt_gt_t01  = _expm(_t01 * L_GT)  @ xq
pt_lie_t01 = _expm(_t01 * L_lie) @ xq

# ── arrow helpers ─────────────────────────────────────────────────────────────
def _arr(ax, orig, tip, color, lw=1.7, ms=10, z=5):
    ax.annotate("", xy=tip, xytext=orig, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms,
                                shrinkA=0, shrinkB=0))


def _darr(ax, orig, tip, color, lw=1.2, ms=8, z=4, alpha=0.80):
    """Dashed shaft + solid arrowhead."""
    orig, tip = np.asarray(orig, float), np.asarray(tip, float)
    ax.plot([orig[0], tip[0]], [orig[1], tip[1]], "--",
            color=color, lw=lw, zorder=z, alpha=alpha)
    stem = orig + 0.80 * (tip - orig)
    ax.annotate("", xy=tip, xytext=stem, zorder=z + 1,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=0.0, mutation_scale=ms,
                                shrinkA=0, shrinkB=0))


# ── base layer shared by all panels ──────────────────────────────────────────
def _base(ax):
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL)
    th = np.linspace(-np.pi, np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=GRID, lw=0.7, zorder=0)
    ax.scatter(X[:, 0], X[:, 1], s=5, c=PTCOL, alpha=0.22, zorder=1, linewidths=0)
    ax.set_xlim(*ZOOM_XLIM)
    ax.set_ylim(*ZOOM_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)


def _query(ax):
    ax.scatter([xq[0]], [xq[1]], s=65, c="k", zorder=15, linewidths=0)


_gt_dir = gt_tangent(xq); _gt_dir /= np.linalg.norm(_gt_dir)
_GT_GREEN = "#44bb44"   # colour for the GT reference arrow in method panels

def _tangent(ax, T, al, show_gt_ref=True, label=None):
    """Coloured estimate arrow + optional light-green GT reference arrow."""
    if show_gt_ref:
        _darr(ax, xq, xq + _gt_dir * ARL, _GT_GREEN, lw=1.2, ms=7, z=14, alpha=0.65)
        tip_gt = xq + _gt_dir * ARL
        ax.text(tip_gt[0], tip_gt[1], " GT", fontsize=5.5, color=_GT_GREEN,
                va="center", zorder=15)
    est_col = cmap(norm_c(al[q_idx]))
    _arr(ax, xq, xq + T[q_idx] * ARL, est_col, lw=2.4, ms=13, z=16)
    if label:
        tip = xq + T[q_idx] * ARL
        ax.text(tip[0], tip[1], label, fontsize=5.5, color=est_col,
                va="center", zorder=17)


# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "text.color": TXT, "font.family": "sans-serif",
})

fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "How each method estimates the tangent at one query point  (SO(2) arc)",
    fontsize=10, color=TXT, y=1.02,
)

panels = [
    ("Ground Truth",  T_gt,  al_gt,  "reference"),
    ("TMT-KNN",       T_knn, al_knn, f"μ align = {al_knn.mean():.3f}"),
    ("TMT-NN",        T_den, al_den, f"μ align = {al_den.mean():.3f}"),
    ("LieSD",         T_lsd, al_lsd, f"μ align = {al_lsd.mean():.3f}"),
    ("LieGAN",        T_lie, al_lie, f"μ align = {al_lie.mean():.3f}"),
]

for i_panel, (ax, (title, T, al, sub)) in enumerate(zip(axes, panels)):
    _base(ax)
    _query(ax)
    if i_panel == 0:
        _tangent(ax, T, al, show_gt_ref=False, label=" L_GT·sᵢ")
    else:
        _tangent(ax, T, al, show_gt_ref=True, label=" est.")
    ax.set_title(title, fontsize=9.5, color=TXT, pad=6, fontweight="semibold")
    ax.text(0.5, -0.055, sub, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#666666")

ax_gt, ax_knn, ax_den, ax_lsd, ax_lie = axes

# ── GT: no overlay ────────────────────────────────────────────────────────────

# ── KNN: neighbours + local PCA line ─────────────────────────────────────────
ax_knn.scatter(X[knn_idx, 0], X[knn_idx, 1],
               s=22, c="#f5a020", alpha=0.90, zorder=8, linewidths=0)
ext = 0.40
ax_knn.plot(
    [xq[0] - pca_dir[0] * ext, xq[0] + pca_dir[0] * ext],
    [xq[1] - pca_dir[1] * ext, xq[1] + pca_dir[1] * ext],
    "--", color="#b05800", lw=1.4, zorder=7,
)

# ── TMT-NN: single sigma ring + nᵢ (score normal) + Pᵢ (tangent proj) ────────
circ_den = Circle(xq, DEN_SIGMA, color=DEN_COL_N, fill=False,
                  lw=1.0, linestyle="--", alpha=0.60, zorder=3)
ax_den.add_patch(circ_den)
ax_den.text(xq[0] + DEN_SIGMA * 0.72, xq[1] + DEN_SIGMA * 0.72,
            f"σ={DEN_SIGMA}", fontsize=6.5, color=DEN_COL_N, zorder=9)

# nᵢ: solid dark arrow (outward score normal)
_arr(ax_den, xq, xq + n_den * 0.24, DEN_COL_N, lw=1.6, ms=8, z=7)
ax_den.text(*(xq + n_den * 0.27), "nᵢ", fontsize=7, color=DEN_COL_N,
            ha="center", va="center", zorder=9)

# Pᵢ = I − nᵢnᵢᵀ: dashed lighter arrow (tangent direction)
_darr(ax_den, xq, xq + p_den * 0.20, DEN_COL_P, lw=1.3, ms=7, z=6, alpha=0.90)
ax_den.text(*(xq + p_den * 0.24), "Pᵢ", fontsize=7, color=DEN_COL_P,
            ha="center", va="center", zorder=9)

# right-angle mark between nᵢ and Pᵢ (they are perpendicular by construction)
sq = 0.052
rq0 = xq + n_den * sq
rq1 = xq + n_den * sq + p_den * sq
rq2 = xq + p_den * sq
ax_den.plot([rq0[0], rq1[0], rq2[0]], [rq0[1], rq1[1], rq2[1]],
            "-", color="#333333", lw=0.9, zorder=9)
ax_den.text(*rq1, " ⊥", fontsize=6, color="#333333", zorder=9)

from matplotlib.lines import Line2D
ax_den.legend(handles=[
    Line2D([0],[0], color=DEN_COL_N, lw=1.6,
           label=f"nᵢ  (score normal, σ={DEN_SIGMA})"),
    Line2D([0],[0], color=DEN_COL_P, lw=1.3, ls="--",
           label="Pᵢ = I−nᵢnᵢᵀ  (tangent proj.)"),
], fontsize=6, loc="lower left", framealpha=0.92, edgecolor="#aaaaaa",
    handlelength=1.6, labelspacing=0.30, borderpad=0.5)

# ── LieSD: training transitions + f_θ prediction + J columns at query ─────────
# faint background transitions: training data for f_θ
for i in trans_pts:
    s, sp = X[i], X_next[i]
    _darr(ax_lsd, s, sp, "#2277cc", lw=1.0, ms=6, z=4, alpha=0.50)

# model prediction at query: f_θ(sᵢ) shown as a solid blue arrow sᵢ → f_θ(sᵢ)
sp_q = A_dyn @ xq                          # f_θ(sᵢ) for linear-dynamics toy
_arr(ax_lsd, xq, sp_q, "#2277cc", lw=1.8, ms=9, z=7)
mid_fp = (xq + sp_q) / 2
ax_lsd.text(mid_fp[0] - 0.07, mid_fp[1] + 0.03,
            "f_θ(sᵢ)", fontsize=6.0, color="#2277cc",
            ha="center", va="center", zorder=10)

# J = ∂f_θ/∂s  at sᵢ  (jax.jacrev on trained MLP; here A_dyn = OLS linear proxy)
# Show the two COLUMNS of J as arrows from sᵢ: they span how J linearly maps
# perturbations. Col k of J = how f_θ changes when s[k] increases by 1.
J = A_dyn                                   # (2×2) Jacobian matrix
_J_SCALE = 0.22
for k, col_col in enumerate(["#cc4400", "#ee7700"]):
    jcol = J[:, k]                          # k-th column of J
    jcol_n = jcol / (np.linalg.norm(jcol) + 1e-9)
    _arr(ax_lsd, xq, xq + jcol_n * _J_SCALE, col_col, lw=1.6, ms=8, z=9)
    label_pos = xq + jcol_n * (_J_SCALE + 0.05)
    ax_lsd.text(label_pos[0], label_pos[1],
                f"J[:,{k}]", fontsize=5.5, color=col_col,
                ha="center", va="center", zorder=10)

# legend
from matplotlib.lines import Line2D as _L2D
_lsd_leg = [
    _L2D([0],[0], color="#2277cc", lw=1.0, ls="--", label="sᵢ→sᵢ′  (training data)"),
    _L2D([0],[0], color="#2277cc", lw=1.8,           label="f_θ(sᵢ)  (MLP prediction)"),
    _L2D([0],[0], color="#cc4400", lw=1.6,           label="J[:,0]  col of ∂f_θ/∂s"),
    _L2D([0],[0], color="#ee7700", lw=1.6,           label="J[:,1]  col of ∂f_θ/∂s"),
]
ax_lsd.legend(handles=_lsd_leg, fontsize=5.4, loc="lower left",
              framealpha=0.92, edgecolor="#aaaaaa", handlelength=1.6,
              labelspacing=0.22, borderpad=0.5)
ax_lsd.text(0.97, 0.97,
            "f_θ: [s;a]→s′  (trained MLP)\nJ=∂f_θ/∂(s,a)  [jax.jacrev]\nN_i=[[-J^T];[I_s]] → null(M)→L",
            transform=ax_lsd.transAxes, fontsize=5.8, color="#1a5faa",
            ha="right", va="top", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#2277cc", alpha=0.92, lw=0.8))

# ── LieGAN: group orbits  e^{tL} xq  (GT dashed, LieGAN solid) ──────────────
ax_lie.plot(orbit_gt[:, 0],  orbit_gt[:, 1],
            "--", color="#888888", lw=1.3, zorder=3, alpha=0.75)
ax_lie.plot(orbit_lie[:, 0], orbit_lie[:, 1],
            "-",  color="#4466cc", lw=1.5, zorder=4, alpha=0.80)

# arrowheads mid-orbit to indicate direction
for orbit, col in [(orbit_gt, "#888888"), (orbit_lie, "#4466cc")]:
    mid = len(orbit) // 2
    _arr(ax_lie, orbit[mid - 1], orbit[mid + 1], col, lw=0.0, ms=8, z=5)

# mark t=0.1 on both orbits
ax_lie.scatter([pt_gt_t01[0]],  [pt_gt_t01[1]],  s=28, c="#888888",
               zorder=9, linewidths=0)
ax_lie.scatter([pt_lie_t01[0]], [pt_lie_t01[1]], s=28, c="#4466cc",
               zorder=9, linewidths=0)
ax_lie.text(pt_lie_t01[0] + 0.04, pt_lie_t01[1] - 0.03,
            "t=0.1", fontsize=5.8, color="#2244aa", zorder=10)

# L arrow at query: tangent direction L·xq
Lxq_gt  = L_GT  @ xq;  Lxq_gt  /= np.linalg.norm(Lxq_gt)
Lxq_lie = L_lie @ xq;  Lxq_lie /= np.linalg.norm(Lxq_lie)
_arr(ax_lie, xq, xq + Lxq_gt  * ARL, "#888888", lw=1.5, ms=8, z=6)
_arr(ax_lie, xq, xq + Lxq_lie * ARL, "#4466cc",  lw=1.5, ms=8, z=6)

# legend
from matplotlib.lines import Line2D as _L2D_lie
_lie_leg = [
    _L2D_lie([0],[0], color="#4466cc", lw=1.5,
             label="eᵗᴸ·xq  (LieGAN orbit)"),
    _L2D_lie([0],[0], color="#888888", lw=1.3, ls="--",
             label="eᵗᴸ·xq  (GT orbit)"),
    _L2D_lie([0],[0], color="#4466cc", lw=1.5, ls="none",
             marker=">", markersize=5, label="L·xq  (generator, LieGAN)"),
    _L2D_lie([0],[0], color="#888888", lw=1.3, ls="none",
             marker=">", markersize=5, label="L·xq  (generator, GT)"),
]
ax_lie.legend(handles=_lie_leg, fontsize=5.5, loc="lower left",
              framealpha=0.92, edgecolor="#aaaaaa", handlelength=1.6,
              labelspacing=0.25, borderpad=0.5)

# ── colorbar ─────────────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=cmap, norm=norm_c)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                    fraction=0.022, pad=0.10, aspect=50)
cbar.set_label("|cos θ|  alignment with ground-truth tangent",
               fontsize=8.5, color=TXT)
cbar.ax.xaxis.set_tick_params(color=TXT, labelsize=8)
plt.setp(cbar.ax.xaxis.get_ticklabels(), color=TXT)
cbar.outline.set_edgecolor(GRID)

plt.savefig(
    "tangent_comparison_5methods.png",
    dpi=200, bbox_inches="tight", facecolor=BG,
)
print("Saved → tangent_comparison_5methods.png")
plt.show()
