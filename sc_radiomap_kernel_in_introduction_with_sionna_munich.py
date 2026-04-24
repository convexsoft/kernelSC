import os
import sys
import importlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


# =============================================================================
# 0.  Path setup
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT   = SCRIPT_DIR / Path(__file__).stem
DATA_DIR   = OUT_ROOT / "data"
FIG_DIR    = OUT_ROOT / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1.  Locate Sionna RT assets
# =============================================================================
def find_sionna_assets():
    try:
        import sionna
        import sionna.rt as rt
    except ImportError:
        print("\n[ERROR] sionna-rt is not installed.")
        print("        Install it with:  pip install sionna-rt plyfile numpy matplotlib scipy")
        sys.exit(1)

    sionna_root = Path(sionna.__file__).resolve().parent
    scene_xml   = sionna_root / "rt" / "scenes" / "munich" / "munich.xml"
    mesh_dir    = sionna_root / "rt" / "scenes" / "munich" / "meshes"

    if not scene_xml.exists():
        print(f"\n[ERROR] Munich scene not found at: {scene_xml}")
        print("        Try reinstalling:  pip install --force-reinstall sionna-rt")
        sys.exit(1)

    return sionna, rt, scene_xml, mesh_dir


# =============================================================================
# 2.  Sionna RT data generation
# =============================================================================
CARRIER_FREQ  = 3.5e9    # Hz  (5G FR1 / 6G candidate band)
TX_POWER_DBM  = 30.0     # dBm
TX_POSITION   = [8.5, 21.0, 30.0]   # [x, y, z] metres
N_SPARSE      = 40       # number of sparse measurement samples
CELL_SIZE_M   = 5.0      # radio map resolution in metres
SAMPLES_PER_TX = int(5e5)  # ray samples (reduce to 2e5 if GPU RAM < 8 GB)
MAX_DEPTH     = 5        # max reflection/diffraction depth

_RNG = np.random.default_rng(42)


def generate_data(sionna, rt, scene_xml: Path):
    from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver

    print(f"[Sionna RT v{rt.__version__}] Loading Munich scene ...")
    scene           = load_scene(str(scene_xml))
    scene.frequency = CARRIER_FREQ

    # Antenna arrays must be set at scene level in Sionna RT v2.x
    _ant = dict(num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="dipole", polarization="V")
    scene.tx_array = PlanarArray(**_ant)
    scene.rx_array = PlanarArray(**_ant)

    # Transmitter  (rooftop base station)
    tx           = Transmitter(name="bs0", position=TX_POSITION)
    tx.power_dbm = TX_POWER_DBM
    scene.add(tx)
    print(f"[TX] position={TX_POSITION}, power={TX_POWER_DBM} dBm, "
          f"freq={CARRIER_FREQ/1e9:.1f} GHz")

    # Radio map
    print(f"[Solver] Computing radio map "
          f"(cell={CELL_SIZE_M} m, samples={SAMPLES_PER_TX:.0e}) ...")
    print("         This may take 1-3 min on GPU, longer on CPU.")
    solver    = RadioMapSolver()
    radio_map = solver(
        scene,
        cell_size      = [CELL_SIZE_M, CELL_SIZE_M],
        samples_per_tx = SAMPLES_PER_TX,
        max_depth      = MAX_DEPTH,
        diffuse_reflection = True,
    )
    print("[Solver] Done.")

    # cell_centers: [H, W, 3]  world coords of each map cell
    cc   = np.array(radio_map.cell_centers)   # [H, W, 3]
    x_2d = cc[:, :, 0]
    y_2d = cc[:, :, 1]

    # path_gain: [1, H, W]  linear, unitless
    pg   = np.array(radio_map.path_gain)[0]   # [H, W]
    pg   = np.where(pg > 0, pg, np.nan)
    path_loss_db = -10.0 * np.log10(pg)       # dB (positive)
    rss_dbm      = TX_POWER_DBM - path_loss_db # dBm

    x_min, x_max = float(x_2d.min()), float(x_2d.max())
    y_min, y_max = float(y_2d.min()), float(y_2d.max())

    print(f"[Map] shape={rss_dbm.shape}, "
          f"RSS=[{np.nanmin(rss_dbm):.1f}, {np.nanmax(rss_dbm):.1f}] dBm")
    print(f"[Map] x=[{x_min:.0f}, {x_max:.0f}] m  "
          f"y=[{y_min:.0f}, {y_max:.0f}] m")

    # Sparse samples
    H, W    = rss_dbm.shape
    valid   = np.argwhere(np.isfinite(rss_dbm))
    chosen  = _RNG.choice(len(valid), size=min(N_SPARSE, len(valid)), replace=False)
    pixels  = valid[chosen]
    sx      = x_2d[pixels[:, 0], pixels[:, 1]]
    sy      = y_2d[pixels[:, 0], pixels[:, 1]]
    s_locs  = np.stack([sx, sy], axis=1)
    s_vals  = rss_dbm[pixels[:, 0], pixels[:, 1]]

    # Save
    np.save(DATA_DIR / "rss_map.npy",       rss_dbm)
    np.save(DATA_DIR / "path_loss_map.npy", path_loss_db)
    np.savez(DATA_DIR / "meta.npz",
             x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
             carrier_freq=CARRIER_FREQ, tx_power_dbm=TX_POWER_DBM,
             tx_position=np.array(TX_POSITION), cell_size=CELL_SIZE_M,
             scene="munich")
    np.savez(DATA_DIR / "sparse_samples.npz",
             locations=s_locs, values=s_vals)

    print(f"[Saved] Data written to: {DATA_DIR}")
    return rss_dbm, path_loss_db, s_locs, s_vals, x_min, x_max, y_min, y_max


def load_data():
    rss    = np.load(DATA_DIR / "rss_map.npy")
    meta   = np.load(DATA_DIR / "meta.npz")
    sparse = np.load(DATA_DIR / "sparse_samples.npz")
    return (rss,
            np.load(DATA_DIR / "path_loss_map.npy"),
            sparse["locations"], sparse["values"],
            float(meta["x_min"]), float(meta["x_max"]),
            float(meta["y_min"]), float(meta["y_max"]))


# =============================================================================
# 3.  Munich building footprints
# =============================================================================
def load_footprints(mesh_dir: Path):
    try:
        from plyfile import PlyData
    except ImportError:
        print("\n[ERROR] plyfile is not installed.")
        print("        Install it with:  pip install plyfile")
        sys.exit(1)

    all_ply = sorted(mesh_dir.glob("*.ply"))
    bnames  = list(set([f.stem.rsplit("-itu_", 1)[0] for f in all_ply]))
    boxes   = []

    for bname in bnames:
        xs, ys = [], []
        for pf in mesh_dir.glob(f"{bname}-itu_*.ply"):
            try:
                pd = PlyData.read(str(pf))
                v  = pd["vertex"]
                xs.extend(np.array(v["x"]))
                ys.extend(np.array(v["y"]))
            except Exception:
                pass
        if xs:
            xa, ya = np.array(xs), np.array(ys)
            boxes.append((xa.min(), ya.min(),
                          xa.max() - xa.min(),
                          ya.max() - ya.min()))

    print(f"[Buildings] {len(boxes)} footprints loaded from {mesh_dir.name}/")
    return np.array(boxes)


# =============================================================================
# 4.  Figure layout constants
# =============================================================================
plt.rcParams["pdf.fonttype"]   = 42
plt.rcParams["ps.fonttype"]    = 42
plt.rcParams["font.family"]    = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 0.75

TX_POS     = np.array([TX_POSITION[0], TX_POSITION[1]])
MAP_ASPECT = 0.817    # H/W of Munich radio map (1205 m / 1475 m)
FIG_HEIGHT = 4.4      # inches — uniform height for all four panels
MAP_W      = FIG_HEIGHT / MAP_ASPECT   # ≈ 5.39 in  (map panels a, b)
PLOT_W     = 5.0                       # in  (plot panels c, d)


# =============================================================================
# 5.  Panel drawing functions
# =============================================================================
def draw_panel_a(ax, rss, sparse_locs, sparse_vals,
                 x_min, x_max, y_min, y_max, footprints):
    ax.set_title("(a) Spectrum Cartography",
                 fontsize=14, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    rss_disp = np.where(np.isfinite(rss), rss, np.nanmin(rss))
    im = ax.imshow(rss_disp,
                   extent=[x_min, x_max, y_min, y_max],
                   origin="lower", cmap="YlOrRd_r",
                   vmin=np.nanpercentile(rss, 2),
                   vmax=np.nanpercentile(rss, 98),
                   interpolation="bilinear", alpha=0.82, zorder=1)

    if footprints is not None:
        for (bx, by, bw, bh) in footprints:
            if bx+bw < x_min or bx > x_max or by+bh < y_min or by > y_max:
                continue
            ax.add_patch(Rectangle((bx, by), bw, bh,
                                   facecolor="#c8cfd6", edgecolor="#8a939e",
                                   linewidth=0.35, alpha=0.70, zorder=2))

    norm_v = ((sparse_vals - np.nanmin(sparse_vals)) /
              (np.nanmax(sparse_vals) - np.nanmin(sparse_vals) + 1e-9))
    ax.scatter(sparse_locs[:, 0], sparse_locs[:, 1],
               s=26, c=plt.cm.plasma(norm_v),
               edgecolors="white", linewidths=0.6, zorder=4)

    ax.scatter(TX_POS[0], TX_POS[1], s=120, marker="^",
               color="#e31a1c", edgecolors="white",
               linewidths=0.8, zorder=5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label("RSS (dBm)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8.5)

    leg = [
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#e31a1c", markeredgecolor="white",
               markersize=8, label="Base station"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#9b59b6", markeredgecolor="white",
               markersize=6, label=fr"Samples ($n={len(sparse_locs)}$)"),
        mpatches.Patch(facecolor="#c8cfd6", edgecolor="#8a939e",
                       linewidth=0.5, label="Buildings"),
    ]
    ax.legend(handles=leg, fontsize=8.5, loc="lower left",
              framealpha=0.88, edgecolor="0.6",
              handlelength=1.2, borderpad=0.4, labelspacing=0.3)

    ax.text(0.98, 0.98, "NVIDIA Sionna RT\nMunich, 3.5 GHz",
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec="0.6", alpha=0.90), zorder=6)


def draw_panel_b(ax, rss, sparse_locs,
                 x_min, x_max, y_min, y_max, footprints):
    ax.set_title("(b) Radio Map Reconstruction",
                 fontsize=14, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    rss_disp = np.where(np.isfinite(rss), rss, np.nanmin(rss))
    im = ax.imshow(rss_disp,
                   extent=[x_min, x_max, y_min, y_max],
                   origin="lower", cmap="turbo",
                   vmin=np.nanpercentile(rss, 2),
                   vmax=np.nanpercentile(rss, 98),
                   interpolation="bilinear", zorder=1)

    H, W = rss_disp.shape
    xg   = np.linspace(x_min, x_max, W)
    yg   = np.linspace(y_min, y_max, H)
    xx, yy = np.meshgrid(xg, yg)
    lvls = np.linspace(np.nanpercentile(rss, 8),
                       np.nanpercentile(rss, 92), 8)
    ax.contour(xx, yy, rss_disp, levels=lvls,
               colors="white", linewidths=0.5, alpha=0.60, zorder=2)

    if footprints is not None:
        for (bx, by, bw, bh) in footprints:
            if bx+bw < x_min or bx > x_max or by+bh < y_min or by > y_max:
                continue
            ax.add_patch(Rectangle((bx, by), bw, bh,
                                   fill=False, edgecolor="white",
                                   linewidth=0.28, alpha=0.40, zorder=3))

    ax.scatter(sparse_locs[:, 0], sparse_locs[:, 1],
               s=16, c="white", edgecolors="black",
               linewidths=0.35, zorder=4)
    ax.scatter(TX_POS[0], TX_POS[1], s=100, marker="*",
               color="#e31a1c", edgecolors="white",
               linewidths=0.5, zorder=5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label("RSS (dBm)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8.5)

    ax.text(0.98, 0.98, "Continuous field\nfrom sparse samples",
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec="0.6", alpha=0.90), zorder=6)


def draw_panel_c(ax, rss, x_min, x_max, y_min, y_max):
    ax.set_title("(c) Kernel Regression",
                 fontsize=14, fontweight="bold", pad=8)

    H, W  = rss.shape
    row   = 149
    prof  = rss[row, :]
    x_w   = np.linspace(x_min, x_max, W)

    ok     = np.isfinite(prof)
    x_f    = x_w[ok]
    y_f    = prof[ok]
    y_true = gaussian_filter(y_f, sigma=3.5)

    n_pts = 12
    idx_s = np.round(np.linspace(0, len(x_f) - 1, n_pts)).astype(int)
    x_s   = x_f[idx_s]
    y_s   = y_true[idx_s] + _RNG.normal(0, 0.9, n_pts)

    sigma_k = (x_f[-1] - x_f[0]) / 8.0
    lam     = 6e-3
    K       = np.exp(-0.5 * ((x_s[:, None] - x_s[None, :]) / sigma_k)**2)
    alpha   = np.linalg.solve(K + lam * np.eye(n_pts), y_s)
    K_pred  = np.exp(-0.5 * ((x_f[:, None] - x_s[None, :]) / sigma_k)**2)
    y_est   = K_pred @ alpha

    ax.plot(x_f, y_true, color="#2c2c2c", lw=2.0,
            label=r"True $f(\mathbf{x})$")
    ax.plot(x_f, y_est, color="#1b9e77", lw=2.0,
            linestyle="--", label=r"Estimate $\hat{f}(\mathbf{x})$")
    ax.scatter(x_s, y_s, color="#2166ac", s=28, zorder=5,
               label=fr"Samples ($n={n_pts}$)")


    xi   = x_s[1]
    xj   = x_s[5]
    y_lo = y_true.min() - 5.5
    mid  = (xi + xj) / 2.0
    ax.annotate("", xy=(xj, y_lo), xytext=(xi, y_lo),
                arrowprops=dict(arrowstyle="<->", color="#555555",
                                lw=1.1, mutation_scale=10))
    # ax.text(mid, y_lo - 2.0, r"$k(\mathbf{x}_i,\mathbf{x}_j)$",
    #         ha="center", va="top", fontsize=9.5, color="#444444")

    ax.set_xlabel("Location $x$ (m)", fontsize=11.5)
    ax.set_ylabel("RSS (dBm)", fontsize=11.5)
    ax.grid(alpha=0.20, linewidth=0.5)


    ax.legend(fontsize=9, frameon=True, loc="lower right",
              framealpha=0.92, edgecolor="0.6",
              handlelength=1.6, labelspacing=0.35, borderpad=0.45)


    ax.text(0.03, 0.97,
            r"$f(\mathbf{x})=\sum_{i}\alpha_i k(\mathbf{x},\mathbf{x}_i)$",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.32", fc="#f7f7f7",
                      ec="0.65", alpha=0.93))


def draw_panel_d(ax):
    ax.set_title("(d) Attention Kernel",
                 fontsize=14, fontweight="bold", pad=8)

    rng_d = np.random.default_rng(7)
    n     = 60
    x_pts = np.sort(rng_d.uniform(0, 1, n))

    # RBF kernel matrix
    sigma_rbf = 0.15
    K_rbf     = np.exp(-0.5 * ((x_pts[:, None] - x_pts[None, :]) / sigma_rbf)**2)
    K_rbf    += 1e-6 * np.eye(n)
    eigs_rbf  = np.sort(np.linalg.eigvalsh(K_rbf))[::-1]
    eigs_rbf  = eigs_rbf / eigs_rbf[0]

    # Attention kernel matrix (scaled dot-product + exponential)
    d       = 8
    W_q     = rng_d.normal(0, 1, (1, d)) / np.sqrt(d)
    W_k     = rng_d.normal(0, 1, (1, d)) / np.sqrt(d)
    Q       = x_pts[:, None] @ W_q
    Km      = x_pts[:, None] @ W_k
    scores  = Q @ Km.T / np.sqrt(d)
    K_att   = np.exp(scores)
    K_att   = (K_att + K_att.T) / 2.0
    K_att  += 1e-6 * np.eye(n)
    eigs_att = np.sort(np.linalg.eigvalsh(K_att))[::-1]
    eigs_att = eigs_att / eigs_att[0]

    kappa_rbf = eigs_rbf[0] / max(eigs_rbf[-1], 1e-12)
    kappa_att = eigs_att[0] / max(eigs_att[-1], 1e-12)

    idx = np.arange(1, n + 1)
    ax.semilogy(idx, eigs_rbf, color="#2166ac", lw=2.0,
                label=r"RBF kernel $k_{\mathrm{rbf}}$", zorder=3)
    ax.semilogy(idx, eigs_att, color="#d6604d", lw=2.0,
                linestyle="--",
                label=r"Attention kernel $k_{\mathrm{att}}$", zorder=3)

    ax.text(n * 0.60, eigs_rbf[int(n * 0.52)] * 4.0,
            fr"$\kappa={kappa_rbf:.0f}$",
            color="#2166ac", fontsize=8.5, ha="center")
    ax.text(n * 0.42, eigs_att[int(n * 0.30)] * 0.15,
            fr"$\kappa \approx {kappa_att:.1e}$",
            color="#d6604d", fontsize=8.5, ha="center")

    # ax.set_xlabel("Eigenvalue index $i$", fontsize=10.5)
    # ax.set_ylabel(r"Normalised eigenvalue $\lambda_i/\lambda_1$", fontsize=10.5)
    ax.set_xlabel("Eigenvalue index", fontsize=10.5)
    ax.set_ylabel(r"Eigenvalue", fontsize=10.5)
    ax.set_xlim(1, n)
    ax.set_ylim(1e-12, 3)
    ax.grid(alpha=0.20, linewidth=0.5, which="both")
    ax.legend(fontsize=8.5, frameon=True, loc="upper right",
              framealpha=0.90, edgecolor="0.6",
              handlelength=1.8, labelspacing=0.35, borderpad=0.45)

    ax.text(0.97, 0.08,
            "Attention kernel: large\ncondition number $\\kappa$\n"
            r"$\Rightarrow$ ill-conditioned system",
            transform=ax.transAxes, fontsize=9.5,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.32", fc="white",
                      ec="0.60", alpha=0.92))


# =============================================================================
# 6.  Save
# =============================================================================
def save_fig(fig, stem: str):
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path,          bbox_inches="tight", pad_inches=0.02)
    print(f"[Saved] {png_path.name}  +  {pdf_path.name}")


# =============================================================================
# 7.  Main
# =============================================================================
def main():
    print("=" * 60)
    print(" sc_kernel_intro_figure.py")
    print(f" Output folder: {OUT_ROOT}")
    print("=" * 60)

    # ── Locate Sionna ─────────────────────────────────────────────────────────
    sionna, rt, scene_xml, mesh_dir = find_sionna_assets()

    # ── Data: generate or load ────────────────────────────────────────────────
    data_files = [DATA_DIR / f for f in
                  ("rss_map.npy", "path_loss_map.npy",
                   "sparse_samples.npz", "meta.npz")]
    data_ready = all(f.exists() for f in data_files)

    if data_ready:
        print("\n[Data] Existing data found -- loading (skipping ray tracing).")
        rss, _, sparse_locs, sparse_vals, x_min, x_max, y_min, y_max = load_data()
    else:
        print("\n[Data] No existing data -- running Sionna RT ray tracing ...")
        rss, _, sparse_locs, sparse_vals, x_min, x_max, y_min, y_max = \
            generate_data(sionna, rt, scene_xml)

    # ── Building footprints ───────────────────────────────────────────────────
    print("\n[Buildings] Loading Munich building footprints ...")
    footprints = load_footprints(mesh_dir)

    # ── Draw combined four-panel figure ───────────────────────────────────────
    print("\n[Figure] Drawing combined four-panel figure ...")

    w_ratios = [MAP_W, MAP_W, PLOT_W, PLOT_W]
    total_w  = sum(w_ratios) + 0.30 * 3    # gap between panels

    fig = plt.figure(figsize=(total_w, FIG_HEIGHT))
    gs  = fig.add_gridspec(
        1, 4,
        width_ratios=w_ratios,
        wspace=0.30,
        left=0.01, right=0.99,
        top=0.90,  bottom=0.10,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])

    draw_panel_a(ax_a, rss, sparse_locs, sparse_vals,
                 x_min, x_max, y_min, y_max, footprints)
    draw_panel_b(ax_b, rss, sparse_locs,
                 x_min, x_max, y_min, y_max, footprints)
    draw_panel_c(ax_c, rss, x_min, x_max, y_min, y_max)
    draw_panel_d(ax_d)

    save_fig(fig, "intro_figure")
    plt.close(fig)

    # ── Draw and save individual panels ───────────────────────────────────────
    print("\n[Figure] Saving individual panels ...")
    panels = [
        ("panel_a", MAP_W,  FIG_HEIGHT,
         lambda ax: draw_panel_a(ax, rss, sparse_locs, sparse_vals,
                                 x_min, x_max, y_min, y_max, footprints)),
        ("panel_b", MAP_W,  FIG_HEIGHT,
         lambda ax: draw_panel_b(ax, rss, sparse_locs,
                                 x_min, x_max, y_min, y_max, footprints)),
        ("panel_c", PLOT_W, FIG_HEIGHT,
         lambda ax: draw_panel_c(ax, rss, x_min, x_max, y_min, y_max)),
        ("panel_d", PLOT_W, FIG_HEIGHT,
         lambda ax: draw_panel_d(ax)),
    ]
    for name, pw, ph, fn in panels:
        fig_s, ax_s = plt.subplots(figsize=(pw, ph))
        fn(ax_s)
        fig_s.tight_layout()
        save_fig(fig_s, name)
        plt.close(fig_s)

    print(f"\n[Done] All outputs written to:\n       {OUT_ROOT}")


if __name__ == "__main__":
    main()