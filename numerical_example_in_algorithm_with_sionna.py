import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


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
        print("        Install with:  pip install sionna-rt plyfile numpy matplotlib scipy")
        sys.exit(1)

    sionna_root = Path(sionna.__file__).resolve().parent
    scene_xml   = sionna_root / "rt" / "scenes" / "munich" / "munich.xml"
    mesh_dir    = sionna_root / "rt" / "scenes" / "munich" / "meshes"

    if not scene_xml.exists():
        print(f"\n[ERROR] Munich scene not found at:\n  {scene_xml}")
        print("        Try:  pip install --force-reinstall sionna-rt")
        sys.exit(1)

    return sionna, rt, scene_xml, mesh_dir


# =============================================================================
# 2.  Sionna RT data generation
# =============================================================================
CARRIER_FREQ   = 3.5e9
TX_POWER_DBM   = 30.0
TX_POSITION    = [8.5, 21.0, 30.0]
N_SPARSE       = 40
CELL_SIZE_M    = 5.0
SAMPLES_PER_TX = int(5e5)   # reduce to 2e5 if GPU RAM < 8 GB
MAX_DEPTH      = 5
_RNG           = np.random.default_rng(42)


def generate_data(sionna, rt, scene_xml: Path):
    from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver

    print(f"[Sionna RT v{rt.__version__}] Loading Munich scene ...")
    scene           = load_scene(str(scene_xml))
    scene.frequency = CARRIER_FREQ

    _ant = dict(num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="dipole", polarization="V")
    scene.tx_array = PlanarArray(**_ant)
    scene.rx_array = PlanarArray(**_ant)

    tx           = Transmitter(name="bs0", position=TX_POSITION)
    tx.power_dbm = TX_POWER_DBM
    scene.add(tx)
    print(f"[TX] position={TX_POSITION}, power={TX_POWER_DBM} dBm, "
          f"freq={CARRIER_FREQ/1e9:.1f} GHz")

    print(f"[Solver] Computing radio map "
          f"(cell={CELL_SIZE_M} m, samples={SAMPLES_PER_TX:.0e}) ...")
    print("         This may take 1-3 min on GPU, longer on CPU.")
    solver    = RadioMapSolver()
    radio_map = solver(
        scene,
        cell_size          = [CELL_SIZE_M, CELL_SIZE_M],
        samples_per_tx     = SAMPLES_PER_TX,
        max_depth          = MAX_DEPTH,
        diffuse_reflection = True,
    )
    print("[Solver] Done.")

    cc   = np.array(radio_map.cell_centers)   # [H, W, 3]
    x_2d = cc[:, :, 0]
    y_2d = cc[:, :, 1]

    pg           = np.array(radio_map.path_gain)[0]
    pg           = np.where(pg > 0, pg, np.nan)
    path_loss_db = -10.0 * np.log10(pg)
    rss_dbm      = TX_POWER_DBM - path_loss_db

    x_min, x_max = float(x_2d.min()), float(x_2d.max())
    y_min, y_max = float(y_2d.min()), float(y_2d.max())

    print(f"[Map] shape={rss_dbm.shape}, "
          f"RSS=[{np.nanmin(rss_dbm):.1f}, {np.nanmax(rss_dbm):.1f}] dBm")

    H, W    = rss_dbm.shape
    valid   = np.argwhere(np.isfinite(rss_dbm))
    chosen  = _RNG.choice(len(valid), size=min(N_SPARSE, len(valid)),
                          replace=False)
    pixels  = valid[chosen]
    sx      = x_2d[pixels[:, 0], pixels[:, 1]]
    sy      = y_2d[pixels[:, 0], pixels[:, 1]]
    s_locs  = np.stack([sx, sy], axis=1)
    s_vals  = rss_dbm[pixels[:, 0], pixels[:, 1]]

    np.save(DATA_DIR / "rss_map.npy",       rss_dbm)
    np.save(DATA_DIR / "path_loss_map.npy", path_loss_db)
    np.savez(DATA_DIR / "meta.npz",
             x_min=x_min, x_max=x_max,
             y_min=y_min, y_max=y_max,
             carrier_freq=CARRIER_FREQ,
             tx_power_dbm=TX_POWER_DBM,
             tx_position=np.array(TX_POSITION),
             cell_size=CELL_SIZE_M,
             scene="munich")
    np.savez(DATA_DIR / "sparse_samples.npz",
             locations=s_locs, values=s_vals)

    print(f"[Saved] Data written to: {DATA_DIR}")
    return rss_dbm, s_locs, s_vals, x_min, x_max, y_min, y_max


def load_data():
    rss    = np.load(DATA_DIR / "rss_map.npy")
    meta   = np.load(DATA_DIR / "meta.npz")
    sparse = np.load(DATA_DIR / "sparse_samples.npz")
    return (rss,
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
        print("\n[WARN] plyfile not installed -- buildings will not be shown.")
        print("       Install with:  pip install plyfile")
        return None

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

    print(f"[Buildings] {len(boxes)} footprints loaded.")
    return np.array(boxes) if boxes else None


# =============================================================================
# 4.  Kernel regression
# =============================================================================
LAMBD      = 0.1
TRAIN_IDX  = [0, 15, 35]   # indices into sparse_samples (n=40)
QUERY_IDX  = 25             # index of query point x*


def build_embeddings(locs, x_min, x_max, y_min, y_max):
    e = locs.astype(float).copy()
    e[:, 0] = (locs[:, 0] - x_min) / (x_max - x_min) - 0.5
    e[:, 1] = (locs[:, 1] - y_min) / (y_max - y_min) - 0.5
    return e * 2.0


def attn_kernel(e1, e2):
    return np.exp(e1 @ e2.T)


def reconstruct_full_map(rss, locs, vals, x_min, x_max, y_min, y_max):
    e_train = build_embeddings(locs, x_min, x_max, y_min, y_max)
    G_train = attn_kernel(e_train, e_train)
    A       = G_train + LAMBD * np.eye(len(locs))
    alpha   = np.linalg.solve(A, vals)

    H, W    = rss.shape
    x_g     = np.linspace(x_min, x_max, W)
    y_g     = np.linspace(y_min, y_max, H)
    xx, yy  = np.meshgrid(x_g, y_g)
    g_locs  = np.stack([xx.ravel(), yy.ravel()], axis=1)
    e_grid  = build_embeddings(g_locs, x_min, x_max, y_min, y_max)

    batch  = 2000
    r_hat  = np.zeros(H * W)
    for s in range(0, H * W, batch):
        end          = min(s + batch, H * W)
        r_hat[s:end] = attn_kernel(e_grid[s:end], e_train) @ alpha

    return r_hat.reshape(H, W), alpha


def compute_toy_numbers(locs, vals, x_min, x_max, y_min, y_max):
    e_all   = build_embeddings(locs, x_min, x_max, y_min, y_max)
    e_train = e_all[TRAIN_IDX]
    e_query = e_all[QUERY_IDX]
    y_train = vals[TRAIN_IDX]

    G_train = attn_kernel(e_train, e_train)
    A       = G_train + LAMBD * np.eye(3)
    alpha_3 = np.linalg.solve(A, y_train)
    G_star  = attn_kernel(e_train, e_query[None, :]).ravel()
    r_star  = float(G_star @ alpha_3)

    print("\n" + "="*55)
    print("  Numbers for LaTeX toy example")
    print("="*55)
    for k, i in enumerate(TRAIN_IDX):
        print(f"  x_{k+1} = ({locs[i,0]:.1f}, {locs[i,1]:.1f}) m  "
              f"e_{k+1} = ({e_train[k,0]:.3f}, {e_train[k,1]:.3f})")
    print(f"  x* = ({locs[QUERY_IDX,0]:.1f}, {locs[QUERY_IDX,1]:.1f}) m  "
          f"e* = ({e_query[0]:.3f}, {e_query[1]:.3f})")
    print(f"  y     = {np.round(y_train, 2)} dBm")
    print(f"  G     =\n{np.round(G_train, 3)}")
    print(f"  lI+G  =\n{np.round(A, 3)}")
    print(f"  alpha = {np.round(alpha_3, 3)}")
    print(f"  G(x*,x_i) = {np.round(G_star, 3)}")
    print(f"  r_hat(x*) = {r_star:.2f} dBm  "
          f"(ground truth: {vals[QUERY_IDX]:.2f} dBm)")
    print("="*55)

    return e_train, e_query, alpha_3, G_star, r_star


# =============================================================================
# 5.  Figure style constants
# =============================================================================
plt.rcParams.update({
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
    "font.family":    "DejaVu Sans",
    "axes.linewidth": 0.75,
    "font.size":      11,
})

C_TRAIN = "#e31a1c"   # red    -- training points
C_QUERY = "#ff7f00"   # orange -- query point
C_OTHER = "#666666"   # grey   -- other sparse samples
MAP_ASPECT = 0.817    # H/W of Munich radio map
FIG_H      = 4.6
MAP_W      = FIG_H / MAP_ASPECT


# =============================================================================
# 6.  Panel drawing functions
# =============================================================================

def draw_panel_a(ax, rss, locs, footprints,
                 x_min, x_max, y_min, y_max):
    ax.set_title("(a) Sparse Measurements",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Faint RSS background
    rss_disp = np.where(np.isfinite(rss), rss, np.nanmin(rss))
    ax.imshow(rss_disp,
              extent=[x_min, x_max, y_min, y_max],
              origin="lower", cmap="YlOrRd_r",
              vmin=np.nanpercentile(rss, 2),
              vmax=np.nanpercentile(rss, 98),
              interpolation="bilinear", alpha=0.30, zorder=1)

    # Building footprints
    if footprints is not None:
        for (bx, by, bw, bh) in footprints:
            if bx+bw < x_min or bx > x_max or by+bh < y_min or by > y_max:
                continue
            ax.add_patch(Rectangle((bx, by), bw, bh,
                                   facecolor="#d0d5db", edgecolor="#9aa0a8",
                                   linewidth=0.30, alpha=0.65, zorder=2))

    # Other sparse samples (grey)
    other_idx = [i for i in range(len(locs))
                 if i not in TRAIN_IDX and i != QUERY_IDX]
    ax.scatter(locs[other_idx, 0], locs[other_idx, 1],
               s=18, color=C_OTHER, edgecolors="white",
               linewidths=0.4, zorder=4, alpha=0.7)

    # Training points (red)
    ax.scatter(locs[TRAIN_IDX, 0], locs[TRAIN_IDX, 1],
               s=80, color=C_TRAIN, edgecolors="white",
               linewidths=0.7, zorder=6, marker="o")
    for k, i in enumerate(TRAIN_IDX):
        ax.text(locs[i, 0] + 18, locs[i, 1] + 18,
                fr"$\mathbf{{x}}_{k+1}$",
                fontsize=9, color=C_TRAIN,
                fontweight="bold", zorder=7)

    # Query point (orange star)
    ax.scatter(locs[QUERY_IDX, 0], locs[QUERY_IDX, 1],
               s=120, color=C_QUERY, edgecolors="white",
               linewidths=0.7, zorder=6, marker="*")
    ax.text(locs[QUERY_IDX, 0] + 18, locs[QUERY_IDX, 1] + 18,
            r"$\mathbf{x}^{\star}$",
            fontsize=9, color=C_QUERY,
            fontweight="bold", zorder=7)

    # Legend
    leg = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=C_TRAIN, markersize=8,
               label=r"Training $\mathbf{x}_i$"),
        Line2D([0],[0], marker="*", color="w",
               markerfacecolor=C_QUERY, markersize=10,
               label=r"Query $\mathbf{x}^\star$"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=C_OTHER, markersize=6,
               alpha=0.7, label="Other samples"),
    ]
    ax.legend(handles=leg, fontsize=8, loc="lower left",
              framealpha=0.90, edgecolor="0.6",
              handlelength=1.0, borderpad=0.5, labelspacing=0.35)

    ax.text(0.98, 0.98, "NVIDIA Sionna RT\nMunich, 3.5 GHz",
            transform=ax.transAxes, fontsize=7.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="0.6", alpha=0.90), zorder=8)


def draw_panel_b(ax, rss, r_hat, locs, footprints,
                 x_min, x_max, y_min, y_max, r_star):
    ax.set_title("(b) Reconstructed Radio Map",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    r_disp = np.where(np.isfinite(rss), r_hat, np.nanmin(r_hat))
    im = ax.imshow(r_disp,
                   extent=[x_min, x_max, y_min, y_max],
                   origin="lower", cmap="turbo",
                   vmin=np.nanpercentile(rss, 2),
                   vmax=np.nanpercentile(rss, 98),
                   interpolation="bilinear", zorder=1)

    # Contours
    H, W   = r_disp.shape
    xg     = np.linspace(x_min, x_max, W)
    yg     = np.linspace(y_min, y_max, H)
    xx, yy = np.meshgrid(xg, yg)
    lvls   = np.linspace(np.nanpercentile(rss, 8),
                         np.nanpercentile(rss, 92), 7)
    ax.contour(xx, yy, r_disp, levels=lvls,
               colors="white", linewidths=0.45, alpha=0.55, zorder=2)

    # Building outlines
    if footprints is not None:
        for (bx, by, bw, bh) in footprints:
            if bx+bw < x_min or bx > x_max or by+bh < y_min or by > y_max:
                continue
            ax.add_patch(Rectangle((bx, by), bw, bh,
                                   fill=False, edgecolor="white",
                                   linewidth=0.25, alpha=0.35, zorder=3))

    # Training points
    ax.scatter(locs[TRAIN_IDX, 0], locs[TRAIN_IDX, 1],
               s=70, color=C_TRAIN, edgecolors="white",
               linewidths=0.6, zorder=5, marker="o")

    # Query point + annotation
    qx, qy = float(locs[QUERY_IDX, 0]), float(locs[QUERY_IDX, 1])
    ax.scatter(qx, qy, s=110, color=C_QUERY,
               edgecolors="white", linewidths=0.7,
               zorder=5, marker="*")
    ax.annotate(
        fr"$\hat{{r}}(\mathbf{{x}}^\star)\approx{r_star:.1f}$ dBm",
        xy=(qx, qy),
        xytext=(qx + 120, qy - 130),
        fontsize=8, color="white",
        arrowprops=dict(arrowstyle="->", color="white",
                        lw=0.8, connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.25", fc="#333333",
                  ec="white", alpha=0.82),
        zorder=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label("RSS (dBm)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    ax.text(0.98, 0.98,
            r"$\hat{r}(\mathbf{x})=\sum_i G(\mathbf{x},\mathbf{x}_i)\,\alpha_i$",
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec="0.6", alpha=0.92), zorder=6)


# =============================================================================
# 7.  Save
# =============================================================================
def save_fig(fig, stem: str):
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf,           bbox_inches="tight", pad_inches=0.02)
    print(f"[Saved] {png.name}  +  {pdf.name}")


# =============================================================================
# 8.  Main
# =============================================================================
def main():
    print("=" * 60)
    print(" numerical_example_in_algorithm_with_sionna.py")
    print(f" Output folder: {OUT_ROOT}")
    print("=" * 60)


    sionna, rt, scene_xml, mesh_dir = find_sionna_assets()


    data_files = [DATA_DIR / f for f in
                  ("rss_map.npy", "path_loss_map.npy",
                   "sparse_samples.npz", "meta.npz")]

    if all(f.exists() for f in data_files):
        print("\n[Data] Existing data found -- loading (skipping ray tracing).")
        rss, locs, vals, x_min, x_max, y_min, y_max = load_data()
    else:
        print("\n[Data] No existing data -- running Sionna RT ray tracing ...")
        rss, locs, vals, x_min, x_max, y_min, y_max = \
            generate_data(sionna, rt, scene_xml)


    print("\n[Buildings] Loading Munich building footprints ...")
    footprints = load_footprints(mesh_dir)


    print("\n[Solve] Running kernel regression on full grid ...")
    r_hat, alpha_full = reconstruct_full_map(
        rss, locs, vals, x_min, x_max, y_min, y_max)
    print(f"        r_hat range: [{r_hat.min():.1f}, {r_hat.max():.1f}] dBm")


    e_train, e_query, alpha_3, G_star, r_star = compute_toy_numbers(
        locs, vals, x_min, x_max, y_min, y_max)


    print("\n[Figure] Drawing two-panel toy example figure ...")
    fig = plt.figure(figsize=(MAP_W * 2 + 0.5, FIG_H))
    gs  = fig.add_gridspec(1, 2, wspace=0.16,
                            left=0.01, right=0.99,
                            top=0.91, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_panel_a(ax_a, rss, locs, footprints,
                 x_min, x_max, y_min, y_max)
    draw_panel_b(ax_b, rss, r_hat, locs, footprints,
                 x_min, x_max, y_min, y_max, r_star)

    save_fig(fig, "numerical_example_in_algorithm_with_sionna")
    plt.close(fig)

    print(f"\n[Done] All outputs written to:\n       {OUT_ROOT}")


if __name__ == "__main__":
    main()