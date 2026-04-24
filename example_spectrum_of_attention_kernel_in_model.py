import numpy as np
import matplotlib.pyplot as plt
import os
import inspect


try:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    current_dir = os.path.dirname(__file__)
except NameError:
    script_name = "toy_example"
    current_dir = os.getcwd()

K = 5  # number of clusters
d = 100  # embedding dimension
lambd = 0.01  # regularization
n_list = [500, 2000, 8000]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "mathtext.fontset": "cm",
})

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

for i, n in enumerate(n_list):
    m = n // K
    np.random.seed(42)

    # Generate clustered embeddings
    A = np.zeros((n, d))
    for k in range(K):
        center = np.random.randn(d)
        center /= np.linalg.norm(center)
        cluster_points = center + 0.1 * np.random.randn(m, d)
        cluster_points /= np.linalg.norm(cluster_points, axis=1, keepdims=True)
        A[k * m:(k + 1) * m, :] = cluster_points

    # Exponential attention kernel
    G = np.exp(A @ A.T)
    # G = np.exp(gamma * (A @ A.T))
    Kmat = G + lambd * np.eye(n)

    # Spectrum
    evals = np.linalg.eigvalsh(Kmat)
    evals = np.sort(evals)[::-1]
    cond = evals[0] / evals[-1]

    # Plot
    axes[i].semilogy(
        evals,
        linewidth=2.5,
        color="#1f77b4",
        alpha=0.85
    )

    axes[i].set_title(
        rf"Spectrum at $n={n}$",
        pad=12,
        fontweight="bold"
    )
    axes[i].set_xlabel("Eigenvalue index", fontweight="bold")

    if i == 0:
        axes[i].set_ylabel("Eigenvalue (log scale)", fontweight="bold")

    axes[i].grid(True, which="major", linestyle="--", alpha=0.5)
    axes[i].minorticks_off()

    # axes[i].grid(True, which="major", linestyle="--", alpha=0.5, linewidth=1.2)
    # axes[i].grid(True, which="minor", linestyle=":", alpha=0.3)

    # Condition number
    axes[i].annotate(
        rf"$\kappa \approx {cond:.1e}$",
        xy=(0.11, 0.88),
        xycoords="axes fraction",
        fontsize=20,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="white",
            ec="gray",
            alpha=0.95,
            linewidth=1.5
        )
    )

    axes[i].grid(True, which="major", linestyle="--", alpha=0.5)
    axes[i].minorticks_off()

    # axes[i].tick_params(axis='both', which='major', length=6, width=1.2)
    # axes[i].tick_params(axis='both', which='minor', length=4, width=1)

plt.tight_layout(rect=[0, 0, 1, 0.97])

base_name = os.path.splitext(os.path.basename(__file__))[0]
out_dir = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
                       f"{base_name}_results")
os.makedirs(out_dir, exist_ok=True)

file_base = os.path.join(out_dir, f"{script_name}_spectrum_scaling")
plt.savefig(f"{file_base}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{file_base}.pdf", format="pdf", bbox_inches="tight")

print(f"[Output] Figures saved to:\n  {file_base}.(png / pdf)")

plt.show()
