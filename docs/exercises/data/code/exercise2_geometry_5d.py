"""Exercicio 2 - Nao-linearidade em dimensoes maiores (5D).

Constroi dois datasets 5D (gaussianas deslocadas e cascas concentricas),
projeta cada um com PCA e mede a geometria das classes. Salva as Figuras 4 e 5.

Uso (a partir da raiz do repositorio):

    python docs/exercises/data/code/exercise2_geometry_5d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import SEED, savefig

N_PER_CLASS = 500

# --- Dataset I: gaussianas multivariadas deslocadas (parametros do enunciado) ---
MU_A = np.zeros(5)
MU_B = np.full(5, 1.5)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# --- Dataset II: cascas concentricas ---
RADIUS_CORE = 2.0
RADIUS_SHELL = 5.0
RADIUS_STD = 0.4


def generate_dataset_i(rng):
    assert SIGMA_A.shape == (5, 5) and SIGMA_B.shape == (5, 5)
    xa = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    xb = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    X = np.vstack([xa, xb])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


def _radial_class(rng, radius_mean):
    v = rng.standard_normal((N_PER_CLASS, 5))
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.maximum(norm, 1e-12)          # direcao unitaria (protege contra norma 0)
    rho = rng.normal(radius_mean, RADIUS_STD, size=(N_PER_CLASS, 1))
    return rho * u


def generate_dataset_ii(rng):
    xc = _radial_class(rng, RADIUS_CORE)
    xd = _radial_class(rng, RADIUS_SHELL)
    X = np.vstack([xc, xd])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


def pca_2d(X):
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)
    return Z, pca.explained_variance_ratio_


def center_distance(X, y):
    """||mu_1 - mu_2|| entre os centros empiricos das classes, no espaco 5D."""
    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)
    return float(np.linalg.norm(c0 - c1))


def figure4(Z1, y1, evr1, Z2, y2, evr2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        (axes[0], Z1, y1, evr1, "Dataset I - gaussianas deslocadas", ("Classe A", "Classe B")),
        (axes[1], Z2, y2, evr2, "Dataset II - cascas concentricas", ("Classe C (nucleo)", "Classe D (casca)")),
    ]
    for ax, Z, y, evr, title, names in panels:
        for cls, name, color in zip((0, 1), names, ("#1f77b4", "#d62728")):
            pts = Z[y == cls]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, color=color, label=name)
        ax.set_title(f"{title}\nvariancia explicada PC1+PC2 = {evr.sum():.3f}")
        ax.set_xlabel(f"PC1 ({evr[0]:.3f})")
        ax.set_ylabel(f"PC2 ({evr[1]:.3f})")
        ax.legend()
    fig.suptitle("Figura 4 - Projecao PCA 2D dos dois datasets 5D")
    fig.tight_layout()
    return fig


def figure5(X1, y1, X2, y2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        (axes[0], X1, y1, "Dataset I - gaussianas deslocadas", ("Classe A", "Classe B")),
        (axes[1], X2, y2, "Dataset II - cascas concentricas", ("Classe C (nucleo)", "Classe D (casca)")),
    ]
    for ax, X, y, title, names in panels:
        radius = np.linalg.norm(X, axis=1)
        bins = np.linspace(radius.min(), radius.max(), 40)
        for cls, name, color in zip((0, 1), names, ("#1f77b4", "#d62728")):
            ax.hist(radius[y == cls], bins=bins, alpha=0.55, color=color, label=name)
        ax.set_title(title)
        ax.set_xlabel(r"raio $\|x\|$")
        ax.set_ylabel("contagem")
        ax.legend()
    fig.suptitle(r"Figura 5 - Distribuicao do raio $\|x\|$ por classe")
    fig.tight_layout()
    return fig


def main():
    rng = np.random.default_rng(SEED)

    X1, y1 = generate_dataset_i(rng)
    X2, y2 = generate_dataset_ii(rng)

    Z1, evr1 = pca_2d(X1)
    Z2, evr2 = pca_2d(X2)

    p4 = savefig(figure4(Z1, y1, evr1, Z2, y2, evr2), "figure4_pca_5d.png")
    p5 = savefig(figure5(X1, y1, X2, y2), "figure5_radius_histograms.png")

    dist1 = center_distance(X1, y1)
    dist2 = center_distance(X2, y2)

    # Regra radial simples para o Dataset II: g(x) = sum(x_i^2).
    # Nucleo ~ raio 2 -> g ~ 4; casca ~ raio 5 -> g ~ 25. Limiar no meio dos raios: (2+5)/2 = 3.5 -> g = 12.25.
    threshold = ((RADIUS_CORE + RADIUS_SHELL) / 2) ** 2
    g = (X2 ** 2).sum(axis=1)
    predicted_shell = g > threshold
    correct = int(np.sum(predicted_shell == (y2 == 1)))

    print("== Exercicio 2 ==")
    print(f"figuras salvas: {p4.name}, {p5.name}")
    print(f"distancia entre centros - Dataset I : {dist1:.3f}")
    print(f"distancia entre centros - Dataset II: {dist2:.3f}")
    print(f"variancia explicada PC1+PC2 - Dataset I : {evr1.sum():.3f} "
          f"(PC1={evr1[0]:.3f}, PC2={evr1[1]:.3f})")
    print(f"variancia explicada PC1+PC2 - Dataset II: {evr2.sum():.3f} "
          f"(PC1={evr2[0]:.3f}, PC2={evr2[1]:.3f})")
    print(f"regra g(x)=sum(x_i^2) > {threshold:.2f}: separa {correct}/1000 pontos do Dataset II")

    return {
        "dist_centers_i": dist1,
        "dist_centers_ii": dist2,
        "evr_i": (float(evr1[0]), float(evr1[1]), float(evr1.sum())),
        "evr_ii": (float(evr2[0]), float(evr2[1]), float(evr2.sum())),
        "radial_threshold": threshold,
        "radial_rule_correct": correct,
    }


if __name__ == "__main__":
    main()
