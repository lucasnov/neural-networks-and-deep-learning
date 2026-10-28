"""Constantes e caminhos usados pelos scripts da entrega Perceptron.

Os caminhos sao derivados da posicao deste arquivo, entao os scripts rodam
a partir de qualquer diretorio de trabalho (em especial, da raiz do repositorio).
"""

from pathlib import Path

# Semente unica para toda a atividade (item "Regras tecnicas" do enunciado):
# um unico rng = np.random.default_rng(SEED) e usado em sequencia pelos dois exercicios.
SEED = 42

# .../docs/exercises/perceptron/code/common.py
_CODE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = _CODE_DIR.parent / "figures"           # docs/exercises/perceptron/figures

CLASS_COLORS = ["#2a78d6", "#eb6834"]
CLASS_MARKERS = ["o", "^"]


def savefig(fig, name: str) -> Path:
    """Salva a figura em figures/ com resolucao fixa e fecha o objeto."""
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_two_gaussians(rng, mean0, mean1, cov, n_per_class=1000):
    """Duas classes 2D gaussianas, n_per_class pontos cada.

    Os pontos sao sorteados com rng.multivariate_normal, primeiro a classe 0 e
    depois a classe 1, e empilhados nessa ordem. O enunciado nao pede embaralhamento,
    entao o treino percorre as 1000 amostras da classe 0 e depois as 1000 da classe 1.
    """
    import numpy as np

    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([X0, X1])                                  # (2000, 2)
    y = np.concatenate([np.zeros(n_per_class, dtype=int),
                        np.ones(n_per_class, dtype=int)])    # (2000,)
    return X, y


def scatter_classes(ax, X, y, size=10, alpha=0.6):
    for k in (0, 1):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=size, alpha=alpha, color=CLASS_COLORS[k],
                   marker=CLASS_MARKERS[k], edgecolor="white", linewidth=0.2,
                   label=f"Classe {k}")


def boundary_points(w, b, x1_limits):
    """Dois pontos da reta w1*x1 + w2*x2 + b = 0, para desenhar no grafico."""
    import numpy as np

    x1 = np.array(x1_limits, dtype=float)
    if abs(w[1]) < 1e-12:                  # reta vertical: x1 = -b / w1
        return np.full(2, -b / w[0]), np.array([-1e3, 1e3])
    return x1, -(w[0] * x1 + b) / w[1]
