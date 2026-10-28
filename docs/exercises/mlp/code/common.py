"""Constantes e caminhos usados pelos scripts da entrega MLP.

Os caminhos sao derivados da posicao deste arquivo, entao os scripts rodam
a partir de qualquer diretorio de trabalho (em especial, da raiz do repositorio).
"""

from pathlib import Path

# Semente unica para toda a atividade (item "Regras tecnicas" do enunciado):
# rng = np.random.default_rng(SEED) e random_state=SEED no sklearn.
SEED = 42

# .../docs/exercises/mlp/code/common.py
_CODE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = _CODE_DIR.parent / "figures"           # docs/exercises/mlp/figures


def savefig(fig, name: str) -> Path:
    """Salva a figura em figures/ com resolucao fixa e fecha o objeto."""
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
