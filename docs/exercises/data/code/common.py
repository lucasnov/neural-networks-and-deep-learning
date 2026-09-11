"""Constantes e caminhos usados pelos três scripts da entrega Data.

Os caminhos são derivados da posição deste arquivo, então os scripts rodam
a partir de qualquer diretório de trabalho (em especial, da raiz do repositório).
"""

from pathlib import Path

# Semente única para toda a atividade (item "Regras técnicas" do enunciado).
SEED = 42

# .../docs/exercises/data/code/common.py
_CODE_DIR = Path(__file__).resolve().parent
DATA_EXERCISE_DIR = _CODE_DIR.parent                 # docs/exercises/data
FIGURES_DIR = DATA_EXERCISE_DIR / "figures"
REPO_ROOT = _CODE_DIR.parents[3]                     # raiz do repositório
DATASET = REPO_ROOT / "data" / "spaceship-titanic" / "train.csv"


def load_spaceship_titanic():
    """Lê o train.csv do Spaceship Titanic, com mensagem clara se ele faltar."""
    import pandas as pd

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em: {DATASET}\n"
            "Baixe o train.csv do Spaceship Titanic (Kaggle) e salve nesse caminho."
        )
    return pd.read_csv(DATASET)


def savefig(fig, name: str) -> Path:
    """Salva a figura em figures/ com resolucao fixa e fecha o objeto."""
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
