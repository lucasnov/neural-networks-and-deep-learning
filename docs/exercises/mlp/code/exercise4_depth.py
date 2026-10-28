"""Exercicio 4 - uma segunda camada oculta ajuda?

Comparacao controlada: mesmos dados do Exercicio 3, mesmo numero de epocas,
mesma taxa de aprendizado, mesmo codigo do MLP. So muda a profundidade.
Cada arquitetura e treinada com varias sementes de inicializacao e reportamos
a media. Salva a Figura 6.

Uso (a partir da raiz do repositorio):

    python docs/exercises/mlp/code/exercise4_depth.py
"""

import numpy as np
import matplotlib.pyplot as plt

import exercise3_multiclass as ex3
from common import SEED, savefig
from mlp import accuracy

SHALLOW_SIZES = ex3.LAYER_SIZES          # [4, 16, 3]     - rede do Exercicio 3
DEEP_SIZES = [4, 16, 16, ex3.N_CLASSES]  # [4, 16, 16, 3] - duas camadas ocultas
# A semente muda so a inicializacao dos pesos (os dados sao sempre os mesmos,
# random_state=42, e o treino e em lote completo, sem embaralhamento).
SEEDS = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]   # 42, 43, 44, 45, 46

COLORS = {"shallow": "#2a78d6", "deep": "#eb6834"}


def run(layer_sizes, X_train, X_test, y_train, y_test):
    """Treina a arquitetura uma vez por semente; cada semente tem seu proprio rng."""
    runs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        model, losses = ex3.train_model(layer_sizes, X_train, y_train, rng)
        runs.append({
            "seed": seed,
            "losses": losses,
            "final_loss": losses[-1],
            "train_acc": accuracy(y_train, model.predict(X_train)),
            "test_acc": accuracy(y_test, model.predict(X_test)),
            "n_params": model.n_parameters,
        })
    return runs


def summarize(runs):
    return {
        "n_params": runs[0]["n_params"],
        "final_loss_mean": float(np.mean([r["final_loss"] for r in runs])),
        "train_acc_mean": float(np.mean([r["train_acc"] for r in runs])),
        "test_acc_mean": float(np.mean([r["test_acc"] for r in runs])),
        "test_acc_std": float(np.std([r["test_acc"] for r in runs])),
    }


def figure6(shallow_runs, deep_runs):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, ex3.EPOCHS + 1)
    for runs, key, label in ((shallow_runs, "shallow", f"Ex. 3 - {SHALLOW_SIZES}"),
                             (deep_runs, "deep", f"Ex. 4 - {DEEP_SIZES}")):
        # Semente 42 em destaque; as outras sementes em linhas finas.
        for r in runs[1:]:
            ax.plot(epochs, r["losses"], color=COLORS[key], linewidth=0.8, alpha=0.35)
        ax.plot(epochs, runs[0]["losses"], color=COLORS[key], linewidth=2,
                label=f"{label} (semente 42)")
    ax.plot([], [], color="gray", linewidth=0.8, alpha=0.6,
            label="linhas finas: sementes 43-46 (mesma cor)")
    ax.set_title("Figura 6 - Perda de treino: 1 camada oculta x 2 camadas ocultas")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Entropia cruzada categorica (treino)")
    ax.grid(alpha=0.3)
    ax.legend()
    return savefig(fig, "figure6_depth_loss_comparison.png")


def main():
    X_train, X_test, y_train, y_test = ex3.make_data()
    print(f"Exercicio 4 - {ex3.EPOCHS} epocas, eta={ex3.LEARNING_RATE}, sementes {SEEDS}")

    shallow_runs = run(SHALLOW_SIZES, X_train, X_test, y_train, y_test)
    deep_runs = run(DEEP_SIZES, X_train, X_test, y_train, y_test)

    for name, sizes, runs in (("Ex.3", SHALLOW_SIZES, shallow_runs),
                              ("Ex.4", DEEP_SIZES, deep_runs)):
        for r in runs:
            print(f"{name} {sizes} semente {r['seed']}: perda final {r['final_loss']:.4f} "
                  f"| treino {r['train_acc']:.4f} | teste {r['test_acc']:.4f}")

    shallow = summarize(shallow_runs)
    deep = summarize(deep_runs)
    diff = deep["test_acc_mean"] - shallow["test_acc_mean"]
    print("\nmedias:")
    for name, s in (("Ex.3", shallow), ("Ex.4", deep)):
        print(f"{name}: {s['n_params']} parametros | perda final {s['final_loss_mean']:.4f} | "
              f"treino {s['train_acc_mean']:.4f} | teste {s['test_acc_mean']:.4f} "
              f"(desvio {s['test_acc_std']:.4f})")
    print(f"diferenca de acuracia no teste (Ex.4 - Ex.3): {diff:+.4f}")

    figure6(shallow_runs, deep_runs)
    return {"shallow": shallow, "deep": deep, "diff": diff,
            "deep_seed42_test_acc": deep_runs[0]["test_acc"],
            "deep_seed42_final_loss": deep_runs[0]["final_loss"],
            "shallow_runs": shallow_runs, "deep_runs": deep_runs}


if __name__ == "__main__":
    main()
