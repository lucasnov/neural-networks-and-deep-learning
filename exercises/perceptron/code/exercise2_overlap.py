"""Exercicio 2 - dados sobrepostos: o caso que o perceptron nao resolve.

Reutiliza train() de perceptron.py sem alteracoes (eta = 0.01, 100 epocas) e
compara os pesos finais com os do pocket. Salva as Figuras 4, 5 e 6.

O rng e o mesmo do Exercicio 1, continuado: os sorteios daqui vem depois dos
sorteios do Exercicio 1. Por isso, rodando este script sozinho, ele refaz antes
os sorteios do Exercicio 1 (sem treinar) para chegar ao mesmo ponto do gerador.

Uso (a partir da raiz do repositorio):

    python docs/exercises/perceptron/code/exercise2_overlap.py
"""

import numpy as np
import matplotlib.pyplot as plt

import exercise1_separable as ex1
from common import (CLASS_COLORS, SEED, boundary_points, make_two_gaussians,
                    savefig, scatter_classes)
from perceptron import accuracy, predict, train

MEAN0 = [3.0, 3.0]
MEAN1 = [4.0, 4.0]
COV = [[1.5, 0.0], [0.0, 1.5]]
LEARNING_RATE = 0.01
MAX_EPOCHS = 100


def figure4(X, y):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 4 - Exercicio 2: 2000 pontos, 1000 por classe")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(title="Classe")
    ax.set_aspect("equal")
    return savefig(fig, "figure4_overlap_data.png")


def figure5(X, y, run):
    """Dois paineis com as duas fronteiras; cada um marca os erros de um par (w, b)."""
    pocket = run["pocket"]
    lims_x = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    lims_y = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    panels = (("pesos finais", run["w"], run["b"], run["final_accuracy"]),
              ("pesos do pocket", pocket["w"], pocket["b"], pocket["accuracy"]))
    for ax, (name, w, b, acc) in zip(axes, panels):
        scatter_classes(ax, X, y, size=8, alpha=0.45)
        ax.plot(*boundary_points(run["w"], run["b"], lims_x), color="black",
                linewidth=2, linestyle="--", label="fronteira final")
        ax.plot(*boundary_points(pocket["w"], pocket["b"], lims_x), color="black",
                linewidth=2, label="fronteira do pocket")
        wrong = predict(X, w, b) != y
        ax.scatter(X[wrong, 0], X[wrong, 1], s=14, marker="x", color="red",
                   linewidth=0.8, label=f"mal classificados: {wrong.sum()}")
        ax.set_xlim(lims_x)
        ax.set_ylim(lims_y)
        ax.set_aspect("equal")
        ax.set_title(f"Erros dos {name} (acuracia {acc:.4f})")
        ax.set_xlabel("$x_1$")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle("Figura 5 - Fronteiras final e do pocket (Exercicio 2)")
    return savefig(fig, "figure5_overlap_boundaries.png")


def figure6(run):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, run["epochs"] + 1)
    ax.plot(epochs, run["accuracy_per_epoch"], color=CLASS_COLORS[0], linewidth=1.5,
            label="pesos atuais (fim de cada epoca)")
    ax.plot(epochs, run["best_so_far_per_epoch"], color=CLASS_COLORS[1], linewidth=2,
            drawstyle="steps-post", label="melhor ate agora (pocket)")
    ax.set_title("Figura 6 - Acuracia x epoca: pesos atuais e pocket (Exercicio 2)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Acuracia no dataset completo")
    ax.set_ylim(0.45, 0.8)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    return savefig(fig, "figure6_overlap_accuracy.png")


def main(rng):
    X, y = make_two_gaussians(rng, MEAN0, MEAN1, COV)
    w_init = rng.normal(0, 0.01, size=2)
    print(f"Exercicio 2 - X {X.shape}, y {y.shape}, classe 1: {int(y.sum())} pontos")
    print(f"w inicial = {w_init}, b inicial = 0, ||x|| medio = "
          f"{np.linalg.norm(X, axis=1).mean():.3f}")
    figure4(X, y)

    run = train(X, y, w_init, 0.0, LEARNING_RATE, MAX_EPOCHS)
    pocket = run["pocket"]
    print(f"epocas = {run['epochs']} (convergiu: {run['converged']})")
    print(f"pesos finais: w = {run['w']}, b = {run['b']:.4f}, "
          f"acuracia = {run['final_accuracy']:.4f}")
    print(f"pocket:       w = {pocket['w']}, b = {pocket['b']:.4f}, "
          f"acuracia = {pocket['accuracy']:.4f}, epoca = {pocket['epoch']}")

    # Numeros usados na analise do item D
    pred_final = predict(X, run["w"], run["b"])
    print(f"pesos finais preveem classe 1 para {pred_final.mean():.4f} dos pontos")
    for name, w, b in (("final", run["w"], run["b"]), ("pocket", pocket["w"], pocket["b"])):
        # ponto da fronteira sobre a diagonal x1 = x2 = t:  (w1 + w2) t + b = 0
        print(f"fronteira {name} cruza a diagonal x1 = x2 em t = {-b / w.sum():.3f}")
    acc_epochs = run["accuracy_per_epoch"]
    print(f"acuracia dos pesos atuais ao fim das epocas: min {min(acc_epochs):.4f}, "
          f"max {max(acc_epochs):.4f}")
    upd = run["updates_per_epoch"]
    print(f"atualizacoes por epoca: min {min(upd)}, max {max(upd)}, total {sum(upd)}")
    print(f"b ao fim das epocas: min {min(run['b_per_epoch']):.2f}, "
          f"max {max(run['b_per_epoch']):.2f}")
    norms = [np.linalg.norm(w) for w in run["w_per_epoch"]]
    print(f"||w|| ao fim das epocas: min {min(norms):.4f}, max {max(norms):.4f}")
    # Referencia geometrica (sem treino): a mediatriz entre as medias, x1 + x2 = 7.
    reference_acc = accuracy(X, y, np.array([1.0, 1.0]), -7.0)
    print(f"reta de referencia x1 + x2 = 7: acuracia = {reference_acc:.4f}")

    figure5(X, y, run)
    figure6(run)
    return {"run": run, "reference_acc": reference_acc, "w_init": w_init,
            "pred_class1_fraction": float(pred_final.mean())}


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    ex1.draw(rng)          # mesmos sorteios do Exercicio 1, para continuar o mesmo rng
    main(rng)
