"""Exercicio 1 - dados separaveis: o caso para o qual o perceptron foi projetado.

Gera os dados, treina o perceptron com eta = 0.01 e repete com eta = 1.0
(mesmos dados e mesmo w inicial). Tambem confere numericamente a analise do
item D partindo de w = 0, b = 0. Salva as Figuras 1, 2 e 3.

Uso (a partir da raiz do repositorio):

    python docs/exercises/perceptron/code/exercise1_separable.py
"""

import numpy as np
import matplotlib.pyplot as plt

from common import (CLASS_COLORS, SEED, boundary_points, make_two_gaussians,
                    savefig, scatter_classes)
from perceptron import predict, train

MEAN0 = [1.5, 1.5]
MEAN1 = [5.0, 5.0]
COV = [[0.5, 0.0], [0.0, 0.5]]
LEARNING_RATE = 0.01
LEARNING_RATE_ALT = 1.0      # re-execucao do item D
MAX_EPOCHS = 100


def draw(rng):
    """Todos os sorteios do Exercicio 1, nesta ordem: dados e depois w inicial."""
    X, y = make_two_gaussians(rng, MEAN0, MEAN1, COV)
    w_init = rng.normal(0, 0.01, size=2)
    return X, y, w_init


def direction(w):
    """w / ||w||: a direcao do vetor de pesos (normal a fronteira)."""
    return w / np.linalg.norm(w)


def figure1(X, y):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 1 - Exercicio 1: 2000 pontos, 1000 por classe")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(title="Classe")
    ax.set_aspect("equal")
    return savefig(fig, "figure1_separable_data.png")


def figure2(X, y, run, run_alt):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter_classes(ax, X, y)
    lims = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.plot(*boundary_points(run["w"], run["b"], lims), color="black", linewidth=2,
            label=f"fronteira, $\\eta$ = {LEARNING_RATE}")
    ax.plot(*boundary_points(run_alt["w"], run_alt["b"], lims), color="black",
            linewidth=1.2, linestyle="--", label=f"fronteira, $\\eta$ = {LEARNING_RATE_ALT}")
    wrong = predict(X, run["w"], run["b"]) != y
    ax.scatter(X[wrong, 0], X[wrong, 1], s=90, facecolors="none", edgecolors="red",
               linewidth=1.5, label=f"mal classificados ($\\eta$ = 0.01): {wrong.sum()}")
    ax.set_xlim(lims)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_title("Figura 2 - Fronteira de decisao $w \\cdot x + b = 0$ (Exercicio 1)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    ax.set_aspect("equal")
    return savefig(fig, "figure2_separable_boundary.png")


def figure3(run, run_alt):
    fig, ax = plt.subplots(figsize=(7, 4))
    for r, eta, color, style in ((run, LEARNING_RATE, CLASS_COLORS[0], "-"),
                                 (run_alt, LEARNING_RATE_ALT, CLASS_COLORS[1], "--")):
        epochs = np.arange(1, r["epochs"] + 1)
        ax.plot(epochs, r["accuracy_per_epoch"], color=color, linestyle=style,
                linewidth=2, marker="o", markersize=4, label=f"$\\eta$ = {eta}")
    ax.set_title("Figura 3 - Acuracia no dataset completo x epoca (Exercicio 1)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Acuracia")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    return savefig(fig, "figure3_separable_accuracy.png")


def main(rng):
    X, y, w_init = draw(rng)
    print(f"Exercicio 1 - X {X.shape}, y {y.shape}, classe 1: {int(y.sum())} pontos")
    print(f"w inicial = {w_init}, b inicial = 0")
    figure1(X, y)

    # C - treino principal, eta = 0.01
    run = train(X, y, w_init, 0.0, LEARNING_RATE, MAX_EPOCHS)
    # D - mesma coisa com eta = 1.0 (mesmos dados, mesmo w inicial)
    run_alt = train(X, y, w_init, 0.0, LEARNING_RATE_ALT, MAX_EPOCHS)

    for eta, r in ((LEARNING_RATE, run), (LEARNING_RATE_ALT, run_alt)):
        w, b = r["w"], r["b"]
        print(f"\neta = {eta}: w = {w}, b = {b:.4f}")
        print(f"  epocas = {r['epochs']} (convergiu: {r['converged']}), "
              f"acuracia final = {r['final_accuracy']:.4f}")
        print(f"  direcao w/||w|| = {direction(w)}, b/||w|| = {b / np.linalg.norm(w):.4f}")
        print(f"  atualizacoes por epoca = {r['updates_per_epoch']} "
              f"(total {sum(r['updates_per_epoch'])})")
        print(f"  b ao fim de cada epoca = {np.round(r['b_per_epoch'], 2).tolist()}")
        print(f"  acuracia por epoca = {np.round(r['accuracy_per_epoch'], 4).tolist()}")
    first_block = [i for (e, i) in run["update_positions"] if i < 1000]
    print(f"\neta = 0.01: indices das amostras que geraram atualizacao "
          f"(classe 0 = 0..999): {[i for (_, i) in run['update_positions']]}")
    print(f"  {len(first_block)} atualizacoes vieram da classe 0, "
          f"{len(run['update_positions']) - len(first_block)} da classe 1")
    cos = float(direction(run["w"]) @ direction(run_alt["w"]))
    print(f"cos(angulo entre as direcoes) = {cos:.6f} "
          f"-> angulo = {np.degrees(np.arccos(min(cos, 1.0))):.3f} graus")

    # D - conferencia numerica da algebra: partindo de w = 0, b = 0
    zero_small = train(X, y, np.zeros(2), 0.0, LEARNING_RATE, MAX_EPOCHS)
    zero_big = train(X, y, np.zeros(2), 0.0, LEARNING_RATE_ALT, MAX_EPOCHS)
    ratio_w = zero_big["w"] / zero_small["w"]
    ratio_b = zero_big["b"] / zero_small["b"]
    print(f"\npartindo de w = 0, b = 0:")
    print(f"  eta = 0.01: w = {zero_small['w']}, b = {zero_small['b']:.4f}, "
          f"epocas = {zero_small['epochs']}")
    print(f"  eta = 1.0 : w = {zero_big['w']}, b = {zero_big['b']:.4f}, "
          f"epocas = {zero_big['epochs']}")
    print(f"  razao w (1.0 / 0.01) = {ratio_w}, razao b = {ratio_b:.4f}")
    same_updates = zero_small["update_positions"] == zero_big["update_positions"]
    print(f"  mesmas amostras atualizadas, na mesma ordem: {same_updates}")

    figure2(X, y, run, run_alt)
    figure3(run, run_alt)
    return {"run": run, "run_alt": run_alt, "zero_small": zero_small,
            "zero_big": zero_big, "cos": cos, "w_init": w_init}


if __name__ == "__main__":
    main(np.random.default_rng(SEED))
