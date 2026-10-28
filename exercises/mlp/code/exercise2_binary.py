"""Exercicio 2 - classificacao binaria que uma reta nao resolve.

Gera os dados do enunciado, mede a linha de base linear (regressao logistica),
treina o MLP de mlp.py e salva as Figuras 1, 2 e 3.

Uso (a partir da raiz do repositorio):

    python docs/exercises/mlp/code/exercise2_binary.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import SEED, savefig
from mlp import MLP, BinaryCrossEntropy, accuracy, train

# Hiperparametros (fixados antes de olhar o teste; ver relatorio)
LAYER_SIZES = [2, 8, 1]      # 2 entradas -> 8 neuronios tanh -> 1 saida sigmoide
LEARNING_RATE = 0.5
EPOCHS = 2000

CLASS_COLORS = ["#2a78d6", "#eb6834"]
CLASS_MARKERS = ["o", "^"]


def make_data():
    """Dados exatamente como no enunciado + split 80/20 estratificado."""
    centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                      cluster_std=1.2, random_state=SEED)
    y = (c > 0).astype(int)      # classe 0: 1 grupo | classe 1: 2 grupos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    return X, y, X_train, X_test, y_train, y_test


def scatter_classes(ax, X, y):
    for k in (0, 1):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.7, color=CLASS_COLORS[k],
                   marker=CLASS_MARKERS[k], edgecolor="white", linewidth=0.3,
                   label=f"Classe {k}")


def figure1(X, y):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 1 - Os 1000 pontos do Exercicio 2, coloridos por classe")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(title="Classe y")
    ax.set_aspect("equal")
    return savefig(fig, "figure1_binary_data.png")


def figure2(losses):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, color=CLASS_COLORS[0],
            linewidth=2, label="Perda de treino (BCE)")
    ax.set_title("Figura 2 - Perda de treino x epoca (Exercicio 2)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Entropia cruzada binaria")
    ax.grid(alpha=0.3)
    ax.legend()
    return savefig(fig, "figure2_binary_loss.png")


def figure3(model, X_all, X_test, y_test):
    """Avalia a rede em uma grade e sombreia as regioes de cada classe."""
    pad = 1.0
    xs = np.linspace(X_all[:, 0].min() - pad, X_all[:, 0].max() + pad, 300)
    ys = np.linspace(X_all[:, 1].min() - pad, X_all[:, 1].max() + pad, 300)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    region = model.predict(grid).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.contourf(gx, gy, region, levels=[-0.5, 0.5, 1.5], colors=CLASS_COLORS, alpha=0.18)
    ax.contour(gx, gy, region, levels=[0.5], colors="black", linewidths=1)
    scatter_classes(ax, X_test, y_test)
    ax.set_title("Figura 3 - Fronteira de decisao do MLP sobre os pontos de teste")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(title="Classe real (teste)\nfundo = previsao da rede", loc="upper left")
    ax.set_aspect("equal")
    return savefig(fig, "figure3_binary_boundary.png")


def main():
    rng = np.random.default_rng(SEED)
    X, y, X_train, X_test, y_train, y_test = make_data()
    print(f"Exercicio 2 - treino {X_train.shape}, teste {X_test.shape}")
    figure1(X, y)

    # B - linha de base: uma unica fronteira linear
    baseline = LogisticRegression().fit(X_train, y_train)
    baseline_acc = accuracy(y_test, baseline.predict(X_test))
    print(f"linha de base (regressao logistica): acuracia no teste = {baseline_acc:.4f}")
    proba = baseline.predict_proba(X_test)[:, 1]
    print(f"  pesos {baseline.coef_[0].round(4)}, bias {baseline.intercept_[0]:.4f}, "
          f"P(classe 1) no teste entre {proba.min():.3f} e {proba.max():.3f}")

    # C - MLP do zero
    model = MLP(LAYER_SIZES, BinaryCrossEntropy, rng)
    Y_train = y_train.reshape(-1, 1).astype(float)   # formato (N, 1)
    losses = train(model, X_train, Y_train, EPOCHS, LEARNING_RATE)
    train_acc = accuracy(y_train, model.predict(X_train))
    test_acc = accuracy(y_test, model.predict(X_test))
    print(f"MLP {LAYER_SIZES}, eta={LEARNING_RATE}, {EPOCHS} epocas, "
          f"{model.n_parameters} parametros")
    print(f"perda inicial {losses[0]:.4f} -> final {losses[-1]:.4f}")
    print(f"acuracia treino = {train_acc:.4f} | teste = {test_acc:.4f}")

    figure2(losses)
    figure3(model, X, X_test, y_test)
    return {"baseline_acc": baseline_acc, "test_acc": test_acc, "train_acc": train_acc,
            "first_loss": losses[0], "final_loss": losses[-1],
            "n_params": model.n_parameters}


if __name__ == "__main__":
    main()
