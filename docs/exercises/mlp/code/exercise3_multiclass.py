"""Exercicio 3 - multiclasse, a mesma rede.

Reutiliza o MLP de mlp.py sem alterar o codigo: so mudam os argumentos
(3 unidades de saida, perda CategoricalCrossEntropy, que usa softmax).
Salva as Figuras 4 e 5.

Uso (a partir da raiz do repositorio):

    python docs/exercises/mlp/code/exercise3_multiclass.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from common import SEED, savefig
from mlp import MLP, CategoricalCrossEntropy, accuracy, one_hot, train

N_CLASSES = 3
# Hiperparametros (o Exercicio 4 importa EPOCHS e LEARNING_RATE daqui)
LAYER_SIZES = [4, 16, N_CLASSES]   # 4 entradas -> 16 tanh -> 3 saidas softmax
LEARNING_RATE = 0.5
EPOCHS = 2000


def make_data():
    """Dados exatamente como no enunciado + split 80/20 estratificado."""
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, class_sep=1.0,
                               random_state=SEED)
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def train_model(layer_sizes, X_train, y_train, rng):
    """Cria e treina uma rede; usado aqui e no Exercicio 4."""
    model = MLP(layer_sizes, CategoricalCrossEntropy, rng)
    losses = train(model, X_train, one_hot(y_train, N_CLASSES), EPOCHS, LEARNING_RATE)
    return model, losses


def figure4(losses):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#2a78d6", linewidth=2,
            label="Perda de treino (entropia cruzada)")
    ax.set_title("Figura 4 - Perda de treino x epoca (Exercicio 3)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Entropia cruzada categorica")
    ax.grid(alpha=0.3)
    ax.legend()
    return savefig(fig, "figure4_multiclass_loss.png")


def figure5(cm):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    labels = [f"Classe {k}" for k in range(N_CLASSES)]
    ax.set_xticks(range(N_CLASSES), labels)
    ax.set_yticks(range(N_CLASSES), labels)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title("Figura 5 - Matriz de confusao (teste, Ex. 3)")
    fig.colorbar(im, ax=ax, label="Numero de amostras")
    return savefig(fig, "figure5_confusion_matrix.png")


def most_confused_pair(cm):
    """Par (i, j) com mais erros somando as duas direcoes: cm[i, j] + cm[j, i]."""
    best, best_errors = None, -1
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            errors = int(cm[i, j] + cm[j, i])
            if errors > best_errors:
                best, best_errors = (i, j), errors
    return best, best_errors


def neighbor_contacts(X, y):
    """Quantas vezes o vizinho mais proximo (em 4D) de um ponto e de outra classe.

    contacts[i, j] = numero de pontos da classe i cujo vizinho mais proximo e da
    classe j. Serve para medir quais classes tem grupos encostados uns nos outros.
    Feito so com NumPy, no conjunto de treino.
    """
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)       # um ponto nao e vizinho de si mesmo
    nearest_label = y[dist.argmin(axis=1)]
    contacts = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for true_label, neighbor_label in zip(y, nearest_label):
        contacts[true_label, neighbor_label] += 1
    return contacts


def main():
    rng = np.random.default_rng(SEED)
    X_train, X_test, y_train, y_test = make_data()
    print(f"Exercicio 3 - treino {X_train.shape}, teste {X_test.shape}")

    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    baseline_acc = accuracy(y_test, baseline.predict(X_test))
    print(f"linha de base (regressao logistica): acuracia no teste = {baseline_acc:.4f}")

    model, losses = train_model(LAYER_SIZES, X_train, y_train, rng)
    train_acc = accuracy(y_train, model.predict(X_train))
    test_acc = accuracy(y_test, model.predict(X_test))
    print(f"MLP {LAYER_SIZES}, eta={LEARNING_RATE}, {EPOCHS} epocas, "
          f"{model.n_parameters} parametros")
    print(f"perda inicial {losses[0]:.4f} -> final {losses[-1]:.4f}")
    print(f"acuracia treino = {train_acc:.4f} | teste = {test_acc:.4f}")

    cm = confusion_matrix(y_test, model.predict(X_test))
    pair, errors = most_confused_pair(cm)
    print("matriz de confusao (linha = real, coluna = prevista):\n", cm)
    print(f"par mais confundido: classes {pair} com {errors} erros")

    contacts = neighbor_contacts(X_train, y_train)
    print("vizinho mais proximo no treino (linha = classe do ponto, coluna = do vizinho):\n",
          contacts)

    figure4(losses)
    figure5(cm)
    return {"baseline_acc": baseline_acc, "test_acc": test_acc, "train_acc": train_acc,
            "final_loss": losses[-1], "losses": losses, "n_params": model.n_parameters,
            "cm": cm, "pair": pair, "pair_errors": errors, "contacts": contacts}


if __name__ == "__main__":
    main()
